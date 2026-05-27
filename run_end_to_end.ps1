$ErrorActionPreference = "Stop"

Write-Host "F1 2026 MIAMI GP -- END-TO-END PIPELINE" -ForegroundColor Cyan
Write-Host "Telemetry -> Kafka -> Inference -> ClickHouse -> Dashboard" -ForegroundColor Cyan
Write-Host ""

$PRODUCER_DIR  = "c:\iot-docker"
$INFERENCE_DIR = "c:\telemetry-producer\f1_predictions"
$MODEL_DIR     = "c:\telemetry-producer\models"
$UI_DIR        = "c:\telemetry-producer\ui"
$DATASET_ROOT  = "c:\telemetry-producer\src\main\java\f1producer\raw_telemetry_per_driver"
$EVENT         = "Miami_Grand_Prix"  # Note: underscores for CSV file matching
$EVENT_DISPLAY = "Miami Grand Prix"  # Display name with spaces
$YEAR          = 2026
$SESSION       = "R"
$SPEED_FACTOR  = 10
$UI_PORT       = 8000
$CH_HOST       = "localhost"
$CH_PORT       = 8123
$POLL_INTERVAL = 500  # milliseconds to wait before checking for new lap data
$PYTHON = "C:\telemetry-producer\udfs\executable\venv_train\Scripts\python.exe"
$PIP    = "C:\telemetry-producer\udfs\executable\venv_train\Scripts\pip.exe"

Write-Host "[CONFIG]" -ForegroundColor Yellow
Write-Host "  Event:           $EVENT_DISPLAY" -ForegroundColor Gray
Write-Host "  Year:            $YEAR / Session: $SESSION" -ForegroundColor Gray
Write-Host "  Telemetry speed: ${SPEED_FACTOR}x real-time" -ForegroundColor Gray
Write-Host "  Dashboard:       http://localhost:$UI_PORT" -ForegroundColor Gray
Write-Host "  ClickHouse:      ${CH_HOST}:${CH_PORT}" -ForegroundColor Gray
Write-Host ""

Write-Host "[VERIFY] Checking prerequisites..." -ForegroundColor Yellow
if (-not (Test-Path $PRODUCER_DIR))  { Write-Host "ERROR - Producer directory not found: $PRODUCER_DIR" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $INFERENCE_DIR)) { Write-Host "ERROR - Inference directory not found: $INFERENCE_DIR" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $MODEL_DIR))     { Write-Host "ERROR - Model directory not found: $MODEL_DIR" -ForegroundColor Red; Write-Host "  Run: cd models/training_models && python train_all_models.py" -ForegroundColor Gray; exit 1 }
if (-not (Test-Path $DATASET_ROOT))  { Write-Host "ERROR - Dataset directory not found: $DATASET_ROOT" -ForegroundColor Red; exit 1 }
Write-Host "OK - All directories found" -ForegroundColor Green

# Check for required model files
$requiredModels = @("winner_model.pkl", "tire_model.pkl", "pit_model.pkl", "pace_model.pkl", "team_encoder.pkl", "compound_classes.pkl", "median_stint_lengths.pkl")
$missingModels = @()
foreach ($model in $requiredModels) {
    if (-not (Test-Path "$MODEL_DIR\$model")) {
        $missingModels += $model
    }
}
if ($missingModels.Count -gt 0) {
    Write-Host "ERROR - Missing required model files:" -ForegroundColor Red
    $missingModels | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "Run: cd models/training_models && python train_all_models.py" -ForegroundColor Gray
    exit 1
}
Write-Host "OK - All model files present" -ForegroundColor Green
Write-Host ""

$csvCount = (Get-ChildItem -Path $DATASET_ROOT -Filter "2026_Miami_Grand_Prix_R_*.csv" | Measure-Object).Count
Write-Host "[CSV FILES] Found $csvCount telemetry files" -ForegroundColor Yellow
if ($csvCount -eq 0) { Write-Host "ERROR - No telemetry CSVs found" -ForegroundColor Red; exit 1 }
Write-Host "OK - Telemetry files ready" -ForegroundColor Green
Write-Host ""

Write-Host "[BUILD] Compiling Java telemetry producer..." -ForegroundColor Yellow
Push-Location
cd c:\telemetry-producer
if (Test-Path "pom.xml") {
    mvn clean package -q -DskipTests
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR - Maven build failed" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Write-Host "OK - JAR file built successfully" -ForegroundColor Green
} else {
    Write-Host "ERROR - pom.xml not found" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location
Write-Host ""

Write-Host "[CLICKHOUSE] Testing connectivity..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "http://${CH_HOST}:${CH_PORT}/?query=SELECT%201" -TimeoutSec 3 -ErrorAction Stop | Out-Null
    Write-Host "OK - ClickHouse is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR - Cannot connect to ClickHouse at ${CH_HOST}:${CH_PORT}" -ForegroundColor Red
    exit 1
}
Write-Host ""

Write-Host "[READY] Ready to launch pipeline. Press Enter to continue..." -ForegroundColor Cyan
Read-Host | Out-Null
Write-Host ""

Write-Host "[LAUNCH] Starting Kafka telemetry producer..." -ForegroundColor Yellow
$producerCmd = "cd '$PRODUCER_DIR'; " +
               "`$env:DATASET_ROOT = '$DATASET_ROOT'; " +
               "`$env:YEAR = $YEAR; " +
               "`$env:EVENT = '$EVENT'; " +
               "`$env:SPEED_FACTOR = $SPEED_FACTOR; " +
               "Write-Host '=== KAFKA PRODUCER ===' -ForegroundColor Green; " +
               "docker compose up --build producer; " +
               "pause"
$producer = Start-Process powershell -ArgumentList "-NoExit", "-Command", $producerCmd -PassThru
Write-Host "Producer window opened (PID: $($producer.Id))" -ForegroundColor Green
Start-Sleep -Seconds 3

Write-Host "[LAUNCH] Starting inference engine..." -ForegroundColor Yellow
$inferenceCmd = "cd '$INFERENCE_DIR'; " +
                "`$env:MODEL_DIR = '$MODEL_DIR'; " +
                "`$env:PYTHONUNBUFFERED = '1'; " +
                "Write-Host '=== INFERENCE ENGINE ===' -ForegroundColor Green; " +
                "Write-Host 'Waiting for telemetry data in ClickHouse raw_telemetry table...' -ForegroundColor Cyan; " +
                "& '$PYTHON' inference_engine.py " +
                "--clickhouse-live " +
                "--write-preds " +
                "--event '$EVENT_DISPLAY' " +
                "--year $YEAR " +
                "--session '$SESSION' " +
                "--ch-host $CH_HOST " +
                "--ch-port $CH_PORT " +
                "--preds-table prediction_results " +
                "--poll-interval-ms $POLL_INTERVAL " +
                "--ui-backend 'http://localhost:${UI_PORT}' 2>&1; " +
                "pause"
$inference = Start-Process powershell -ArgumentList "-NoExit", "-Command", $inferenceCmd -PassThru
Write-Host "Inference window opened (PID: $($inference.Id))" -ForegroundColor Green
Start-Sleep -Seconds 3

Write-Host "[LAUNCH] Starting Dashboard UI..." -ForegroundColor Yellow
$uiCmd = "cd '$UI_DIR'; " +
         "& '$PIP' install -q -r requirements.txt; " +
         "Write-Host '=== DASHBOARD UI ===' -ForegroundColor Green; " +
         "`$env:F1_EVENT = '$EVENT_DISPLAY'; " +
         "`$env:F1_YEAR = $YEAR; " +
         "`$env:CH_HOST = '$CH_HOST'; " +
         "`$env:CH_PORT = '$CH_PORT'; " +
         "python main.py; " +
         "pause"
$ui = Start-Process powershell -ArgumentList "-NoExit", "-Command", $uiCmd -PassThru
Write-Host "UI window opened (PID: $($ui.Id))" -ForegroundColor Green
Start-Sleep -Seconds 5

Write-Host "[UI] Dashboard available at http://localhost:$UI_PORT" -ForegroundColor Cyan
Write-Host "[UI] Note: UI may take 10-15 seconds to become ready" -ForegroundColor Gray
Start-Sleep -Seconds 2
Start-Process "chrome.exe" -ArgumentList "http://localhost:$UI_PORT/"

Write-Host ""
Write-Host "[STREAMING STARTED]" -ForegroundColor Green
Write-Host "  CSVs -> Kafka -> ClickHouse raw_telemetry" -ForegroundColor Green
Write-Host "  -> Inference Engine processes new laps" -ForegroundColor Green
Write-Host "  -> ClickHouse prediction_results" -ForegroundColor Green
Write-Host "  -> UI Dashboard" -ForegroundColor Green
Write-Host ""
Write-Host "[DEBUGGING] To monitor data flow:" -ForegroundColor Gray
Write-Host "  1. Check raw_telemetry count: http://${CH_HOST}:${CH_PORT}/?query=SELECT%20COUNT(*)%20FROM%20raw_telemetry%20WHERE%20event%20=%20'${EVENT_DISPLAY}'%20AND%20year=%20${YEAR}" -ForegroundColor Gray
Write-Host "  2. Check predictions count: http://${CH_HOST}:${CH_PORT}/?query=SELECT%20COUNT(*)%20FROM%20prediction_results%20WHERE%20event%20=%20'${EVENT_DISPLAY}'%20AND%20year=%20${YEAR}" -ForegroundColor Gray
Write-Host ""
Write-Host "[WAITING] Pipeline running. Press Ctrl+C to stop..." -ForegroundColor Yellow

try {
    $producer.WaitForExit()
    $inference.WaitForExit()
} finally {
    Write-Host ""
    Write-Host "Stopping pipeline..." -ForegroundColor Yellow
    foreach ($proc in @($producer, $inference, $ui)) {
        if ($proc -and -not $proc.HasExited) {
            $proc.CloseMainWindow() | Out-Null
            if (-not $proc.WaitForExit(3000)) { $proc.Kill() }
        }
    }
    Write-Host "[DONE]" -ForegroundColor Green
    Write-Host ""
    Write-Host "[TROUBLESHOOTING] If inference engine showed no output:" -ForegroundColor Yellow
    Write-Host "  1. Check ClickHouse has raw_telemetry data:" -ForegroundColor Gray
    Write-Host "     SELECT COUNT(*), MAX(LapNumber), SESSION FROM raw_telemetry WHERE event='$EVENT_DISPLAY' AND year=$YEAR" -ForegroundColor Gray
    Write-Host "  2. Verify models exist:" -ForegroundColor Gray
    Write-Host "     ls $MODEL_DIR/*.pkl" -ForegroundColor Gray
    Write-Host "  3. Run producer manually and check Kafka messages arrive" -ForegroundColor Gray
    Write-Host "  4. Check inference logs: search for 'ClickHouse poll failed' or 'No telemetry rows'" -ForegroundColor Gray
}
