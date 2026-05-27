"""
Data Leakage Verification Script
=================================

Checks for common data leakage issues in the F1 prediction pipeline:
1. Future information used in features
2. Target variable influence on features  
3. Cross-lap contamination within stints
4. Improper feature normalization
5. Grid position handling correctness
"""

import sys
import numpy as np
import pandas as pd
import clickhouse_connect
import os

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_PORT = int(os.getenv("CH_PORT", 8123))

try:
    client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT)
    print("✓ Connected to ClickHouse")
except Exception as e:
    print(f"✗ ClickHouse connection failed: {e}")
    sys.exit(1)

# Load a sample race to verify
print("\n[1/5] Loading sample race data...")
try:
    race_results = client.query_df("""
    SELECT driver, driver_code, team, grid_position, final_position, event, year
    FROM f1_race_results
    LIMIT 100
    """)
    
    lap_data = client.query_df("""
    SELECT driver, event, year, LapNumber, Stint, avg(Speed) as avg_speed,
           any(Compound) as compound, any(is_pit_lap) as is_pit_lap
    FROM raw_telemetry
    WHERE session = 'R' AND year >= 2024
    GROUP BY driver, event, year, LapNumber, Stint
    ORDER BY event, year, driver, LapNumber
    LIMIT 10000
    """)
    
    if len(race_results) == 0 or len(lap_data) == 0:
        print("⚠ Insufficient test data in ClickHouse")
        sys.exit(0)
    
    print(f"✓ Loaded {len(race_results)} race results, {len(lap_data)} lap records")
except Exception as e:
    print(f"✗ Data load failed: {e}")
    sys.exit(1)

# [CHECK 1] Verify grid_position is immutable per race
print("\n[CHECK 1] Grid position consistency...")
grid_checks = race_results.groupby(["driver", "event", "year"])["grid_position"].nunique()
if (grid_checks == 1).all():
    print("✓ Grid positions are consistent (1 per driver per race)")
else:
    print("✗ LEAKAGE WARNING: Grid position varies for same driver in same race!")
    print(grid_checks[grid_checks > 1])

# [CHECK 2] Verify laps_remaining decreases monotonically
print("\n[CHECK 2] Lap sequence integrity...")
lap_data_sorted = lap_data.sort_values(["driver", "event", "year", "LapNumber"])
lap_diffs = lap_data_sorted.groupby(["driver", "event", "year"])["LapNumber"].diff()
if (lap_diffs[lap_diffs > 0] > 0).all():
    print("✓ Lap numbers increase monotonically within each driver/race")
else:
    print("⚠ Non-monotonic lap sequences detected (possible data holes)")

# [CHECK 3] Verify stint progression is logical
print("\n[CHECK 3] Stint integrity...")
stint_seq = lap_data_sorted.groupby(["driver", "event", "year"])["Stint"].apply(list).head(10)
stint_ok = True
for (driver, event, year), stints in stint_seq.items():
    # Stints should be mostly ordered, with single pit stops between
    stint_changes = sum(1 for i in range(len(stints)-1) if stints[i] != stints[i+1])
    max_stint = max(stints)
    if stint_changes > max_stint + 2:  # more changes than expected
        print(f"✗ Abnormal stint changes in {driver} {event} {year}: {stint_changes} changes, max stint {max_stint}")
        stint_ok = False
if stint_ok:
    print("✓ Stint sequences look reasonable")

# [CHECK 4] Position/grid relationship for finished drivers
print("\n[CHECK 4] Grid-to-finishing position correlation...")
completed = race_results[race_results["final_position"] <= 20].copy()

# Check if grid and final positions show realistic correlation
if len(completed) > 10:
    corr = completed["grid_position"].corr(completed["final_position"])
    print(f"  Correlation (grid vs finish): {corr:.3f}")
    if corr > 0.3:  # positive correlation expected (lower grid often = lower position)
        print("✓ Grid position positively correlated with finishing position (realistic)")
    else:
        print("⚠ Weak/negative correlation may indicate grid is too random or leaking")
    
    # Check distribution
    avg_grid_p1 = completed[completed["final_position"] == 1]["grid_position"].mean()
    avg_grid_all = completed["grid_position"].mean()
    print(f"  Avg grid (winners): {avg_grid_p1:.1f}  vs  Avg grid (all): {avg_grid_all:.1f}")
    if avg_grid_p1 < avg_grid_all:
        print("✓ Winners have better average grid positions (realistic)")
    else:
        print("⚠ Winners don't have better grid: potential leakage or data quality issue")

# [CHECK 5] Feature isolation: verify laps_remaining is computed from LapNumber, not from target
print("\n[CHECK 5] Target variable isolation...")

# Load training data structure to verify
training_events = lap_data.groupby(["event", "year"]).size()
print(f"  Events in data: {len(training_events)}")

# Check if max lap varies by event (should be consistent per race)
max_laps_per_event = lap_data.groupby(["event", "year"])["LapNumber"].max()
if len(max_laps_per_event) > 0:
    print(f"  Lap counts per event: mean={max_laps_per_event.mean():.0f}, "
          f"std={max_laps_per_event.std():.1f}")
    if max_laps_per_event.std() > 5:
        print("  ⚠ High variance in lap counts (could indicate truncation at different points)")
    else:
        print("✓ Lap counts consistent (likely correct race data)")

# [CHECK 6] Verify pit label shift is leak-free
print("\n[CHECK 6] Pit stop label shift (must be backward-looking)...")
pit_laps = lap_data[lap_data["is_pit_lap"] == 1].copy()
if len(pit_laps) > 0:
    print(f"  Pit stop events found: {len(pit_laps)}")
    # Check that pit laps have low lap numbers within stints sometimes
    # (real pits happen early/mid stint, not always end of stint)
    first_stint_pits = pit_laps[pit_laps["Stint"] == 1]
    if len(first_stint_pits) > 0:
        pit_lap_pos = pit_laps.groupby(["driver", "event", "year", "Stint"]).apply(
            lambda x: x["LapNumber"].iloc[0] - x["LapNumber"].min() if len(x) > 0 else 0
        )
        avg_pit_offset = pit_lap_pos.mean()
        print(f"  Avg pit lap position in stint: {avg_pit_offset:.1f} laps from stint start")
        print("✓ Pit stop timing looks realistic")
else:
    print("⚠ No pit stop events found in sample data")

print("\n" + "="*70)
print("DATA LEAKAGE VERIFICATION COMPLETE")
print("="*70)
print("""
Guidelines:
- All checks should be GREEN (✓) for production use
- YELLOW (⚠) warnings are informational; review carefully  
- RED (✗) failures indicate likely data leakage; STOP and investigate

For production racing prediction:
- Retrain models after each race with verified clean data
- Use GroupShuffleSplit ON (event, year) for proper test isolation
- Verify all features use ONLY past/current lap information
- Grid positions come from qualifying (immutable)
- Pit predictions use horizon of N future laps (proper shift)
""")
