package f1producer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DriverCSVStream implements DriverStream {
    private static final Logger logger = LoggerFactory.getLogger(DriverCSVStream.class);

    private BufferedReader reader;
    private String nextLine;
    private String[] header;
    private String driverId;

    public DriverCSVStream(File file) throws Exception {
        if (!file.exists()) {
            throw new IllegalArgumentException("CSV file not found: " + file.getAbsolutePath());
        }
        
        reader = new BufferedReader(new FileReader(file));
        this.driverId = file.getName();

        String head = reader.readLine();
        if (head == null) {
            throw new IllegalArgumentException("CSV file is empty: " + file.getName());
        }
        header = parseCsvLine(head);
        Map<String, Integer> colIndex = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            colIndex.put(header[i].trim().toLowerCase(), i);
        }
        String[] required = {"lapnumber", "time_ms", "speed", "driver", "event", "year", "session"};
        for (String col : required) {
            if (!colIndex.containsKey(col)) {
                throw new IllegalArgumentException("CSV missing required column: " + col + " in " + file.getName());
            }
        }

        nextLine = reader.readLine();
    }

    public String getDriverId() {
        return driverId;
    }

    public boolean hasNext() {
        return nextLine != null;
    }

    public TelemetryRecord poll() throws Exception {
        if (nextLine == null) return null;

        String currentLine = nextLine;
        nextLine = reader.readLine();
        if (nextLine == null) reader.close();
        
        String[] parts = parseCsvLine(currentLine);
        if (parts.length != header.length) {
            logger.warn("CSV row has " + parts.length + " columns but header has " + header.length 
                + " columns in file " + driverId);
        }

        Map<String,String> row = new HashMap<>();

        int limit = Math.min(header.length, parts.length);
        for (int i = 0; i < limit; i++) {
            row.put(header[i], parts[i]);
        }
        for (int i = limit; i < header.length; i++) {
            row.put(header[i], "");
        }

        return new TelemetryRecord(

                row.get("Time"),
                parseLong(row.get("Time_ms")),

                parseFloat(row.get("Speed")),
                parseFloat(row.get("RPM")),
                parseInt(row.get("nGear")),
                parseFloat(row.get("Throttle")),
                parseInt(row.get("Brake")),
                parseInt(row.get("DRS")),

                parseFloat(row.get("Distance")),
                parseFloat(row.get("RelativeDistance")),

                parseFloat(row.get("X")),
                parseFloat(row.get("Y")),
                parseFloat(row.get("Z")),

                parseInt(row.get("LapNumber")),
                parseInt(row.get("Stint")),
                row.get("Compound"),

                parseInt(row.get("is_pit_lap")),

                parseFloat(row.get("TrackTemp")),
                parseFloat(row.get("AirTemp")),
                parseFloat(row.get("Rainfall")),
                row.get("weather"),

                parseInt(row.get("corner_id")),
                row.get("track_segment"),

                parseInt(row.get("hard_brake")),
                parseInt(row.get("full_throttle")),

                row.get("driver"),
                row.get("driver_name"),
                resolveDriverCode(row),
                row.get("team"),
                row.get("event"),
                parseInt(row.get("year")),
                row.get("session")
        );
    }

    private String[] parseCsvLine(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == ',' && !inQuotes) {
                fields.add(current.toString());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString());
        return fields.toArray(new String[0]);
    }

    private Float parseFloat(String v) {
        if (v == null || v.isEmpty()) return null;
        // Python booleans arrive as 'True' / 'False' — map to 1.0 / 0.0 silently
        if (v.equalsIgnoreCase("true"))  return 1.0f;
        if (v.equalsIgnoreCase("false")) return 0.0f;
        try {
            return Float.parseFloat(v);
        } catch (NumberFormatException e) {
            logger.warn("Could not parse float value '" + v + "' in " + driverId);
            return null;
        }
    }

    private Integer parseInt(String v) {
        if (v == null || v.isEmpty()) return null;
        // Python booleans arrive as 'True' / 'False' — map to 1 / 0 silently
        if (v.equalsIgnoreCase("true"))  return 1;
        if (v.equalsIgnoreCase("false")) return 0;
        try {
            // handle "4.0", "1.0" etc
            if (v.contains(".")) return (int) Double.parseDouble(v);
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            logger.warn("Could not parse integer value '" + v + "' in " + driverId);
            return null;
        }
    }

    private Long parseLong(String v) {
        if (v == null || v.isEmpty()) return 0L;
        try {
            if (v.contains(".")) return (long) Double.parseDouble(v);
            return Long.parseLong(v);
        } catch (NumberFormatException e) {
            logger.warn("Could not parse long value '" + v + "' in " + driverId);
            return 0L;
        }
    }

    private String resolveDriverCode(Map<String, String> row) {
        String code = normalizeDriverCode(row.get("driver_code"));
        if (!code.isEmpty()) return code;
        // Some datasets carry a canonical 3-letter token in driver_name.
        code = normalizeDriverCode(row.get("driver_name"));
        if (!code.isEmpty()) return code;
        return "";
    }

    private String normalizeDriverCode(String v) {
        if (v == null) return "";
        String code = v.trim().toUpperCase(Locale.ROOT);
        if (code.length() != 3) return "";
        for (int i = 0; i < code.length(); i++) {
            if (!Character.isLetter(code.charAt(i))) return "";
        }
        return code;
    }
}
