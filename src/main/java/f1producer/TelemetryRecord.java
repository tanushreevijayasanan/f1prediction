package f1producer;

public class TelemetryRecord {

    public String Time;
    public long Time_ms;
    public Float Speed;
    public Float RPM;
    public Integer nGear;
    public Float Throttle;
    public Integer Brake;
    public Integer DRS;
    public Float Distance;
    public Float RelativeDistance;
    public Float X;
    public Float Y;
    public Float Z;
    public Integer LapNumber;
    public Integer Stint;
    public String Compound;
    public Integer is_pit_lap;
    public Float TrackTemp;
    public Float AirTemp;
    public Float Rainfall;
    public String weather;
    public Integer corner_id;
    public String track_segment;
    public Integer hard_brake;
    public Integer full_throttle;
    public String driver;
    public String driver_name;
    public String driver_code;
    public String team;
    public String event;
    public Integer year;
    public String session;

    public TelemetryRecord(
        String Time,
        long Time_ms,
        Float Speed,
        Float RPM,
        Integer nGear,
        Float Throttle,
        Integer Brake,
        Integer DRS,
        Float Distance,
        Float RelativeDistance,
        Float X,
        Float Y,
        Float Z,
        Integer LapNumber,
        Integer Stint,
        String Compound,
        Integer is_pit_lap,
        Float TrackTemp,
        Float AirTemp,
        Float Rainfall,
        String weather,
        Integer corner_id,
        String track_segment,
        Integer hard_brake,
        Integer full_throttle,
        String driver,
        String driver_name,
        String driver_code,
        String team,
        String event,
        Integer year,
        String session
    ) {
        // Validate critical fields
        if (driver == null || driver.trim().isEmpty()) {
            throw new IllegalArgumentException("driver cannot be null or empty");
        }
        if (event == null || event.trim().isEmpty()) {
            throw new IllegalArgumentException("event cannot be null or empty");
        }
        if (year == null) {
            throw new IllegalArgumentException("year cannot be null");
        }
        if (LapNumber == null || LapNumber < 0) {
            throw new IllegalArgumentException("LapNumber must be >= 0, got: " + LapNumber);
        }
        if (Speed != null && Speed < 0) {
            Speed = 0.0f;  // clamp negative speed (sensor noise)
        }
        // FastF1 throttle/brake occasionally exceeds 100 by a fraction due to
        // floating-point encoding — clamp rather than discard the whole record.
        if (Throttle != null && Throttle < 0)   Throttle = 0.0f;
        if (Throttle != null && Throttle > 100) Throttle = 100.0f;
        if (Brake    != null && Brake    < 0)   Brake    = 0;
        if (Brake    != null && Brake    > 100) Brake    = 100;

        this.Time = Time;
        this.Time_ms = Time_ms;
        this.Speed = Speed;
        this.RPM = RPM;
        this.nGear = nGear;
        this.Throttle = Throttle;
        this.Brake = Brake;
        this.DRS = DRS;
        this.Distance = Distance;
        this.RelativeDistance = RelativeDistance;
        this.X = X;
        this.Y = Y;
        this.Z = Z;
        this.LapNumber = LapNumber;
        this.Stint = Stint;
        this.Compound = Compound;
        this.is_pit_lap = is_pit_lap;
        this.TrackTemp = TrackTemp;
        this.AirTemp = AirTemp;
        this.Rainfall = Rainfall;
        this.weather = weather;
        this.corner_id = corner_id;
        this.track_segment = track_segment;
        this.hard_brake = hard_brake;
        this.full_throttle = full_throttle;
        this.driver = driver;
        this.driver_name = driver_name;
        this.driver_code = driver_code;
        this.team = team;
        this.event = event;
        this.year = year;
        this.session = session;
    }
}
