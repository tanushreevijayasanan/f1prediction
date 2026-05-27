package f1producer;

public interface DriverStream {

    boolean hasNext();

    TelemetryRecord poll() throws Exception;
}