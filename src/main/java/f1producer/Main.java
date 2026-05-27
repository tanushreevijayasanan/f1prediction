package f1producer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    private static final int TARGET_YEAR = 2026;
    private static final String TARGET_EVENT = "Miami_Grand_Prix";

    public static void main(String[] args) throws Exception {

        String datasetPath = System.getenv("DATASET_ROOT");

        if (datasetPath == null && args.length > 0) {
            datasetPath = args[0];
        }

        if (datasetPath == null) {
            throw new RuntimeException("DATASET_ROOT not set");
        }

        File root = new File(datasetPath);

        logger.info("Looking for CSVs in: {}", root.getAbsolutePath());
        logger.debug("Directory exists: {}", root.exists());
        logger.debug("Is directory: {}", root.isDirectory());

        if (!root.exists() || !root.isDirectory()) {
            throw new RuntimeException("Invalid dataset directory: " + datasetPath);
        }

        File[] allFiles = root.listFiles();
        logger.info("Total files in directory: {}", (allFiles == null ? 0 : allFiles.length));

        Properties props = new Properties();
        String kafkaBootstrap = System.getenv("KAFKA_BOOTSTRAP_SERVERS");
        if (kafkaBootstrap == null || kafkaBootstrap.isEmpty()) {
            kafkaBootstrap = "kafka:29092";  // Default for Docker
        }
        props.put("bootstrap.servers", kafkaBootstrap);
        logger.info("Kafka bootstrap servers: {}", kafkaBootstrap);
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("acks", "1");
        props.put("retries", "3");
        // Throughput-oriented defaults for smoother streaming
        props.put("linger.ms", "5");
        props.put("batch.size", String.valueOf(32 * 1024));
        props.put("compression.type", "lz4");

        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        int targetYear = TARGET_YEAR;
        String targetEvent = TARGET_EVENT;

        String yearEnv = System.getenv("YEAR");
        if (yearEnv != null && !yearEnv.isEmpty()) {
            try {
                targetYear = Integer.parseInt(yearEnv);
            } catch (NumberFormatException e) {
                logger.warn("Invalid YEAR env value, using default: {}", targetYear);
            }
        }

        String eventEnv = System.getenv("EVENT");
        if (eventEnv != null && !eventEnv.isEmpty()) {
            targetEvent = eventEnv;
        }

        logger.info("Target year: {}", targetYear);
        logger.info("Target event: {}", targetEvent);

        List<DriverStream> streams = discoverStreams(root, targetYear, targetEvent);

        if (streams.isEmpty()) {
            logger.error("No CSV telemetry files found.");
            logger.debug("Expected files starting with: {}_", targetYear);
            logger.debug("Expected files containing: _{}_", targetEvent);

            if (allFiles != null) {
                logger.debug("First 10 files in directory:");
                int count = 0;
                for (File f : allFiles) {
                    if (count++ >= 10) break;
                    logger.debug("  - {}", f.getName());
                }
            }

            producer.close();
            return;
        }

        logger.info("Found {} driver streams for {}", streams.size(), targetEvent);

        // Read SPEED_FACTOR from env — default 1x for real-time simulation
        double speedFactor = 1.0;
        String speedEnv = System.getenv("SPEED_FACTOR");
        if (speedEnv != null && !speedEnv.isEmpty()) {
            try {
                speedFactor = Double.parseDouble(speedEnv);
            } catch (NumberFormatException e) {
                logger.warn("Invalid SPEED_FACTOR env value, using default: {}", speedFactor);
            }
        }
        logger.info("Streaming at {}x real-time speed", speedFactor);

        new MergeCoordinator(speedFactor).run(streams, producer);

        logger.info("Streaming complete.");
        producer.close();
    }

    private static List<DriverStream> discoverStreams(File root, int targetYear, String targetEvent) throws Exception {

        List<DriverStream> streams = new ArrayList<>();
        File[] files = root.listFiles();

        if (files == null) return streams;

        for (File f : files) {

            if (!f.isFile()) continue;
            if (!f.getName().endsWith(".csv")) continue;

            String name = f.getName();

            boolean yearMatch = name.startsWith(targetYear + "_");
            boolean eventMatch = name.contains("_" + targetEvent + "_");

            logger.debug("[SCAN] {} | year={} | event={}", name, yearMatch, eventMatch);

            if (!yearMatch || !eventMatch) continue;

            try {
                logger.info("[LOAD] Adding stream: {}", name);
                streams.add(new DriverCSVStream(f));
            } catch (Exception e) {
                logger.error("Failed to load CSV stream {}: {}", f.getName(), e.getMessage(), e);
            }
        }

        return streams;
    }
}
