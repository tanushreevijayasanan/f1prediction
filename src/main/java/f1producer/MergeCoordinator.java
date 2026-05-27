package f1producer;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MergeCoordinator {
    private static final Logger logger = LoggerFactory.getLogger(MergeCoordinator.class);

    private final double speedFactor;
    private AtomicLong skippedRecords = new AtomicLong(0);
    private AtomicLong failedKafkaSends = new AtomicLong(0);

    public MergeCoordinator(double speedFactor) {
        this.speedFactor = speedFactor;
    }

    public long getSkippedRecords() { return skippedRecords.get(); }
    public long getFailedKafkaSends() { return failedKafkaSends.get(); }

    private static class Entry implements Comparable<Entry> {

        DriverStream stream;
        TelemetryRecord record;

        Entry(DriverStream s, TelemetryRecord r) {
            stream = s;
            record = r;
        }

        @Override
        public int compareTo(Entry o) {
            int lapCmp = Integer.compare(record.LapNumber, o.record.LapNumber);
            if (lapCmp != 0) return lapCmp;
            return Long.compare(record.Time_ms, o.record.Time_ms);
        }
    }

    private boolean hasLapInHeap(PriorityQueue<Entry> heap, int lap) {
        for (Entry e : heap) {
            if (e.record.LapNumber == lap) return true;
        }
        return false;
    }

    public void run(List<DriverStream> streams,
                    KafkaProducer<String, String> producer)
            throws Exception {

        PriorityQueue<Entry> heap = new PriorityQueue<>();
        logger.info("Initializing merge coordinator with {} streams", streams.size());

        for (DriverStream s : streams) {
            try {
                TelemetryRecord r = s.poll();
                if (r != null) heap.add(new Entry(s, r));
            } catch (Exception e) {
                logger.error("Error initializing stream, skipping", e);
            }
        }
        logger.info("Heap initialized with {} entries", heap.size());

        int currentLap = -1;
        long lapWallStartNs = -1;
        long lapBaseTimeMs = 0;
        long maxBehindMs = 0;
        long recordsSent = 0;

        while (!heap.isEmpty()) {
            if (speedFactor <= 0) {
                throw new IllegalArgumentException("speedFactor must be > 0");
            }

            if (currentLap < 0) {
                currentLap = heap.peek().record.LapNumber;
                lapWallStartNs = System.nanoTime();
                lapBaseTimeMs = heap.peek().record.Time_ms;
                logger.info("Starting lap {}", currentLap);
            }

            if (!hasLapInHeap(heap, currentLap)) {
                logger.info("Lap {} complete", currentLap);
                currentLap = heap.peek().record.LapNumber;
                lapWallStartNs = System.nanoTime();
                lapBaseTimeMs = heap.peek().record.Time_ms;
                logger.info("Starting lap {}", currentLap);
                continue;
            }

            Entry first = heap.poll();
            long t = first.record.Time_ms;
            long elapsedLapMs = Math.max(0L, t - lapBaseTimeMs);
            long targetNs = lapWallStartNs + (long) ((elapsedLapMs * 1_000_000.0) / speedFactor);
            long nowNs = System.nanoTime();
            long sleepNs = targetNs - nowNs;
            if (sleepNs > 0) {
                long sleepMs = sleepNs / 1_000_000L;
                int sleepRemainderNs = (int) (sleepNs % 1_000_000L);
                Thread.sleep(sleepMs, sleepRemainderNs);
            } else {
                long behindMs = (-sleepNs) / 1_000_000L;
                if (behindMs > maxBehindMs) maxBehindMs = behindMs;
            }

            List<Entry> group = new ArrayList<>();
            group.add(first);

            while (!heap.isEmpty() &&
                   heap.peek().record.LapNumber == currentLap &&
                   heap.peek().record.Time_ms == t) {
                group.add(heap.poll());
            }

            for (Entry e : group) {
                try {
                    String json = JsonUtil.toJson(e.record);
                    
                    producer.send(
                        new ProducerRecord<>("telemetryf1test", e.record.driver, json),
                        new Callback() {
                            public void onCompletion(RecordMetadata metadata, Exception exception) {
                                if (exception != null) {
                                    logger.error("Failed to send telemetry for driver " + e.record.driver 
                                        + " lap " + e.record.LapNumber, exception);
                                    failedKafkaSends.incrementAndGet();
                                }
                            }
                        }
                    );

                    recordsSent++;

                    if (recordsSent % 1000 == 0) {
                        logger.info("Records sent: {} | lap={} | t={}ms | heap size={} | max behind={}ms | failed kafka sends={}",
                            recordsSent, currentLap, t, heap.size(), maxBehindMs, failedKafkaSends.get());
                    }

                    try {
                        TelemetryRecord next = e.stream.poll();
                        if (next != null) {
                            heap.add(new Entry(e.stream, next));
                        }
                    } catch (Exception streamErr) {
                        logger.error("Error polling from stream (skipping record): ", streamErr);
                        skippedRecords.incrementAndGet();
                    }
                    
                } catch (Exception sendErr) {
                    logger.error("Error sending record for driver " + e.record.driver, sendErr);
                    failedKafkaSends.incrementAndGet();
                }
            }
        }

        producer.flush();
        logger.info("Merge coordinator finished. Total sent: {} | Skipped: {} | Failed Kafka sends: {}",
            recordsSent, skippedRecords.get(), failedKafkaSends.get());
        if (failedKafkaSends.get() > 0) {
            logger.error("WARNING: {} Kafka messages failed to send", failedKafkaSends.get());
        }
    }
}
