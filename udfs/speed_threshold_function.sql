-- Speed Threshold Function - Returns speed classification thresholds
CREATE OR REPLACE FUNCTION f1_speed_threshold AS (track_name) ->
    CASE 
        WHEN track_name IN ('Monza', 'Bahrain', 'Jeddah') THEN 240.0
        WHEN track_name IN ('Silverstone', 'Spa') THEN 220.0
        WHEN track_name IN ('Suzuka', 'Barcelona') THEN 225.0
        WHEN track_name IN ('Miami', 'Austin') THEN 235.0
        WHEN track_name IN ('Montreal', 'Melbourne') THEN 210.0
        WHEN track_name IN ('Baku') THEN 215.0
        ELSE 220.0 -- default threshold
    END;
