-- Track Encoder Function - Maps track names to types
CREATE OR REPLACE FUNCTION f1_track_encoder AS (track_name) ->
    CASE 
        WHEN track_name IN ('Monza', 'Monaco', 'Hungary', 'Singapore') THEN 'Street'
        WHEN track_name IN ('Silverstone', 'Spa', 'Suzuka', 'Bahrain') THEN 'High-Speed'
        WHEN track_name IN ('Barcelona', 'Miami', 'Jeddah', 'Baku') THEN 'Medium'
        WHEN track_name IN ('Montreal', 'Melbourne') THEN 'Technical'
        WHEN track_name IN ('Austin', 'Mexico City') THEN 'Mixed'
        ELSE 'Unknown'
    END;