-- Stint Length Function - Returns median stint lengths
CREATE OR REPLACE FUNCTION f1_stint_length_median AS (track_name) ->
    CASE 
        WHEN track_name = 'Monza' THEN 25.0
        WHEN track_name = 'Silverstone' THEN 28.0
        WHEN track_name = 'Suzuka' THEN 26.0
        WHEN track_name = 'Spa' THEN 24.0
        WHEN track_name = 'Bahrain' THEN 22.0
        WHEN track_name = 'Jeddah' THEN 23.0
        WHEN track_name = 'Miami' THEN 25.0
        WHEN track_name = 'Barcelona' THEN 27.0
        WHEN track_name = 'Montreal' THEN 20.0
        WHEN track_name = 'Baku' THEN 18.0
        WHEN track_name = 'Austin' THEN 22.0
        WHEN track_name = 'Mexico City' THEN 21.0
        WHEN track_name = 'Melbourne' THEN 24.0
        ELSE 25.0 -- default stint length
    END;
