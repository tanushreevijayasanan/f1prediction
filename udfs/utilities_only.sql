-- Utility SQL UDFs kept in ClickHouse SQL form.
-- Model-scoring UDFs are intentionally moved to executable UDFs.

CREATE OR REPLACE FUNCTION f1_team_encoder AS (driver_code) ->
    CASE
        WHEN driver_code IN ('ANT', 'RUS') THEN 'Mercedes'
        WHEN driver_code IN ('NOR', 'PIA') THEN 'McLaren'
        WHEN driver_code IN ('LEC', 'HAM') THEN 'Ferrari'
        WHEN driver_code IN ('VER', 'HAD') THEN 'Red Bull Racing'
        WHEN driver_code IN ('ALO', 'STR') THEN 'Aston Martin'
        WHEN driver_code IN ('GAS', 'COL') THEN 'Alpine'
        WHEN driver_code IN ('OCO', 'BEA') THEN 'Haas F1 Team'
        WHEN driver_code IN ('ALB', 'SAI') THEN 'Williams'
        WHEN driver_code IN ('LAW', 'LIN') THEN 'Racing Bulls'
        WHEN driver_code IN ('HUL', 'BOR') THEN 'Audi'
        WHEN driver_code IN ('BOT', 'PER') THEN 'Cadillac'
        ELSE 'Unknown'
    END;

CREATE OR REPLACE FUNCTION f1_track_encoder AS (track_name) ->
    CASE
        WHEN track_name IN ('Monza', 'Monaco', 'Hungary', 'Singapore') THEN 'Street'
        WHEN track_name IN ('Silverstone', 'Spa', 'Suzuka', 'Bahrain') THEN 'High-Speed'
        WHEN track_name IN ('Barcelona', 'Miami', 'Jeddah', 'Baku') THEN 'Medium'
        WHEN track_name IN ('Montreal', 'Melbourne') THEN 'Technical'
        WHEN track_name IN ('Austin', 'Mexico City') THEN 'Mixed'
        ELSE 'Unknown'
    END;

CREATE OR REPLACE FUNCTION f1_compound_encoder AS (compound_name) ->
    CASE
        WHEN compound_name = 'SOFT' THEN 0
        WHEN compound_name = 'MEDIUM' THEN 1
        WHEN compound_name = 'HARD' THEN 2
        WHEN compound_name = 'INTERMEDIATE' THEN 3
        WHEN compound_name = 'WET' THEN 4
        ELSE 1
    END;

CREATE OR REPLACE FUNCTION f1_tire_encoder AS (tire_age_laps) ->
    CASE
        WHEN tire_age_laps <= 5 THEN 'New'
        WHEN tire_age_laps <= 15 THEN 'Fresh'
        WHEN tire_age_laps <= 25 THEN 'Optimal'
        WHEN tire_age_laps <= 35 THEN 'Worn'
        ELSE 'Critical'
    END;

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
        ELSE 25.0
    END;

CREATE OR REPLACE FUNCTION f1_speed_threshold AS (track_name) ->
    CASE
        WHEN track_name IN ('Monza', 'Bahrain', 'Jeddah') THEN 240.0
        WHEN track_name IN ('Silverstone', 'Spa') THEN 220.0
        WHEN track_name IN ('Suzuka', 'Barcelona') THEN 225.0
        WHEN track_name IN ('Miami', 'Austin') THEN 235.0
        WHEN track_name IN ('Montreal', 'Melbourne') THEN 210.0
        WHEN track_name IN ('Baku') THEN 215.0
        ELSE 220.0
    END;
