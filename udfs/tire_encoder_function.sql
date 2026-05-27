-- Tire Encoder Function - Maps tire age to categories
CREATE OR REPLACE FUNCTION f1_tire_encoder AS (tire_age_laps) ->
    CASE 
        WHEN tire_age_laps <= 5 THEN 'New'
        WHEN tire_age_laps <= 15 THEN 'Fresh'
        WHEN tire_age_laps <= 25 THEN 'Optimal'
        WHEN tire_age_laps <= 35 THEN 'Worn'
        ELSE 'Critical'
    END;
