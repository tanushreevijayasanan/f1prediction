-- Team Encoder Function - Maps driver codes to teams
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
