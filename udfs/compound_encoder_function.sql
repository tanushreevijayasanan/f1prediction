-- Compound Encoder Function - Maps compound names to classes
CREATE OR REPLACE FUNCTION f1_compound_encoder AS (compound_name) ->
    CASE 
        WHEN compound_name = 'SOFT' THEN 0
        WHEN compound_name = 'MEDIUM' THEN 1
        WHEN compound_name = 'HARD' THEN 2
        WHEN compound_name = 'INTERMEDIATE' THEN 3
        WHEN compound_name = 'WET' THEN 4
        ELSE 1 -- default to medium
    END;
