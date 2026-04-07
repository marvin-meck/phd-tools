SELECT 
    file_location
FROM files
WHERE file_id IN (
    SELECT
        file_id
    FROM data_set_has_file
    WHERE data_set_id IN (
        SELECT data_set_id 
        FROM [mole_fractions_new] 
        WHERE temperature_value BETWEEN 600 AND 625
    )
);