
CREATE VIEW [mole_fractions] AS
WITH total_amount AS (
    SELECT solution_id, value AS total_amount_mol
    FROM variable_data
    WHERE var_id = (
        SELECT var_id 
        FROM variables
        WHERE model_id = 1 AND var_name = 'total_amount_substance'
    )
), 
amount AS (
    SELECT solution_id, index_str, value AS amount_mol
    FROM variable_data
    WHERE var_id = (
        SELECT var_id 
        FROM variables
        WHERE model_id = 1 AND var_name = 'amount_substance'
    )
),
pressure AS (
    SELECT
        data_set_id, 
        value
    FROM parameter_data
    WHERE param_id=(
        SELECT param_id FROM parameters
        WHERE model_id=1 AND param_name="pressure"
    )
),
temperature AS (
    SELECT
        data_set_id, 
        value
    FROM parameter_data
    WHERE param_id=(
        SELECT param_id FROM parameters
        WHERE model_id=1 AND param_name="temperature"
    )
),
amount_carbon AS (
    SELECT
        data_set_id, 
        value
    FROM parameter_data
    WHERE param_id=(
        SELECT param_id FROM parameters
        WHERE model_id=1 AND param_name="amount_element" AND index_str="C"
    )
),
amount_hydrogen AS (
    SELECT
        data_set_id, 
        value
    FROM parameter_data
    WHERE param_id=(
        SELECT param_id FROM parameters
        WHERE model_id=1 AND param_name="amount_element" AND index_str="H"
    )
),
amount_oxygen AS (
    SELECT
        data_set_id, 
        value
    FROM parameter_data
    WHERE param_id=(
        SELECT param_id FROM parameters
        WHERE model_id=1 AND param_name="amount_element" AND index_str="O"
    )
),
hydrogen_to_carbon AS (
    SELECT 
        c.data_set_id, 
        (h.value/c.value) AS value
    FROM amount_carbon c
    JOIN amount_hydrogen h
    ON c.data_set_id=h.data_set_id
),
oxygen_to_carbon AS (
    SELECT 
        c.data_set_id, 
        (o.value/c.value) AS value
    FROM amount_carbon c
    JOIN amount_oxygen o
    ON c.data_set_id=o.data_set_id
)
SELECT
    params.data_set_id,
    sol.solution_id,
    params.temperature_kelvin,
    params.pressure_bar,
    params.hydrogen_to_carbon_molar,
    params.oxygen_to_carbon_molar,
    sol.compound,
    sol.mole_fraction
FROM (
    SELECT
        solutions.data_set_id,
        solutions.solution_id,
        post.compound,
        post.mole_fraction
    FROM (
        SELECT
            a.solution_id,
            a.index_str AS compound, 
            a.amount_mol / t.total_amount_mol AS mole_fraction
        FROM amount a
        INNER JOIN total_amount t
        ON a.solution_id = t.solution_id
    ) post
    INNER JOIN solutions
    ON solutions.solution_id=post.solution_id
) sol
JOIN (
    SELECT
        pressure.data_set_id,
        hydrogen_to_carbon.value AS hydrogen_to_carbon_molar,
        oxygen_to_carbon.value AS oxygen_to_carbon_molar,
        pressure.value AS pressure_bar,
        temperature.value AS temperature_kelvin
    FROM pressure
    INNER JOIN temperature
    ON pressure.data_set_id=temperature.data_set_id
    INNER JOIN hydrogen_to_carbon
    ON pressure.data_set_id=hydrogen_to_carbon.data_set_id
    INNER JOIN oxygen_to_carbon
    ON hydrogen_to_carbon.data_set_id=oxygen_to_carbon.data_set_id
) params
ON sol.data_set_id=params.data_set_id
;