CREATE VIEW [yield_conversion] AS
WITH methane_out AS (
    SELECT 
        solution_id,
        value AS amount_mol
    FROM variable_data
    WHERE var_id = (
        SELECT var_id 
        FROM variables
        WHERE model_id = 1 AND var_name = 'amount_substance' and index_str="C1H4(g)"
    )
),
hydrogen_out AS (
    SELECT 
        solution_id,
        value AS amount_mol
    FROM variable_data
    WHERE var_id = (
        SELECT var_id 
        FROM variables
        WHERE model_id = 1 AND var_name = 'amount_substance' and index_str="H2(ref)"
    )
),
methane_in AS (
    SELECT
        solutions.solution_id, 
        solutions.data_set_id,
        value AS amount_mol
    FROM parameter_data
    INNER JOIN solutions
    ON 
        solutions.data_set_id=parameter_data.data_set_id
    WHERE parameter_data.param_id=(
        SELECT param_id FROM parameters
        WHERE model_id=1 AND param_name="amount_element" AND index_str="C"
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
steam_to_carbon AS (
    SELECT 
        c.data_set_id, 
        (0.5*h.value/c.value - 2) AS value
    FROM amount_carbon c
    JOIN amount_hydrogen h
    ON c.data_set_id=h.data_set_id
)
SELECT
    temperature_kelvin,
    conversion,
    yield
FROM (
    SELECT
        methane_in.data_set_id,
        methane_out.solution_id,
        1 - methane_out.amount_mol/methane_in.amount_mol AS conversion,
        hydrogen_out.amount_mol / (methane_in.amount_mol-methane_out.amount_mol) AS yield
    FROM methane_in
    INNER JOIN methane_out
    ON methane_in.solution_id = methane_out.solution_id
    INNER JOIN hydrogen_out
    ON hydrogen_out.solution_id = methane_out.solution_id
) sol
INNER JOIN (
    SELECT
        pressure.data_set_id,
        steam_to_carbon.value AS steam_to_carbon_molar, 
        pressure.value AS pressure_bar,
        temperature.value AS temperature_kelvin
    FROM pressure
    INNER JOIN temperature
    ON pressure.data_set_id=temperature.data_set_id
    INNER JOIN steam_to_carbon
    ON pressure.data_set_id=steam_to_carbon.data_set_id
) params
ON sol.data_set_id=params.data_set_id
;