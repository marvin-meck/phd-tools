CREATE VIEW [reaction_quotients] AS
SELECT
    temperature_kelvin,
    pressure_bar,
    oxygen_to_carbon_molar,
    hydrogen_to_carbon_molar,
    SUM(DISTINCT LN(
        POWER(pressure_bar * mole_fraction,
            CASE 
                WHEN compound = 'C1H4(g)' THEN -1
                WHEN compound = 'H2O1(g)' THEN -1
                WHEN compound = 'C1O1(g)' THEN 1
                WHEN compound = 'H2(ref)' THEN 3
                WHEN compound = 'C1O2(g)' THEN 0
                ELSE 0 -- Default case if compound doesn't match any of the above
            END
        )
    )) AS "log Q1* (SR)", 
    SUM(DISTINCT LN(
        POWER(pressure_bar * mole_fraction,
            CASE 
                WHEN compound = 'C1H4(g)' THEN 0
                WHEN compound = 'H2O1(g)' THEN -1
                WHEN compound = 'C1O1(g)' THEN -1
                WHEN compound = 'H2(ref)' THEN 1
                WHEN compound = 'C1O2(g)' THEN 1
                ELSE 0 -- Default case if compound doesn't match any of the above
            END
        )
    )) AS "log Q2* (WGS)",
    SUM(DISTINCT LN(
        POWER(pressure_bar * mole_fraction,
            CASE 
                WHEN compound = 'C1H4(g)' THEN -1
                WHEN compound = 'H2O1(g)' THEN -2
                WHEN compound = 'C1O1(g)' THEN 0
                WHEN compound = 'H2(ref)' THEN 4
                WHEN compound = 'C1O2(g)' THEN 1
                ELSE 0 -- Default case if compound doesn't match any of the above
            END
        )
    )) AS "log Q3* (DSR)"
FROM mole_fractions 
GROUP BY temperature_kelvin, pressure_bar, oxygen_to_carbon_molar,hydrogen_to_carbon_molar
ORDER BY temperature_kelvin
;