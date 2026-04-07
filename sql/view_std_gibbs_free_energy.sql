CREATE VIEW [std_gibbs_free_energy] AS
WITH gef AS (
    SELECT 
        c.jcode,
        c."formula",
        t."T(K)", -- in K
        t."-[G-H(Tr)]/T" -- in J/(mol K)
    FROM thermo_tables t
    JOIN compounds c on c.jcode = t.jcode
), 
dfH0 AS (
    SELECT 
        c.jcode,
        t."delta-f H" AS "dfH(298.15K)" -- in kJ/mol # Gordon and McBride (1994): NASA Reference Publication 1311, ch. 4, p. 19
    FROM (
        SELECT 
            "jcode",
            "delta-f H" 
        FROM thermo_tables
        WHERE "T(K)" == 298.15
    ) t
    JOIN compounds c on c.jcode = t.jcode
)
SELECT 
    "T(K)", 
    "formula" AS "FORMULA",
    -1e-3 * ( "-[G-H(Tr)]/T" * "T(K)" ) + "dfH(298.15K)" AS "G" -- in kJ/(mol K)
FROM gef g 
JOIN dfH0 h
ON g.jcode = h.jcode
;