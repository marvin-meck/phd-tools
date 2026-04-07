SELECT 
    t."T(K)",
    c."formula",
    t."delta-f G" -- in kJ / mol
FROM (
    SELECT 
        "jcode",
        "T(K)",
        "delta-f G"
    FROM thermo_tables
    WHERE jcode IN (
        SELECT jcode FROM compounds WHERE formula IN 
        (
            "C1H4(g)",
            "C1O1(g)",
            "C1O2(g)",
            "H2(ref)",
            "H2O1(g)",
            "H2O1(l)",
            "N2(ref)",
            "O2(ref)"
        )
    )
    AND "T(K)" BETWEEN 298.15 AND 1600
) t
JOIN compounds c on c.jcode = t.jcode
;