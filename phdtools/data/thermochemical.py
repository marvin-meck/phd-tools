"""phdtools.data.thermochemical.py

Copyright 2025 Marvin Meck

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import sqlite3

import numpy as np
import pandas as pd

from phdtools import DATA_DIR
from phdtools.data.constants import GAS_CONST_SI
from phdtools.data import Compound, Reaction

STST_PRESSURE_BAR = 1

query_template = """SELECT 
    t."T(K)",
    c."formula",
    t."{}"
FROM (
    SELECT 
        "jcode",
        "T(K)",
        "{}"
    FROM thermo_tables
    WHERE jcode IN (
        SELECT jcode FROM compounds WHERE formula IN 
        (
            {}
        )
    )
    AND "T(K)" BETWEEN {} AND {}
) t
JOIN compounds c on c.jcode = t.jcode
;
"""

__stoichiometric_number = {
    "SMR": {"C1H4(g)": -1, "H2O1(g)": -1, "C1O1(g)": 1, "H2(ref)": 3},
    "WGS": {"H2O1(g)": -1, "C1O1(g)": -1, "H2(ref)": 1, "C1O2(g)": 1},
    "DSR": {"C1H4(g)": -1, "H2O1(g)": -2, "H2(ref)": 4, "C1O2(g)": 1},
    "MCR1": {"C1H4(g)": -1, "O2(ref)": -2, "C1O2(g)": 1, "H2O1(g)": 2},
    "MCR2": {"C1H4(g)": -1, "O2(ref)": -2, "C1O2(g)": 1, "H2O1(l)": 2},
    "HCR1": {"H2(ref)": -2, "O2(ref)": -1, "H2O1(g)": 2},
    "HCR2": {"H2(ref)": -2, "O2(ref)": -1, "H2O1(l)": 2},
}

stoichiometricNumber = pd.DataFrame(
    index=[c.name for c in Compound],
    columns=[r.name for r in Reaction],
    data={
        r.name: {
            c.name: (
                __stoichiometric_number[r.name].get(c.name, 0)
                if r.name in __stoichiometric_number.keys()
                else 0
            )
            for c in Compound
        }
        for r in Reaction
    },
)


def get_stdHeatCapacitySI(
    Tmin,
    Tmax,
    compounds={
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    },
):
    query = query_template.format(
        "Cp",
        "Cp",
        ",\n\t\t\t".join(
            f'"{c.name}"' for c in Compound if c.name in compounds
        ).expandtabs(4),
        Tmin,
        Tmax,
    )
    with sqlite3.connect(
        DATA_DIR / "nist-janaf" / "nist_janaf_thermochemical_tables.sqlite"
    ) as con:
        stdHeatCapacitySI = pd.read_sql(query, con).pivot(
            index="T(K)", columns="FORMULA", values="Cp"
        )
        return stdHeatCapacitySI


def get_stdEntropySI(
    Tmin,
    Tmax,
    compounds={
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    },
):
    query = query_template.format(
        "S",
        "S",
        ",\n\t\t\t".join(
            f'"{c.name}"' for c in Compound if c.name in compounds
        ).expandtabs(4),
        Tmin,
        Tmax,
    )
    with sqlite3.connect(
        DATA_DIR / "nist-janaf" / "nist_janaf_thermochemical_tables.sqlite"
    ) as con:
        stdEntropySI = pd.read_sql(query, con).pivot(
            index="T(K)", columns="FORMULA", values="S"
        )
        return stdEntropySI


def get_stdFormationEnthalpySI(
    Tmin,
    Tmax,
    compounds={
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    },
):
    query = query_template.format(
        "delta-f H",
        "delta-f H",
        ",\n\t\t\t".join(
            f'"{c.name}"' for c in Compound if c.name in compounds
        ).expandtabs(4),
        Tmin,
        Tmax,
    )
    with sqlite3.connect(
        DATA_DIR / "nist-janaf" / "nist_janaf_thermochemical_tables.sqlite"
    ) as con:
        stdFormationEnthalpySI = 1e3 * pd.read_sql(query, con).pivot(
            index="T(K)", columns="FORMULA", values="delta-f H"
        )
        return stdFormationEnthalpySI


def get_stdFormationGibbsEnergySI(
    Tmin,
    Tmax,
    compounds={
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    },
):
    query = query_template.format(
        "delta-f G",
        "delta-f G",
        ",\n\t\t\t".join(
            f'"{c.name}"' for c in Compound if c.name in compounds
        ).expandtabs(4),
        Tmin,
        Tmax,
    )
    with sqlite3.connect(
        DATA_DIR / "nist-janaf" / "nist_janaf_thermochemical_tables.sqlite"
    ) as con:
        stdFormationGibbsEnergySI = 1e3 * pd.read_sql(query, con).pivot(
            index="T(K)", columns="FORMULA", values="delta-f G"
        )
        return stdFormationGibbsEnergySI


def get_stdEnthalpySI(
    Tmin,
    Tmax,
    compounds={
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    },
):
    stdFormationEnthalpyRefSI = get_stdFormationEnthalpySI(
        Tmin=298.15, Tmax=298.15, compounds=compounds
    ).loc[298.15]
    query = query_template.format(
        "H-H(Tr)",
        "H-H(Tr)",
        ",\n\t\t\t".join(
            f'"{c.name}"' for c in Compound if c.name in compounds
        ).expandtabs(4),
        Tmin,
        Tmax,
    )
    with sqlite3.connect(
        DATA_DIR / "nist-janaf" / "nist_janaf_thermochemical_tables.sqlite"
    ) as con:
        fun = 1e3 * pd.read_sql(query, con).pivot(
            index="T(K)", columns="FORMULA", values="H-H(Tr)"
        )

        stdEnthalpySI = fun + stdFormationEnthalpyRefSI

        return stdEnthalpySI


def get_stdGibbsEnergy():
    pass


def get_stdReactionHeatCapacitySI(
    Tmin, Tmax, reactions={"SMR", "WGS", "DSR", "MCR1", "MCR2", "HCR1", "HCR2"}
):
    compounds = set()
    for r in reactions:
        compounds.update(set(stoichiometricNumber[stoichiometricNumber[r] != 0].index))
    stdHeatCapacitySI = get_stdHeatCapacitySI(Tmin=Tmin, Tmax=Tmax, compounds=compounds)
    for s in Compound:
        if s.name not in stdHeatCapacitySI.columns:
            stdHeatCapacitySI[s.name] = np.full(len(stdHeatCapacitySI), np.nan)
    stdReactionHeatCapacitySI = pd.DataFrame(
        index=stdHeatCapacitySI.index, columns=list(reactions)
    )
    for r in reactions:
        mask = stoichiometricNumber[r] != 0
        stdReactionHeatCapacitySI[r] = stdHeatCapacitySI[
            stoichiometricNumber[r][mask].keys()
        ].dot(stoichiometricNumber[r][mask])

    return stdReactionHeatCapacitySI.dropna(how="all")


def get_stdReactionEnthalpySI(
    Tmin, Tmax, reactions={"SMR", "WGS", "DSR", "MCR1", "MCR2", "HCR1", "HCR2"}
):
    compounds = set()
    for r in reactions:
        compounds.update(set(stoichiometricNumber[stoichiometricNumber[r] != 0].index))
    stdFormationEnthalpySI = get_stdFormationEnthalpySI(
        Tmin=Tmin, Tmax=Tmax, compounds=compounds
    )
    for s in Compound:
        if s.name not in stdFormationEnthalpySI.columns:
            stdFormationEnthalpySI[s.name] = np.full(
                len(stdFormationEnthalpySI), np.nan
            )
    stdReactionEnthalpySI = pd.DataFrame(
        index=stdFormationEnthalpySI.index, columns=list(reactions)
    )
    for r in reactions:
        mask = stoichiometricNumber[r] != 0
        stdReactionEnthalpySI[r] = stdFormationEnthalpySI[
            stoichiometricNumber[r][mask].keys()
        ].dot(stoichiometricNumber[r][mask])

    return stdReactionEnthalpySI.dropna(how="all")


def get_stdReactionEntropySI(
    Tmin, Tmax, reactions={"SMR", "WGS", "DSR", "MCR1", "MCR2", "HCR1", "HCR2"}
):
    compounds = set()
    for r in reactions:
        compounds.update(set(stoichiometricNumber[stoichiometricNumber[r] != 0].index))
    stdEntropySI = get_stdEntropySI(Tmin=Tmin, Tmax=Tmax, compounds=compounds)
    for s in Compound:
        if s.name not in stdEntropySI.columns:
            stdEntropySI[s.name] = np.full(len(stdEntropySI), np.nan)
    stdReactionEntropySI = pd.DataFrame(
        index=stdEntropySI.index, columns=list(reactions)
    )
    for r in reactions:
        mask = stoichiometricNumber[r] != 0
        stdReactionEntropySI[r] = stdEntropySI[
            stoichiometricNumber[r][mask].keys()
        ].dot(stoichiometricNumber[r][mask])

    return stdReactionEntropySI.dropna(how="all")


def get_stdReactionGibbsEnergySI(
    Tmin, Tmax, reactions={"SMR", "WGS", "DSR", "MCR1", "MCR2", "HCR1", "HCR2"}
):
    compounds = set()
    for r in reactions:
        compounds.update(set(stoichiometricNumber[stoichiometricNumber[r] != 0].index))
    stdFormationGibbsEnergySI = get_stdFormationGibbsEnergySI(
        Tmin=Tmin, Tmax=Tmax, compounds=compounds
    )
    for s in Compound:
        if s.name not in stdFormationGibbsEnergySI.columns:
            stdFormationGibbsEnergySI[s.name] = np.full(
                len(stdFormationGibbsEnergySI), np.nan
            )
    stdReactionGibbsEnergySI = pd.DataFrame(
        index=stdFormationGibbsEnergySI.index, columns=list(reactions)
    )
    for r in reactions:
        mask = stoichiometricNumber[r] != 0
        stdReactionGibbsEnergySI[r] = stdFormationGibbsEnergySI[
            stoichiometricNumber[r][mask].keys()
        ].dot(stoichiometricNumber[r][mask])

    return stdReactionGibbsEnergySI.dropna(how="all")


def get_logEquilibriumConst(
    Tmin, Tmax, reactions={"SMR", "WGS", "DSR", "MCR1", "MCR2", "HCR1", "HCR2"}
):
    stdReactionGibbsEnergySI = get_stdReactionGibbsEnergySI(
        Tmin=Tmin, Tmax=Tmax, reactions=reactions
    )
    return -(stdReactionGibbsEnergySI) / (
        GAS_CONST_SI * stdReactionGibbsEnergySI.index.to_numpy()[:, np.newaxis]
    )


def get_stdReactionEnthalpyFromKirchhoffsLaw(
    temperatureKelvin,
    Tmin=298.15,
    Tmax=500,
    reactions={"MCR2", "MCR1", "HCR1", "WGS", "HCR2", "SMR", "DSR"},
):
    """see Atkins, Paula, and Keeler (2023, eq. (2C.7a))."""
    stdReactionEnthalpySI = get_stdReactionEnthalpySI(
        Tmin, Tmax, reactions=reactions
    ).dropna()
    stdReactionHeatCapacitySI = get_stdReactionHeatCapacitySI(
        Tmin, Tmax, reactions=reactions
    ).dropna()

    idx = abs(stdReactionEnthalpySI.index - temperatureKelvin).argmin()

    # print(idx,stdReactionEnthalpySI.index[idx],abs(stdReactionEnthalpySI.index - temperatureKelvin))

    return (
        stdReactionEnthalpySI.iloc[idx]
        + stdReactionHeatCapacitySI.iloc[idx]
        * (temperatureKelvin - stdReactionEnthalpySI.index[idx])
    ).rename(temperatureKelvin)


def get_stdEnthalpyFromIntegration(
    temperatureKelvin,
    Tmin=298.15,
    Tmax=500,
    compounds={
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    },
):
    """see Atkins, Paula, and Keeler (2023, eq. (2B.6))."""
    stdEnthalpySI = get_stdEnthalpySI(Tmin, Tmax, compounds=compounds).dropna()
    stdHeatCapacitySI = get_stdHeatCapacitySI(Tmin, Tmax, compounds=compounds).dropna()

    idx = abs(stdEnthalpySI.index - temperatureKelvin).argmin()

    # print(idx,stdReactionEnthalpySI.index[idx],abs(stdReactionEnthalpySI.index - temperatureKelvin))

    return (
        stdEnthalpySI.iloc[idx]
        + stdHeatCapacitySI.iloc[idx] * (temperatureKelvin - stdEnthalpySI.index[idx])
    ).rename(temperatureKelvin)


if __name__ == "__main__":
    pass
