"""phdtools.models.mamlouk_sousa_scott_2011.py

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

Description
-----------

Fuel cell model by Mamlouk, Sousa and Scott (2011).

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de

References
----------

Klinedinst, K. et al. (1974) “Oxygen solubility and diffusivity in hot concentrated
    H3PO4,” Journal of Electroanalytical Chemistry and Interfacial Electrochemistry,
    57(3), pp. 281–289. Available at: https://doi.org/10.1016/S0022-0728(74)80053-7.

Mamlouk, M., Sousa, T. and Scott, K. (2011) “A High Temperature Polymer Electrolyte
    Membrane Fuel Cell Model for Reformate Gas,” International Journal of Electro-
    chemistry, 2011, pp. 1–18. Available at: https://doi.org/10.4061/2011/520473.

"""

import numpy as np
import numpy.typing as npt

from phdtools.data import Compound
from phdtools.data.thermophysical import (
    get_molarConcentrationPhosphoricAcid,
    vapourPressureModel,
)
from phdtools.data.solubility import molarOxygenSolubilityInH3PO4

# from phdtools.data.constants import GAS_CONST_SI


def get_electrolyteFilmActivities(
    moleFractionAnodeIn: np.ndarray,
    moleFractionCathodeIn: np.ndarray,
    temperatureKelvin: npt.ArrayLike,
    pressureBar: float,
    massFractionPhosphoricAcid: float = 0.95,
) -> np.ndarray:

    molarDensityElectrolyteSI = get_molarConcentrationPhosphoricAcid(
        massFractionPhosphoricAcid, temperatureKelvin
    )[0, 0]

    concentrationSI = np.full(len(Compound), np.nan)

    for c in Compound:
        if c.name in {"H2(ref)", "O2(ref)"}:
            if c.name == "H2(ref)":
                concentrationSI[c.value] = (
                    4
                    * pressureBar
                    * 1e5
                    * moleFractionAnodeIn[c.value]
                    * molarOxygenSolubilityInH3PO4(
                        temperatureKelvin, massFractionPhosphoricAcid
                    )[0, 0]
                )
            elif c.name == "O2(ref)":
                concentrationSI[c.value] = (
                    pressureBar
                    * 1e5
                    * moleFractionCathodeIn[c.value]
                    * molarOxygenSolubilityInH3PO4(
                        temperatureKelvin, massFractionPhosphoricAcid
                    )[0, 0]
                )
            else:
                raise (ValueError)

    activity = concentrationSI / molarDensityElectrolyteSI

    activity[Compound["H2O1(g)"].value] = (
        moleFractionCathodeIn[Compound["H2O1(g)"].value] * pressureBar * 1e5
    ) / vapourPressureModel(temperatureKelvin)[0]

    return activity
