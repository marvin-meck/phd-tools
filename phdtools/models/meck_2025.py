"""phdtools.models.meck_2025.py

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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from phdtools.data import (
    Compound,
    ISO_STD_REF_PRESSURE_SI,
    ISO_STD_REF_TEMPERATURE_SI,
    ISO_STD_REF_REL_HUMIDITY,
)

from phdtools.data.constants import FARADAY_CONST_SI, GAS_CONST_SI

from phdtools.data.thermophysical import (
    vapourPressureModel,
    waterVapourPressureOverH3PO4Model,
    get_moleFractionH3PO4,
    moleFractionFromDry,
)
from phdtools.data.diffusion import pressureDiffusivityProductModel

from phdtools.models.ohayre_2016 import (
    ELECTRODE_POROSITY,
    ELECTRONS_TRANSFERRED,
    reversibleCellPotentialModel,
    activationOverpotentialModel,
    concentrationOverpotentialModel,
)

from phdtools.models.mamlouk_sousa_scott_2011 import get_electrolyteFilmActivities

from phdtools.models.zhang_2007 import (
    ModelParameters as KineticParameters,
    MEMBRANE_THICKNESS_SI,
    conductivityModel,
    exchangeCurrentDensityModel,
    transferCoefModel,
    ELECTRONS_TRANSFERRED_ORR,
)

DIFFUSION_LAYER_THICKNESS_SI = 500e-6
FUEL_CELL_MASS_FRACTION_H3PO4 = 0.9
# 250e-6 # (Amphlett et al., 1995)


@dataclass
class ModelParameters:
    temperatureKelvin: float
    reversibleCellPotentialSI: float
    exchangeCurrentDensityCathodeSI: float
    transferCoefficientCathode: float
    areaSpecificResistanceSI: float
    limitingCurrentDensitySI: float


def get_reversibleCellVoltage(
    moleFractionAnodeIn,
    moleFractionCathodeIn,
    temperatureKelvin,
    pressureBar,
    massFractionPhosphoricAcid,
):

    # Compute reversible cell potential
    activity = get_electrolyteFilmActivities(
        moleFractionAnodeIn,
        moleFractionCathodeIn,
        temperatureKelvin,
        pressureBar,
        massFractionPhosphoricAcid,
    )

    # assume constant electrolyte composition despite pH2O = 0 at the cathode
    vapourPressureOverPhosphoricAcidBar = (
        waterVapourPressureOverH3PO4Model(
            moleFractionH3PO4=get_moleFractionH3PO4(massFractionPhosphoricAcid),
            temperatureKelvin=temperatureKelvin,
        )[0, 0]
        * 1e-5
    )

    activity[Compound["H2O1(g)"].value] = vapourPressureOverPhosphoricAcidBar / (
        vapourPressureModel(temperatureKelvin)[0] * 1e-5
    )

    reversibleCellPotentialSI = reversibleCellPotentialModel(
        temperatureKelvin, activity
    )

    return reversibleCellPotentialSI


def stefanMaxwellInitialValueProblemCathode(
    z: npt.ArrayLike,
    x: npt.ArrayLike,
    currentDensitySI: npt.ArrayLike,
    temperatureKelvin: npt.ArrayLike,
    pressureBar: npt.ArrayLike,
) -> np.ndarray:
    """see Bird, Stewart and Lightfoot (2001)

    References
    ----------
    Amphlett, J.C. et al. (1995) “Performance Modeling of the Ballard Mark IV
        Solid Polymer Electrolyte Fuel Cell: I . Mechanistic method Development,”
        Journal of The Electrochemical Society, 142(1), pp. 1–8.
        Available at: https://doi.org/10.1149/1.2043866.

    Bernardi, D.M. and Verbrugge, M.W. (1991) “Mathematical method of a gas diffusion
        electrode bonded to a polymer electrolyte,” AIChE Journal, 37(8), pp. 1151–1163.
        Available at: https://doi.org/10.1002/aic.690370805.

    Bird, R.B., Stewart, W.E. and Lightfoot, E.N. (2001) Transport phenomena.
        2nd ed. New York; Chichester: Wiley.

    Springer, T.E., Zawodzinski, T.A. and Gottesfeld, S. (1991) “Polymer Electrolyte
        Fuel Cell Model,” Journal of The Electrochemical Society, 138(8), pp. 2334–2342.
        Available at: https://doi.org/10.1149/1.2085971.
    """
    dxdz = np.full(len(Compound), np.nan)
    N = np.zeros(len(Compound))

    N[Compound["O2(ref)"].value] = currentDensitySI / (4 * FARADAY_CONST_SI)
    N[Compound["H2O1(g)"].value] = -1 * currentDensitySI / (2 * FARADAY_CONST_SI)

    c = pressureBar * 1e5 / (GAS_CONST_SI * temperatureKelvin)

    D = np.full((len(Compound), len(Compound)), np.nan)

    for c1 in Compound:
        for c2 in Compound:
            D[c1.value, c2.value] = D[c2.value, c1.value] = (
                pressureDiffusivityProductModel(c1, c2, temperatureKelvin)[0]
                / (1e5 * pressureBar)
            )

    # Bruggemann correction
    Deff = D * ELECTRODE_POROSITY**1.5

    for k in Compound:
        dxdz[k.value] = DIFFUSION_LAYER_THICKNESS_SI * (
            -1
            / c
            * sum(
                1
                / Deff[k.value, j.value]
                * (x[j.value] * N[k.value] - x[k.value] * N[j.value])
                for j in Compound
                if j.value != k.value
            )
        )

    return dxdz


def stefanMaxwellInitialValueProblemAnode(
    z: npt.ArrayLike,
    x: npt.ArrayLike,
    currentDensitySI: npt.ArrayLike,
    temperatureKelvin: npt.ArrayLike,
    pressureBar: npt.ArrayLike,
) -> np.ndarray:
    """see Bird, Stewart and Lightfoot (2001)

    References
    ----------
    Amphlett, J.C. et al. (1995) “Performance Modeling of the Ballard Mark IV
        Solid Polymer Electrolyte Fuel Cell: I . Mechanistic Model Development,”
        Journal of The Electrochemical Society, 142(1), pp. 1–8.
        Available at: https://doi.org/10.1149/1.2043866.

    Bernardi, D.M. and Verbrugge, M.W. (1991) “Mathematical model of a gas diffusion
        electrode bonded to a polymer electrolyte,” AIChE Journal, 37(8), pp. 1151–1163.
        Available at: https://doi.org/10.1002/aic.690370805.

    Bird, R.B., Stewart, W.E. and Lightfoot, E.N. (2001) Transport phenomena.
        2nd ed. New York; Chichester: Wiley.

    Springer, T.E., Zawodzinski, T.A. and Gottesfeld, S. (1991) “Polymer Electrolyte
        Fuel Cell Model,” Journal of The Electrochemical Society, 138(8), pp. 2334–2342.
        Available at: https://doi.org/10.1149/1.2085971.
    """
    dxdz = np.full(len(Compound), np.nan)
    N = np.zeros(len(Compound))

    N[Compound["H2(ref)"].value] = currentDensitySI / (2 * FARADAY_CONST_SI)

    c = pressureBar * 1e5 / (GAS_CONST_SI * temperatureKelvin)

    D = np.full((len(Compound), len(Compound)), np.nan)

    for c1 in Compound:
        for c2 in Compound:
            D[c1.value, c2.value] = D[c2.value, c1.value] = (
                pressureDiffusivityProductModel(c1, c2, temperatureKelvin)[0]
                / (1e5 * pressureBar)
            )

    # Bruggemann correction
    Deff = D * ELECTRODE_POROSITY**1.5

    for k in Compound:
        dxdz[k.value] = DIFFUSION_LAYER_THICKNESS_SI * (
            -1
            / c
            * sum(
                1
                / Deff[k.value, j.value]
                * (x[j.value] * N[k.value] - x[k.value] * N[j.value])
                for j in Compound
                if j.value != k.value
            )
        )

    return dxdz


def diffusionModelCathode(
    currentDensitySI,
    moleFractionIn,
    temperatureKelvin,
    pressureBar,
    coordinate=1,
    method="analytical",
):

    j = np.atleast_1d(currentDensitySI)
    z_plus = np.atleast_1d(coordinate)

    moleFraction = np.zeros((len(Compound), len(z_plus), len(j)))

    c = pressureBar * 1e5 / (GAS_CONST_SI * temperatureKelvin)

    D = np.full((len(Compound), len(Compound)), np.nan)

    for c1 in Compound:
        for c2 in Compound:
            D[c1.value, c2.value] = D[c2.value, c1.value] = (
                pressureDiffusivityProductModel(c1, c2, temperatureKelvin)[0]
                / (pressureBar * 1e5)
            )

    Deff = D * ELECTRODE_POROSITY**1.5

    if method == "analytical":

        A1 = (
            1
            / c
            * (
                1 / Deff[Compound["N2(ref)"].value, Compound["O2(ref)"].value]
                - 2 * 1 / Deff[Compound["N2(ref)"].value, Compound["H2O1(g)"].value]
            )
            * DIFFUSION_LAYER_THICKNESS_SI
            / (4 * FARADAY_CONST_SI)
        )

        A2 = (
            1
            / c
            * (
                1 / Deff[Compound["O2(ref)"].value, Compound["H2O1(g)"].value]
                - 1 / Deff[Compound["O2(ref)"].value, Compound["N2(ref)"].value]
            )
            * DIFFUSION_LAYER_THICKNESS_SI
            / (4 * FARADAY_CONST_SI)
        )

        A3 = (
            -1
            / c
            * 1
            / Deff[Compound["O2(ref)"].value, Compound["H2O1(g)"].value]
            * DIFFUSION_LAYER_THICKNESS_SI
            / (4 * FARADAY_CONST_SI)
        )

        moleFraction[Compound["N2(ref)"].value] = moleFractionIn[
            Compound["N2(ref)"].value
        ] * np.exp(A1 * z_plus[:, np.newaxis] * j)

        moleFraction[Compound["O2(ref)"].value] = (
            A2
            / (A1 - A3)
            * moleFractionIn[Compound["N2(ref)"].value]
            * np.exp(A1 * z_plus[:, np.newaxis] * j)
            + (
                1
                + moleFractionIn[Compound["O2(ref)"].value]
                - A2 / (A1 - A3) * moleFractionIn[Compound["N2(ref)"].value]
            )
            * np.exp(A3 * z_plus[:, np.newaxis] * j)
            - 1
        )

        moleFraction[Compound["H2O1(g)"].value] = (
            1
            - moleFraction[Compound["O2(ref)"].value]
            - moleFraction[Compound["N2(ref)"].value]
        )

    elif method == "numerical":
        for num, currentDensitySI in enumerate(j):
            sol = solve_ivp(
                fun=stefanMaxwellInitialValueProblemCathode,
                t_span=(0, z_plus.max()),
                y0=moleFractionIn,
                method="RK45",
                t_eval=z_plus,
                dense_output=False,
                events=None,
                vectorized=False,
                args=(currentDensitySI, temperatureKelvin, pressureBar),
            )

            moleFraction[:, :, num] = sol.y
    else:
        raise ValueError(
            f'No such method {method}. Use either "numerical" or "analytical"'
        )

    return moleFraction


def diffusionModelAnode(
    currentDensitySI,
    moleFractionIn,
    temperatureKelvin,
    pressureBar,
    coordinate=1,
    method="analytical",
):

    j = np.atleast_1d(currentDensitySI)
    z_plus = np.atleast_1d(coordinate)

    moleFraction = np.zeros((len(Compound), len(z_plus), len(j)))

    c = pressureBar * 1e5 / (GAS_CONST_SI * temperatureKelvin)

    D = np.full((len(Compound), len(Compound)), np.nan)

    for c1 in Compound:
        for c2 in Compound:
            D[c1.value, c2.value] = D[c2.value, c1.value] = (
                pressureDiffusivityProductModel(c1, c2, temperatureKelvin)[0]
                / (pressureBar * 1e5)
            )

    Deff = D * ELECTRODE_POROSITY**1.5

    if method == "analytical":
        for i in Compound:
            if i.name != "H2(ref)":
                Bi = (
                    1
                    / c
                    * 1
                    / Deff[i.value, Compound["H2(ref)"].value]
                    * DIFFUSION_LAYER_THICKNESS_SI
                    / (2 * FARADAY_CONST_SI)
                )

                moleFraction[i.value] = moleFractionIn[i.value] * np.exp(
                    Bi * z_plus[:, np.newaxis] * j
                )

        moleFraction[Compound["H2(ref)"].value] = 1 - sum(
            moleFraction[j.value] for j in Compound if j.name != "H2(ref)"
        )

    elif method == "numerical":
        for num, currentDensitySI in enumerate(j):
            sol = solve_ivp(
                fun=stefanMaxwellInitialValueProblemAnode,
                t_span=(0, z_plus.max()),
                y0=moleFractionIn,
                method="RK45",
                t_eval=z_plus,
                dense_output=False,
                events=None,
                vectorized=False,
                args=(currentDensitySI, temperatureKelvin, pressureBar),
            )

            moleFraction[:, :, num] = sol.y
    else:
        raise ValueError(
            f'No such method {method}. Use either "numerical" or "analytical"'
        )

    return moleFraction


def get_limitingCurrentDensityCathode(
    moleFractionIn: npt.ArrayLike,
    temperatureKelvin: float,
    pressureBar: float,
    fun=diffusionModelCathode,
) -> float:

    sol = root_scalar(
        lambda currentDensitySI: fun(
            currentDensitySI,
            moleFractionIn,
            temperatureKelvin,
            pressureBar,
            coordinate=1,
            method="analytical",
        )[Compound["O2(ref)"].value, 0, 0],
        x0=0,
    )

    return sol.root


def get_limitingCurrentDensityAnode(
    moleFractionIn: npt.ArrayLike,
    temperatureKelvin: float,
    pressureBar: float,
    fun=diffusionModelAnode,
) -> float:

    sol = root_scalar(
        lambda currentDensitySI: fun(
            currentDensitySI,
            moleFractionIn,
            temperatureKelvin,
            pressureBar,
            coordinate=1,
            method="analytical",
        )[Compound["H2(ref)"].value, 0, 0],
        x0=0,
    )

    return sol.root


def get_CathodeFeedComposition(
    temperatureKelvin=ISO_STD_REF_TEMPERATURE_SI,
    pressureSI=ISO_STD_REF_PRESSURE_SI,
    relHumidity=ISO_STD_REF_REL_HUMIDITY,
):

    dryMoleFractionAir = np.zeros(len(Compound))

    dryMoleFractionAir[Compound["O2(ref)"].value] = 0.2095
    dryMoleFractionAir[Compound["N2(ref)"].value] = 0.7905

    moleFractionCathodeIn = moleFractionFromDry(
        dryMoleFraction=dryMoleFractionAir,
        temperatureKelvin=temperatureKelvin,
        pressureSI=pressureSI,
        relHumidity=relHumidity,
    )

    return moleFractionCathodeIn


def get_fuelCellParameters(
    temperatureKelvin,
    pressureBar,
    massFractionPhosphoricAcid=FUEL_CELL_MASS_FRACTION_H3PO4,
):

    # Feed composition
    moleFractionAnodeIn = np.zeros(len(Compound))

    moleFractionAnodeIn[Compound["H2(ref)"].value] = 1
    moleFractionCathodeIn = get_CathodeFeedComposition()

    reversibleCellPotentialSI = get_reversibleCellVoltage(
        moleFractionAnodeIn,
        moleFractionCathodeIn,
        temperatureKelvin=temperatureKelvin,
        pressureBar=pressureBar,
        massFractionPhosphoricAcid=massFractionPhosphoricAcid,
    )[0]

    params = KineticParameters.init()

    exchangeCurrentDensityCathodeSI = exchangeCurrentDensityModel(
        params, temperatureKelvin
    )[:, 1][0]
    transferCoefficientCathode = transferCoefModel(temperatureKelvin)[:, 1][0]

    areaSpecificResistanceSI = (
        MEMBRANE_THICKNESS_SI / conductivityModel(params, temperatureKelvin)[0]
    )

    limitingCurrentDensitySI = get_limitingCurrentDensityCathode(
        moleFractionIn=moleFractionCathodeIn,
        temperatureKelvin=temperatureKelvin,
        pressureBar=pressureBar,
    )

    return ModelParameters(
        temperatureKelvin=float(temperatureKelvin),
        reversibleCellPotentialSI=float(reversibleCellPotentialSI),
        exchangeCurrentDensityCathodeSI=float(exchangeCurrentDensityCathodeSI),
        transferCoefficientCathode=float(transferCoefficientCathode),
        areaSpecificResistanceSI=float(areaSpecificResistanceSI),
        limitingCurrentDensitySI=float(limitingCurrentDensitySI),
    )


def fuelCellVoltageModel(
    currentDensitySI: float,
    params: ModelParameters,
) -> float:

    j = currentDensitySI

    E = params.reversibleCellPotentialSI

    R = GAS_CONST_SI
    F = FARADAY_CONST_SI
    T = params.temperatureKelvin

    a = params.transferCoefficientCathode
    j0 = params.exchangeCurrentDensityCathodeSI

    Rs = params.areaSpecificResistanceSI

    jL = params.limitingCurrentDensitySI

    U = (
        E
        - 1 / a * R * T / F * np.log(j / j0)
        + R * T / F * (1 / 4 + 1 / a) * np.log(1 - j / jL)
        - Rs * j
    )

    return U
