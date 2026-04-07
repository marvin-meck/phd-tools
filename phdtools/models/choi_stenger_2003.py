"""phdtools.models.choi_stenger_2003.py

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

Model for the water-gas shift reaction kinetics by Choi and Stenger (2003).

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de


References
----------

Choi, Y. and Stenger, H.G. (2003) ‘Water gas shift reaction kinetics and reactor
    modeling for fuel cell grade hydrogen’, Journal of Power Sources, 124(2),
    pp. 432–439. Available at: https://doi.org/10.1016/S0378-7753(03)00614-1.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize_scalar

from phdtools.data.constants import GAS_CONST_SI
from phdtools.data.thermochemical import (
    STST_PRESSURE_BAR,
    Compound,
    stoichiometricNumber,
    get_logEquilibriumConst,
    get_stdReactionEnthalpySI,
)

REACTOR_DIAMETER_METER = 0.5 * 25.4e-3
REACTOR_LENGTH_METER = 12 * 25.4e-3
REACTOR_VOLUME_CUBIC_METER = (
    REACTOR_LENGTH_METER * (np.pi / 4) * REACTOR_DIAMETER_METER**2
)

CATALYST_MASS_KILOGRAMM = 1e-3
CATALYST_BULK_DENSITY_SI = 1200  # estimate
CATALYST_VOLUME_CUBIC_METER = CATALYST_MASS_KILOGRAMM / CATALYST_BULK_DENSITY_SI

stdReactionEnthalpySI = get_stdReactionEnthalpySI(Tmin=298.15, Tmax=1000)
logEquilibriumConst = get_logEquilibriumConst(Tmin=298.15, Tmax=1000)

refTemperatureEquilibriumSI = 500

equilibriumConstRef = np.exp(
    logEquilibriumConst.loc[refTemperatureEquilibriumSI, "WGS"]
)
enthalpyReactionSI = stdReactionEnthalpySI.loc[refTemperatureEquilibriumSI, "WGS"]


def equilibriumConstModel(temperatureKelvin, model="vantHoff"):
    """van't Hoff equation for the equilibrium coefficients Ki"""
    T = np.atleast_1d(temperatureKelvin)
    if model == "choi11":
        f = lambda T: np.exp(
            5693.5 / T[:, np.newaxis]
            + 1.077 * np.log(T[:, np.newaxis])
            + 5.44e-4 * T[:, np.newaxis]
            - 1.125e-7 * T[:, np.newaxis] ** 2
            - 49170 / T[:, np.newaxis] ** 2
            - 13.148
        )
    elif model == "choi12":
        f = lambda T: np.exp(4577.8 / T[:, np.newaxis] - 4.33)
    elif model == "vantHoff":
        f = lambda T: equilibriumConstRef * np.exp(
            -enthalpyReactionSI
            / GAS_CONST_SI
            * ((1 / T)[:, np.newaxis] - 1 / refTemperatureEquilibriumSI)
        )
    return f(T)


def rateConstModel(temperatureKelvin):
    """ """
    T = np.atleast_1d(temperatureKelvin)

    # convert k0 from mol / (g(cat) h atm*2) to  mol / (kg(cat) s)
    stst_pressure_atm = (1 / 1.01325) * STST_PRESSURE_BAR

    k0 = 2.96e5 * stst_pressure_atm**2 / 3.6
    E = 47.4e3
    # k0 = 17.973 * stst_pressure_atm**2 * 1000
    # E = 43.7e3

    return k0 * np.exp(-E / (GAS_CONST_SI * T[:, np.newaxis]))


def stoichiometryShift(moleFractionIn, conversion):
    moleFractionOut = np.zeros(moleFractionIn.shape)

    moleFractionOut[Compound["C1O1(g)"].value] = moleFractionIn[
        Compound["C1O1(g)"].value
    ] * (1 - conversion)
    moleFractionOut[Compound["H2O1(g)"].value] = moleFractionIn[
        Compound["C1O1(g)"].value
    ] * (
        moleFractionIn[Compound["H2O1(g)"].value]
        / moleFractionIn[Compound["C1O1(g)"].value]
        - conversion
    )
    moleFractionOut[Compound["C1O2(g)"].value] = moleFractionIn[
        Compound["C1O1(g)"].value
    ] * (
        moleFractionIn[Compound["C1O2(g)"].value]
        / moleFractionIn[Compound["C1O1(g)"].value]
        + conversion
    )
    moleFractionOut[Compound["H2(ref)"].value] = moleFractionIn[
        Compound["C1O1(g)"].value
    ] * (
        moleFractionIn[Compound["H2(ref)"].value]
        / moleFractionIn[Compound["C1O1(g)"].value]
        + conversion
    )

    for c in Compound:
        if c.name not in stoichiometricNumber["WGS"].keys():
            moleFractionOut[c.value] = moleFractionIn[c.value]

    return moleFractionOut


def equilibriumConversionWGS(moleFractionIn, temperatureSI, model="vantHoff"):

    Keq = equilibriumConstModel(temperatureSI, model)[0, 0]

    def obj(x):
        moleFractionOut = stoichiometryShift(moleFractionIn, x)
        return (
            moleFractionOut[Compound["H2O1(g)"].value]
            * moleFractionOut[Compound["C1O1(g)"].value]
            - moleFractionOut[Compound["H2(ref)"].value]
            * moleFractionOut[Compound["C1O2(g)"].value]
            / Keq
        ) ** 2

    res = minimize_scalar(obj, bounds=(0, 1))

    return res.x


def reactionRateModel(partialPressureBar, temperatureKelvin, model="vantHoff"):

    k = rateConstModel(temperatureKelvin)[0]
    Keq = equilibriumConstModel(temperatureKelvin, model=model)[0]

    a = partialPressureBar / STST_PRESSURE_BAR  # ideal gas activities

    r = k * (
        a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value]
        - a[Compound["C1O2(g)"].value] * a[Compound["H2(ref)"].value] / Keq
    )

    return r


def initialValueProblem(
    t, y, moleFractionIn, temperatureKelvin, pressureBar, model="vantHoff"
):

    conversion = y[0]
    moleFractionOut = stoichiometryShift(moleFractionIn, conversion)
    # assert all(np.isclose(moleFractionOut.sum(axis=0),1))
    partialPressureBar = pressureBar * moleFractionOut

    r = reactionRateModel(partialPressureBar, temperatureKelvin, model=model)
    return np.array([r])


def initialValueProblemConversion(
    t, y, moleFractionIn, temperatureKelvin, pressureBar, model="vantHoff"
):

    conversion = t
    moleFractionOut = stoichiometryShift(moleFractionIn, conversion)
    # assert all(np.isclose(moleFractionOut.sum(axis=0),1))
    partialPressureBar = pressureBar * moleFractionOut

    r = reactionRateModel(partialPressureBar, temperatureKelvin, model=model)
    return np.array([1 / r])


if __name__ == "__main__":
    pass
