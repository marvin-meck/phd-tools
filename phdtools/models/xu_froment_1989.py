"""phdtools.models.xu_froment_1989.py

Copyright 2024 Technical University Darmstadt

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

Model for the steam methane reforming reaction kinetics by Xu and Froment (1989a).

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de


References
----------

Xu, J. and Froment, G.F. (1989a) 'Methane steam reforming, methanation and
    water-gas shift: I. Intrinsic kinetics', AIChE Journal, 35(1), pp. 88–96.
    Available at: https://doi.org/10.1002/aic.690350109.

Xu, J. and Froment, G.F. (1989b) 'Methane steam reforming: II. Diffusional
    limitations and reactor simulation', AIChE Journal, 35(1), pp. 97–103.
    Available at: https://doi.org/10.1002/aic.690350110.

"""

from dataclasses import dataclass
from enum import Enum
import sqlite3

import numpy as np
import numpy.typing as npt
import pandas as pd

# from scipy.optimize import minimize
from scipy.optimize import newton

from phdtools import DATA_DIR
from phdtools.data.constants import GAS_CONST_SI
from phdtools.data.thermochemical import (
    STST_PRESSURE_BAR,
    Compound,
    Reaction,
    stoichiometricNumber,
    get_logEquilibriumConst,
    get_stdReactionEnthalpySI,
)

# Compound = Enum(
#     "Compound", ["C1H4(g)", "C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"], start=0
# )

Reaction = Enum("Reaction", ["SMR", "WGS", "DSR"], start=0)

DBFILE = ".pyoptdb/pyoptdb.sqlite3"

logEquilibriumConst = get_logEquilibriumConst(
    600, 1600, reactions={"SMR", "WGS", "DSR"}
)

stdReactionEnthalpySI = get_stdReactionEnthalpySI(
    600, 1600, reactions={"SMR", "WGS", "DSR"}
)


@dataclass
class ModelParameters:
    rateConstantRefSI: np.ndarray
    activationEnergySI: np.ndarray
    equilibriumConstRef: np.array
    enthalpyReactionSI: np.array
    adsorptionCoefRef: np.ndarray
    enthalpyAdsorptionSI: np.ndarray
    refTemperatureEquilibriumSI: np.array
    refTemperatureAdsorptionSI: np.ndarray
    refTemperatureRateSI: np.ndarray

    @classmethod
    def init(
        cls,
        fname=DATA_DIR / "xu-froment-1989" / r"240805_table_5_xu_froment.csv",
    ):
        # Model parameter declarations
        rateConstantRefSI = np.zeros(len(Reaction), dtype=float)
        rateConstantRefSI[:] = np.nan
        adsorptionCoefRef = np.zeros(len(Compound), dtype=np.float64)
        adsorptionCoefRef[:] = np.nan
        activationEnergySI = np.zeros(len(Reaction), dtype=float)
        activationEnergySI[:] = np.nan
        enthalpyAdsorptionSI = np.zeros(len(Compound), dtype=float)
        enthalpyAdsorptionSI[:] = np.nan

        refTemperatureAdsorptionSI = np.zeros(len(Compound), dtype=np.float64)
        refTemperatureAdsorptionSI[:] = np.nan

        refTemperatureRateSI = np.array([648, 648, 648])

        refTemperatureEquilibriumSI = 1000 * np.ones(len(Reaction), dtype=float)

        equilibriumConstRef = np.zeros(len(Reaction))
        equilibriumConstRef[:] = np.nan
        enthalpyReactionSI = np.zeros(len(Reaction))
        enthalpyReactionSI[:] = np.nan

        for r in {"SMR", "WGS", "DSR"}:
            equilibriumConstRef[Reaction[r].value] = np.exp(
                logEquilibriumConst.loc[refTemperatureEquilibriumSI[Reaction[r].value]][
                    r
                ]
            )
            enthalpyReactionSI[Reaction[r].value] = stdReactionEnthalpySI.loc[
                refTemperatureEquilibriumSI[Reaction[r].value]
            ][r]

        tmp = pd.read_csv(fname, nrows=1)

        rateConstantRefSI[Reaction["SMR"].value] = (
            1e3 * tmp.iloc[0, 1] / np.power(STST_PRESSURE_BAR, 0.5)
        ) / 3600  # in mol/(kg cat s)
        rateConstantRefSI[Reaction["WGS"].value] = (
            1e3 * tmp.iloc[0, 2] * STST_PRESSURE_BAR
        ) / 3600  # in mol/(kg cat s)
        rateConstantRefSI[Reaction["DSR"].value] = (
            1e3 * tmp.iloc[0, 3] / np.power(STST_PRESSURE_BAR, 0.5)
        ) / 3600  # in mol/(kg cat s)

        adsorptionCoefRef[Compound["C1O1(g)"].value] = (
            tmp.iloc[0, 4] * STST_PRESSURE_BAR
        )
        adsorptionCoefRef[Compound["H2(ref)"].value] = (
            tmp.iloc[0, 5] * STST_PRESSURE_BAR
        )
        adsorptionCoefRef[Compound["C1H4(g)"].value] = (
            tmp.iloc[0, 6] * STST_PRESSURE_BAR
        )
        adsorptionCoefRef[Compound["H2O1(g)"].value] = tmp.iloc[0, 7]
        # adsorptionCoefRef[Compound["C1O2(g)"].value] = np.nan

        tmp = pd.read_csv(
            fname,
            skiprows=[0, 1, 2, 3, 4, 6],
            nrows=1,
        )

        activationEnergySI[Reaction["SMR"].value] = 1e3 * tmp.iloc[0, 1]
        activationEnergySI[Reaction["WGS"].value] = 1e3 * tmp.iloc[0, 2]
        activationEnergySI[Reaction["DSR"].value] = 1e3 * tmp.iloc[0, 3]

        enthalpyAdsorptionSI[Compound["C1O1(g)"].value] = 1e3 * tmp.iloc[0, 4]
        enthalpyAdsorptionSI[Compound["H2(ref)"].value] = 1e3 * tmp.iloc[0, 5]
        enthalpyAdsorptionSI[Compound["C1H4(g)"].value] = 1e3 * tmp.iloc[0, 6]
        enthalpyAdsorptionSI[Compound["H2O1(g)"].value] = 1e3 * tmp.iloc[0, 7]
        enthalpyAdsorptionSI[Compound["C1O2(g)"].value] = np.nan

        refTemperatureAdsorptionSI[Compound["C1O1(g)"].value] = 648
        refTemperatureAdsorptionSI[Compound["H2(ref)"].value] = 648
        refTemperatureAdsorptionSI[Compound["C1H4(g)"].value] = 823
        refTemperatureAdsorptionSI[Compound["H2O1(g)"].value] = 823
        refTemperatureAdsorptionSI[Compound["C1O2(g)"].value] = np.nan

        return cls(
            rateConstantRefSI=rateConstantRefSI,
            activationEnergySI=activationEnergySI,
            equilibriumConstRef=equilibriumConstRef,
            enthalpyReactionSI=enthalpyReactionSI,
            adsorptionCoefRef=adsorptionCoefRef,
            enthalpyAdsorptionSI=enthalpyAdsorptionSI,
            refTemperatureEquilibriumSI=refTemperatureEquilibriumSI,
            refTemperatureAdsorptionSI=refTemperatureAdsorptionSI,
            refTemperatureRateSI=refTemperatureRateSI,
        )


def equilibriumConstModel(temperatureKelvin, params: ModelParameters):
    """van't Hoff equation for the equilibrium coefficients Ki, i = 1,2,3"""
    T = np.atleast_1d(temperatureKelvin)
    return params.equilibriumConstRef * np.exp(
        -params.enthalpyReactionSI
        / GAS_CONST_SI
        * ((1 / T)[:, np.newaxis] - 1 / params.refTemperatureEquilibriumSI)
    )


def rateConstModel(temperatureKelvin, params: ModelParameters):
    """Arrhenius equation for the rate coefficients ki, i = 1,2,3; c.f. Eq.(6)"""
    T = np.atleast_1d(temperatureKelvin)
    return params.rateConstantRefSI * np.exp(
        -params.activationEnergySI
        / GAS_CONST_SI
        * ((1 / T)[:, np.newaxis] - 1 / params.refTemperatureRateSI)
    )


def adsorptionCoefModel(temperatureKelvin, params: ModelParameters):
    """van't Hoff equation for the adsorption coefficients Kj, j = CO,H2,CH4,H2O; c.f. Eq.(7)"""
    T = np.atleast_1d(temperatureKelvin)
    return params.adsorptionCoefRef * np.exp(
        -params.enthalpyAdsorptionSI
        / GAS_CONST_SI
        * ((1 / T)[:, np.newaxis] - 1 / params.refTemperatureAdsorptionSI)
    )


def reactionRateModel(
    partialPressureBar: np.ndarray, temperatureSI: float, params: ModelParameters
):
    """
    Rate equations ri, i=1,2,3 according to Eq. (3), Xu and Froment (1989).
    Indices i refer to the following reactions:
        1. CH4 + H2O <=> CO + 3H2
        2. CO + H2O <=> CO2 + H2
        3. CH4 + 2 H2O <=> CO2 + 4H2

    References:
    -----------
    Xu, Jianguo; Froment, Gilbert F. (1989): Methane steam reforming, methanation and water-gas shift:
        I. Intrinsic kinetics. In AIChE J. 35 (1), pp. 88–96. DOI: 10.1002/aic.690350109.
    """
    k = rateConstModel(temperatureSI, params)[0]
    Kad = adsorptionCoefModel(temperatureSI, params)[0]
    Keq = equilibriumConstModel(temperatureSI, params)[0]

    a = partialPressureBar / STST_PRESSURE_BAR  # ideal gas activities

    DEN = (
        1
        + Kad[Compound["C1O1(g)"].value] * a[Compound["C1O1(g)"].value]
        + Kad[Compound["H2(ref)"].value] * a[Compound["H2(ref)"].value]
        + Kad[Compound["C1H4(g)"].value] * a[Compound["C1H4(g)"].value]
        + Kad[Compound["H2O1(g)"].value]
        * a[Compound["H2O1(g)"].value]
        / a[Compound["H2(ref)"].value]
    )

    r1 = (
        k[Reaction["SMR"].value]
        / (a[Compound["H2(ref)"].value] ** 2.5)
        * (
            a[Compound["C1H4(g)"].value] * a[Compound["H2O1(g)"].value]
            - a[Compound["H2(ref)"].value] ** 3
            * a[Compound["C1O1(g)"].value]
            / Keq[Reaction["SMR"].value]
        )
        / (DEN**2)
    )

    r2 = (
        k[Reaction["WGS"].value]
        / a[Compound["H2(ref)"].value]
        * (
            a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value]
            - a[Compound["H2(ref)"].value]
            * a[Compound["C1O2(g)"].value]
            / Keq[Reaction["WGS"].value]
        )
        / (DEN**2)
    )

    r3 = (
        k[Reaction["DSR"].value]
        / (a[Compound["H2(ref)"].value] ** 3.5)
        * (
            a[Compound["C1H4(g)"].value] * a[Compound["H2O1(g)"].value] ** 2
            - a[Compound["H2(ref)"].value] ** 4
            * a[Compound["C1O2(g)"].value]
            / Keq[Reaction["DSR"].value]
        )
        / (DEN**2)
    )

    return np.array([r1, r2, r3])


def stoichiometryReformer(molarFlowRateIn: npt.NDArray, conversion: npt.NDArray):
    """
    conversion[0]: X_CH4 := 1 - F_CH4 / F_CH4,0          --> F_CH4 = F_CH4,0 (1 - X_CH4)
    conversion[1]: X_CO2 := (F_CO2,0 - F_CO2) / F_CH4,0  --> F_CO2 = F_CH4,0 (F_CO2,0 / F_CH4,0 - X_CO2)
    """
    molarFlowRateOut = np.zeros((molarFlowRateIn.shape[0], conversion.shape[1]))
    molarFlowRateOut[:] = np.nan

    molarFlowRateOut[Compound["C1H4(g)"].value] = molarFlowRateIn[
        Compound["C1H4(g)"].value
    ] * (1 - conversion[0])
    molarFlowRateOut[Compound["C1O2(g)"].value] = molarFlowRateIn[
        Compound["C1H4(g)"].value
    ] * (
        molarFlowRateIn[Compound["C1O2(g)"].value]
        / molarFlowRateIn[Compound["C1H4(g)"].value]
        - conversion[1]
    )

    _coef = np.zeros(2)

    den = (
        stoichiometricNumber.loc["C1H4(g)", "SMR"]
        * stoichiometricNumber.loc["C1O2(g)", "WGS"]
        - stoichiometricNumber.loc["C1O2(g)", "SMR"]
        * stoichiometricNumber.loc["C1H4(g)", "WGS"]
    )

    for c in Compound:
        _coef[0] = (
            stoichiometricNumber.loc[c.name, "SMR"]
            * stoichiometricNumber.loc["C1O2(g)", "WGS"]
            - stoichiometricNumber.loc["C1O2(g)", "SMR"]
            * stoichiometricNumber.loc[c.name, "WGS"]
        ) / den

        _coef[1] = (
            stoichiometricNumber.loc["C1H4(g)", "SMR"]
            * stoichiometricNumber.loc[c.name, "WGS"]
            - stoichiometricNumber.loc[c.name, "SMR"]
            * stoichiometricNumber.loc["C1H4(g)", "WGS"]
        ) / den

        molarFlowRateOut[c.value] = (
            molarFlowRateIn[c.value]
            + _coef[0]
            * (
                molarFlowRateOut[Compound["C1H4(g)"].value]
                - molarFlowRateIn[Compound["C1H4(g)"].value]
            )
            + _coef[1]
            * (
                molarFlowRateOut[Compound["C1O2(g)"].value]
                - molarFlowRateIn[Compound["C1O2(g)"].value]
            )
        )

    return molarFlowRateOut


def initialValueProblemSpaceTime(
    spaceTime,
    conversion,
    molarFlowRateIn,
    temperatureKelvin,
    pressureBar,
    params: ModelParameters,
):
    """IVP:
    -------

    X_CH4 := 1 - F_CH4/F_CH4,0         --> d(F_CH4/F_CH4,0) = - dX_CH4
    X_CO2 := (F_CO2,0 - F_CO2)/F_CH4,0 --> d(F_CO2/F_CH4,0) = - dX_CO2


    dX_CH4 / d(W/F_CH4,0) = r1 + r3
    dX_CO2 / d(W/F_CH4,0) = -r2 - r3

    where

    F_CH4(W/F_CH4,0 = 0) = F_CH4,0 --> X_CH4,0 = 0
    F_CO2(W/F_CH4,0 = 0) = F_CO2,0 --> X_CO2,0 = 0
    """
    molarFlowRateOut = stoichiometryReformer(molarFlowRateIn, conversion)
    moleFraction = molarFlowRateOut / molarFlowRateOut.sum(axis=0)
    assert all(np.isclose(moleFraction.sum(axis=0), 1))
    partialPressureBar = pressureBar * moleFraction

    r1, r2, r3 = reactionRateModel(partialPressureBar, temperatureKelvin, params)
    return np.array([r1 + r3, -r2 - r3])


spaceTimeIVP = initialValueProblemSpaceTime


def initialValueProblemConversion(
    t, y, molarFlowRateIn, temperatureKelvin, pressureBar, params: ModelParameters
):
    """IVP:
    -------

    X_CH4 := 1 - F_CH4/F_CH4,0         --> d(F_CH4/F_CH4,0) = - dX_CH4
    X_CO2 := (F_CO2,0 - F_CO2)/F_CH4,0 --> d(F_CO2/F_CH4,0) = - dX_CO2

    d(W/F_CH4,0) / dX_CH4 = -1/(-r1 - r3)

    dX_CO2 / dX_CH4 = dX_CO2 / d(W/F_CH4,0) * d(W/F_CH4,0) / dX_CH4
                    = (-r2 - r3) * (-1/(-r1 - r3))
                    = -(r2 + r3) / (r1 + r2)

    where

    W/F_CH4,0(X_CH4 = 0) = 0
    X_CO2(X_CH4 = 0) = 0

    Inputs:
    -------
      t: methane conversion
      y: y[0]: 'space time' W/F_CH4,0, y[1]: X_CO2

    Returns:
    --------
      np.array([-rCH4, rCH4/rCO2])

    Raises:
    -------

    """
    conversion = np.array([np.atleast_1d(t), y[1]])

    molarFlowRateOut = stoichiometryReformer(molarFlowRateIn, conversion)
    moleFraction = molarFlowRateOut / molarFlowRateOut.sum(axis=0)
    assert all(np.isclose(moleFraction.sum(axis=0), 1))
    partialPressureBar = pressureBar * moleFraction

    r1, r2, r3 = reactionRateModel(partialPressureBar, temperatureKelvin, params)
    return np.array([-1 / (-r1 - r3), -(r2 + r3) / (r1 + r2)])


def reactionQuotient(partialPressureBar, molarFlowRate):

    a = partialPressureBar / STST_PRESSURE_BAR  # ideal gas activities

    Q = np.zeros((3, molarFlowRate.shape[1]))
    Q[:] = np.nan

    Q[Reaction["SMR"].value] = (
        a[Compound["H2(ref)"].value] ** 3 * a[Compound["C1O1(g)"].value]
    ) / (a[Compound["C1H4(g)"].value] * a[Compound["H2O1(g)"].value])

    Q[Reaction["WGS"].value] = (
        a[Compound["H2(ref)"].value] * a[Compound["C1O2(g)"].value]
    ) / (a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value])

    Q[Reaction["DSR"].value] = (
        a[Compound["H2(ref)"].value] ** 4 * a[Compound["C1O2(g)"].value]
    ) / (a[Compound["C1H4(g)"].value] * a[Compound["H2O1(g)"].value] ** 2)

    return Q


def inEquilibrium(
    pressureBar: float,
    temperatureKelvin: float,
    molarFlowRate: np.array,
    params: ModelParameters,
    rtol=1e-02,
    atol=1e-03,
):

    moleFraction = molarFlowRate / molarFlowRate.sum(axis=0)
    assert all(np.isclose(moleFraction.sum(axis=0), 1))
    partialPressureBar = pressureBar * moleFraction

    Q = reactionQuotient(partialPressureBar, molarFlowRate)
    Keq = equilibriumConstModel(temperatureKelvin, params).T

    return np.isclose(Q, Keq, rtol=rtol, atol=atol).prod(axis=0)


def getEqMolarFlowOut(molarFlowRateIn, pressureBar, dbfile=DBFILE):

    # TODO support scalar and vector queries

    molarMass = np.zeros(len(Compound), dtype=np.float64)

    hydrogenToCarbonMolar = (
        2 * molarFlowRateIn[Compound["H2(ref)"].value]
        + 2 * molarFlowRateIn[Compound["H2O1(g)"].value]
        + 4 * molarFlowRateIn[Compound["C1H4(g)"].value]
    ) / sum(
        molarFlowRateIn[idx.value]
        for idx in Compound
        if idx.name in {"C1H4(g)", "C1O1(g)", "C1O2(g)"}
    )
    oxygenToCarbonMolar = (
        molarFlowRateIn[Compound["H2O1(g)"].value]
        + molarFlowRateIn[Compound["C1O1(g)"].value]
        + 2 * molarFlowRateIn[Compound["C1O2(g)"].value]
    ) / sum(
        molarFlowRateIn[idx.value]
        for idx in Compound
        if idx.name in {"C1H4(g)", "C1O1(g)", "C1O2(g)"}
    )

    query = f"""
    SELECT
        temperature_kelvin AS 'T(K)',
        compound AS 'COMPOUND',
        mole_fraction
    FROM
        [mole_fractions]
    WHERE
        temperature_kelvin BETWEEN 600 AND 1600
    AND
        pressure_bar = {pressureBar}
    AND
        hydrogen_to_carbon_molar = {hydrogenToCarbonMolar}
    AND
        oxygen_to_carbon_molar = {oxygenToCarbonMolar}
    GROUP BY temperature_kelvin, compound
    ORDER BY temperature_kelvin
    ;
    """

    # logger.debug(query)

    with sqlite3.connect(dbfile) as con:
        eqMoleFractions = pd.read_sql(query, con).pivot(
            index="T(K)", columns="COMPOUND", values="mole_fraction"
        )

    for c in Compound:
        if c.name not in eqMoleFractions.columns:
            eqMoleFractions.loc[:, c.name] = 0

    molarMass[Compound["C1H4(g)"].value] = 16
    molarMass[Compound["C1O1(g)"].value] = 28
    molarMass[Compound["C1O2(g)"].value] = 44
    molarMass[Compound["H2O1(g)"].value] = 18
    molarMass[Compound["H2(ref)"].value] = 2

    massFlowRate = (molarMass * molarFlowRateIn).sum()
    molarFlowRateOut = eqMoleFractions.mul(
        (
            massFlowRate
            / eqMoleFractions.dot(
                pd.Series(molarMass, index=[c.name for c in Compound])
            )
        ),
        axis=0,
    )
    for c in Compound:
        if c.name not in molarFlowRateOut.columns:
            molarFlowRateOut.loc[:, c.name] = molarFlowRateIn.loc[:, c.name]

    return molarFlowRateOut


def getEqConversion(molarFlowRateIn, temperatureKelvin, pressureBar):
    molarFlowRateOut = getEqMolarFlowOut(molarFlowRateIn, pressureBar)
    eqConversion = (
        1 - molarFlowRateOut["C1H4(g)"] / molarFlowRateIn[Compound["C1H4(g)"].value]
    )
    return eqConversion.loc[temperatureKelvin]


def get_equilibriumConversion(
    molarFlowRateIn,
    temperatureKelvin,
    pressureBar,
    params,
    Compound=Compound,
    Reaction=Reaction,
    maxiter=100,
):

    equilibriumConst = equilibriumConstModel(temperatureKelvin, params)[0]

    def fun(conversion, molarFlowRateIn, temperatureKelvin, pressureBar):
        # print(conversion)
        molarFlowRateOut = stoichiometryReformer(
            molarFlowRateIn, np.array([conversion]).T
        )
        moleFraction = molarFlowRateOut / molarFlowRateOut.sum(axis=0)
        assert all(np.isclose(moleFraction.sum(axis=0), 1))
        activity = pressureBar / STST_PRESSURE_BAR * moleFraction
        residual = np.array(
            [
                activity[Compound["C1H4(g)"].value]
                * activity[Compound["H2O1(g)"].value]
                - activity[Compound["H2(ref)"].value] ** 3
                * activity[Compound["C1O1(g)"].value]
                / equilibriumConst[Reaction["SMR"].value],
                activity[Compound["C1O1(g)"].value]
                * activity[Compound["H2O1(g)"].value]
                - activity[Compound["H2(ref)"].value]
                * activity[Compound["C1O2(g)"].value]
                / equilibriumConst[Reaction["WGS"].value],
            ]
        )
        # print(residual[:,0])
        return residual[:, 0]

    root = newton(
        fun,
        np.array([0, 0]),
        maxiter=maxiter,
        args=(molarFlowRateIn, temperatureKelvin, pressureBar),
    )

    return root
