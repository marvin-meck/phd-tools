"""phdtools.models.mendes_2010.py

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

Models for the water-gas shift reaction kinetics by Mendes et al. (2010).

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de


References
----------

Mendes, D. et al. (2010) 'Determination of the Low-Temperature Water−Gas Shift React-
    ion Kinetics Using a Cu-Based Catalyst', Industrial & Engineering Chemistry Research,
    49(22), pp. 11269–11279. Available at: https://doi.org/10.1021/ie101137b.

"""

# from typing import assert_never
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from phdtools import DATA_DIR
from phdtools.data.constants import GAS_CONST_SI
from phdtools.data.thermochemical import (
    STST_PRESSURE_BAR,
    Compound,
    stoichiometricNumber,
    get_logEquilibriumConst,
    get_stdReactionEnthalpySI,
)

STST_PRESSURE_PASCAL = 1e5 * STST_PRESSURE_BAR


@dataclass
class ModelParameters:
    model: str
    rateConstantFactorSI: float
    activationEnergySI: float
    equilibriumConstRef: float
    enthalpyReactionSI: float
    adsorptionCoefFactor: np.ndarray
    enthalpyAdsorptionSI: np.ndarray
    powerLawReactionOrder: np.ndarray
    refTemperatureEquilibriumSI: float = 500
    refTemperatureAdsorptionSI: float = 503
    refTemperatureRateSI: float = 503

    @classmethod
    def init(
        cls,
        fname: Path = DATA_DIR / "mendes-2010" / "250729_table_1_mendes_et_al_2010.csv",
        frame=None,
        model: str = "LH1",
    ):

        stdReactionEnthalpySI = get_stdReactionEnthalpySI(Tmin=298.15, Tmax=1000)
        logEquilibriumConst = get_logEquilibriumConst(Tmin=298.15, Tmax=1000)

        refTemperatureEquilibriumSI = 500

        equilibriumConstRef = np.exp(
            logEquilibriumConst.loc[refTemperatureEquilibriumSI, "WGS"]
        )
        enthalpyReactionSI = stdReactionEnthalpySI.loc[
            refTemperatureEquilibriumSI, "WGS"
        ]

        if frame is None:
            frame = pd.read_csv(
                fname,
                usecols=[0, 1, 3, 5, 7, 9],
                skiprows=[0, 1, 16],
                header=None,
                index_col=0,
                names=["Parameter", "Moe", "Power law", "LH1", "LH2", "Redox"],
            )

        if model in {"LH1", "LH2", "Redox", "Moe", "Power law"}:

            rateConstantFactorSI = frame.loc["k0", model] / 3.6
            activationEnergySI = 1e3 * frame.loc["Ea", model]

            adsorptionCoefFactor = np.full(len(Compound), np.nan)
            enthalpyAdsorptionSI = np.full(len(Compound), np.nan)
            powerLawReactionOrder = np.full(len(Compound), np.nan)
        else:
            # unreachable
            raise AssertionError(f'Unreachable. model="{model}"')
            # raise assert_never(model)

        if model in {"LH1", "LH2"}:
            adsorptionCoefFactor[Compound["C1O1(g)"].value] = frame.loc["KCO,0", model]
            adsorptionCoefFactor[Compound["H2O1(g)"].value] = frame.loc["KH2O,0", model]
            adsorptionCoefFactor[Compound["C1O2(g)"].value] = frame.loc["KCO2,0", model]
            adsorptionCoefFactor[Compound["H2(ref)"].value] = frame.loc["KH2,0", model]

            enthalpyAdsorptionSI[Compound["C1O1(g)"].value] = (
                1e3 * frame.loc["dHCO", model]
            )
            enthalpyAdsorptionSI[Compound["H2O1(g)"].value] = (
                1e3 * frame.loc["dHH2O", model]
            )
            enthalpyAdsorptionSI[Compound["C1O2(g)"].value] = (
                1e3 * frame.loc["dHCO2", model]
            )
            enthalpyAdsorptionSI[Compound["H2(ref)"].value] = (
                1e3 * frame.loc["dHH2", model]
            )
        elif model == "Redox":
            adsorptionCoefFactor[Compound["C1O2(g)"].value] = frame.loc["KCO2,0", model]
            enthalpyAdsorptionSI[Compound["C1O2(g)"].value] = (
                1e3 * frame.loc["dHCO2", model]
            )
        elif model == "Power law":
            # according to tables 1 and 2, this is not consistent with eq. (18)
            powerLawReactionOrder[Compound["C1O1(g)"].value] = frame.loc["a", model]
            powerLawReactionOrder[Compound["H2O1(g)"].value] = frame.loc["b", model]
            powerLawReactionOrder[Compound["H2(ref)"].value] = frame.loc["d", model]
            powerLawReactionOrder[Compound["C1O2(g)"].value] = frame.loc["c", model]
        else:
            pass

        if model == "LH1":
            rateConstantFactorSI *= np.power(STST_PRESSURE_PASCAL, 2)
            adsorptionCoefFactor *= STST_PRESSURE_PASCAL
        elif model == "LH2":
            rateConstantFactorSI *= np.power(STST_PRESSURE_PASCAL, 2)
            adsorptionCoefFactor[Compound["C1O1(g)"].value] *= STST_PRESSURE_PASCAL
            adsorptionCoefFactor[Compound["H2O1(g)"].value] *= STST_PRESSURE_PASCAL
            adsorptionCoefFactor[Compound["C1O2(g)"].value] *= STST_PRESSURE_PASCAL ** (
                1.5
            )
            adsorptionCoefFactor[Compound["H2(ref)"].value] *= STST_PRESSURE_PASCAL ** (
                0.5
            )
        elif model == "Redox":
            rateConstantFactorSI *= STST_PRESSURE_PASCAL
        elif model == "Moe":
            rateConstantFactorSI *= STST_PRESSURE_PASCAL**2
        elif model == "Power law":
            rateConstantFactorSI *= STST_PRESSURE_PASCAL ** (
                powerLawReactionOrder[~np.isnan(powerLawReactionOrder)].sum()
            )

        return cls(
            model=model,
            rateConstantFactorSI=rateConstantFactorSI,
            activationEnergySI=activationEnergySI,
            equilibriumConstRef=equilibriumConstRef,
            enthalpyReactionSI=enthalpyReactionSI,
            adsorptionCoefFactor=adsorptionCoefFactor,
            enthalpyAdsorptionSI=enthalpyAdsorptionSI,
            powerLawReactionOrder=powerLawReactionOrder,
            refTemperatureEquilibriumSI=refTemperatureEquilibriumSI,
        )


def equilibriumConstModel(temperatureKelvin, params: ModelParameters):
    """van't Hoff equation for the equilibrium constant K"""
    T = np.atleast_1d(temperatureKelvin)
    return params.equilibriumConstRef * np.exp(
        -params.enthalpyReactionSI
        / GAS_CONST_SI
        * ((1 / T)[:, np.newaxis] - 1 / params.refTemperatureEquilibriumSI)
    )


def rateConstModel(temperatureKelvin, params: ModelParameters):
    """Arrhenius equation for the rate constant k0"""
    T = np.atleast_1d(temperatureKelvin)

    return params.rateConstantFactorSI * np.exp(
        -params.activationEnergySI / (GAS_CONST_SI * T[:, np.newaxis])
    )


def adsorptionCoefModel(temperatureKelvin, params: ModelParameters):
    """van't Hoff equation for the adsorption coefficients Kj, j = CO,H2,H2O"""
    T = np.atleast_1d(temperatureKelvin)
    return params.adsorptionCoefFactor * np.exp(
        -params.enthalpyAdsorptionSI / (GAS_CONST_SI * T[:, np.newaxis])
    )


def reactionRateModel(partialPressureBar, temperatureKelvin, params: ModelParameters):
    """ToDo: write docstring"""
    a = partialPressureBar / STST_PRESSURE_BAR  # ideal gas activities

    k = rateConstModel(temperatureKelvin, params=params)[0]
    Keq = equilibriumConstModel(temperatureKelvin, params=params)[0]

    if params.model == "LH1":

        Kad = adsorptionCoefModel(temperatureKelvin, params=params)[0]

        DEN = (
            1
            + Kad[Compound["C1O1(g)"].value] * a[Compound["C1O1(g)"].value]
            + Kad[Compound["H2O1(g)"].value] * a[Compound["H2O1(g)"].value]
            + Kad[Compound["C1O2(g)"].value] * a[Compound["C1O2(g)"].value]
            + Kad[Compound["H2(ref)"].value] * a[Compound["H2(ref)"].value]
        ) ** 2

        r = (
            k
            * (
                a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value]
                - a[Compound["C1O2(g)"].value] * a[Compound["H2(ref)"].value] / Keq
            )
            / DEN
        )

    elif params.model == "LH2":

        Kad = adsorptionCoefModel(temperatureKelvin, params=params)[0]

        DEN = (
            1
            + Kad[Compound["C1O1(g)"].value] * a[Compound["C1O1(g)"].value]
            + Kad[Compound["H2O1(g)"].value] * a[Compound["H2O1(g)"].value]
            + Kad[Compound["C1O2(g)"].value]
            * a[Compound["C1O2(g)"].value]
            * a[Compound["H2(ref)"].value] ** (0.5)
            + Kad[Compound["H2(ref)"].value] ** (0.5)
            * a[Compound["H2(ref)"].value] ** (0.5)
        ) ** 2

        r = (
            k
            * (
                a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value]
                - a[Compound["C1O2(g)"].value] * a[Compound["H2(ref)"].value] / Keq
            )
            / DEN
        )

    elif params.model == "Redox":

        Kad = adsorptionCoefModel(temperatureKelvin, params=params)[0]

        beta = (
            a[Compound["H2(ref)"].value]
            * a[Compound["C1O2(g)"].value]
            / (a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value])
            * (1 / Keq)
        )

        r = (
            k
            * a[Compound["H2O1(g)"].value]
            * (1 - beta)
            / (
                1
                + Kad[Compound["C1O2(g)"].value]
                * a[Compound["C1O2(g)"].value]
                / a[Compound["C1O1(g)"].value]
            )
        )
    elif params.model == "Moe":
        beta = (
            a[Compound["H2(ref)"].value]
            * a[Compound["C1O2(g)"].value]
            / (a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value])
            * (1 / Keq)
        )
        r = k * a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value] * (1 - beta)
    elif params.model == "Power law":
        beta = (
            a[Compound["H2(ref)"].value]
            * a[Compound["C1O2(g)"].value]
            / (a[Compound["C1O1(g)"].value] * a[Compound["H2O1(g)"].value])
            * (1 / Keq)
        )
        r = (
            k
            * a[Compound["C1O1(g)"].value]
            ** params.powerLawReactionOrder[Compound["C1O1(g)"].value]
            * a[Compound["H2O1(g)"].value]
            ** params.powerLawReactionOrder[Compound["H2O1(g)"].value]
            * a[Compound["H2(ref)"].value]
            ** params.powerLawReactionOrder[Compound["H2(ref)"].value]
            * a[Compound["C1O2(g)"].value]
            ** params.powerLawReactionOrder[Compound["C1O2(g)"].value]
            * (1 - beta)
        )
    else:
        pass

    return -r


def stoichiometryShift(moleFractionIn, conversion):
    """ToDo: write docstring"""
    conversion = np.atleast_2d(conversion)
    moleFractionOut = np.zeros((moleFractionIn.shape[0], conversion.shape[1]))

    moleFractionOut[Compound["C1O1(g)"].value] = moleFractionIn[
        Compound["C1O1(g)"].value
    ] * (1 - conversion)

    for c in Compound:
        if c.name != "C1O1(g)":
            moleFractionOut[c.value] = moleFractionIn[
                c.value
            ] + -1 * stoichiometricNumber["WGS"][c.name] * (
                moleFractionOut[Compound["C1O1(g)"].value]
                - moleFractionIn[Compound["C1O1(g)"].value]
            )

    return moleFractionOut


def initialValueProblemConversion(
    t, y, moleFractionIn, temperatureKelvin, pressureBar, params
):
    """IVP: d(W/FCO,0)/d(XCO) = -1/rCO, W/FCO(XCO = 0) = 0"""

    conversion = t
    moleFractionOut = stoichiometryShift(moleFractionIn, conversion)
    # assert all(np.isclose(moleFractionOut.sum(axis=0),1))
    partialPressureBar = pressureBar * moleFractionOut

    r = reactionRateModel(partialPressureBar, temperatureKelvin, params=params)
    return np.array([-1 / r])


def initialValueProblemSpaceTime(
    t, y, moleFractionIn, temperatureKelvin, pressureBar, params
):
    """IVP: d(XCO) / d(W/FCO,0) = -rCO, XCO(W/FCO,0 = 0) = 0"""

    spaceTime = t
    conversion = y
    moleFractionOut = stoichiometryShift(moleFractionIn, conversion)
    assert np.isclose(moleFractionOut.sum(axis=0), 1)
    partialPressureBar = pressureBar * moleFractionOut

    r = reactionRateModel(partialPressureBar, temperatureKelvin, params=params)
    return np.array([-r])
