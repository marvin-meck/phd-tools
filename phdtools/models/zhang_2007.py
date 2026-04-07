"""phdtools.models.zhang_2007.py

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

Fuel cell model by Zhang et al., 2007.

References
----------

Zhang, Jianlu et al. (2007) “Polybenzimidazole-membrane-based PEM fuel cell in the
    temperature range of 120–200 °C,” Journal of Power Sources, 172(1), pp. 163–171.
    Available at: https://doi.org/10.1016/j.jpowsour.2007.07.047.

"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from phdtools import PROJECT_ROOT
from phdtools.data.constants import GAS_CONST_SI

MEMBRANE_THICKNESS_SI = 511e-6  # estimated to match the range of area specific resitance reported, Rs = 0.08..0.11 Ohm cm2
ELECTRONS_TRANSFERRED_HOR = 2
ELECTRONS_TRANSFERRED_ORR = 1


def fit_exchangeCurrentDensityModel(
    fname: Path = PROJECT_ROOT / "phd-data" / "zhang-2007" / "2511102_table1.csv",
):

    table1 = pd.read_csv(fname, usecols=[0, 1, 2, 3, 4])
    table1["T(K)"] = table1["Temperature (C)"] + 273.15
    table1 = table1.drop("Temperature (C)", axis=1).set_index("T(K)")

    model = pd.DataFrame(columns=["Tr (K)", "j0[Tr] (A/cm2)", "E (kJ/mol)", "R2"])

    T = table1.index.to_numpy()
    Tr = (table1.index.min() + table1.index.max()) / 2

    for name in table1.columns:

        j0 = table1[name] * 1e4

        X = np.c_[-1 / (GAS_CONST_SI * T)]
        Y = np.log(j0.to_numpy())

        reg = LinearRegression().fit(X, Y)

        logExchangeCurrentDensitySIRef = reg.predict(
            np.c_[np.array([-1 / (GAS_CONST_SI * Tr)])]
        )

        model.loc[name, "Tr (K)"] = Tr
        model.loc[name, "j0[Tr] (A/cm2)"] = (
            np.exp(logExchangeCurrentDensitySIRef) * 1e-4
        )
        model.loc[name, "E (kJ/mol)"] = reg.coef_ * 1e-3
        model.loc[name, "R2"] = reg.score(X, Y)

    return model


@dataclass
class ModelParameters:
    exchangeCurrentDensitySIRef: np.array
    activationEnergyExchangeCurrentDensitySI: np.array
    refTemperatureExchangeCurrentDensitySI: np.array
    preExponentialFactorConductivitySI: float
    activationEnergyConductivitySI: float

    @classmethod
    def init(
        cls,
        fname_table1=PROJECT_ROOT / "phd-data" / "zhang-2007" / "2511102_table1.csv",
        fname_table2=PROJECT_ROOT / "phd-data" / "zhang-2007" / "2511102_table2.csv",
    ):

        model_parameters = fit_exchangeCurrentDensityModel(fname_table1)

        table2 = pd.read_csv(
            fname_table2,
            skiprows=9,
            usecols=[1, 2],
            names=["activationEnergy", "logPreExponentialFactor"],
            delimiter=",",
            comment="#",
        )

        return cls(
            exchangeCurrentDensitySIRef=np.array(
                [
                    model_parameters.loc["i^0_H2, apparent (A cm-2)", "j0[Tr] (A/cm2)"]
                    * 1e4,
                    model_parameters.loc["i^0_O2, apparent (A cm-2)", "j0[Tr] (A/cm2)"]
                    * 1e4,
                ]
            ),
            activationEnergyExchangeCurrentDensitySI=np.array(
                [
                    model_parameters.loc["i^0_H2, apparent (A cm-2)", "E (kJ/mol)"]
                    * 1e3,
                    model_parameters.loc["i^0_O2, apparent (A cm-2)", "E (kJ/mol)"]
                    * 1e3,
                ]
            ),
            refTemperatureExchangeCurrentDensitySI=np.array(
                [
                    model_parameters.loc["i^0_H2, apparent (A cm-2)", "Tr (K)"],
                    model_parameters.loc["i^0_O2, apparent (A cm-2)", "Tr (K)"],
                ]
            ),
            preExponentialFactorConductivitySI=np.exp(table2.iloc[0, 1])
            * 1e2,  # K / (Ohm m)
            activationEnergyConductivitySI=table2.iloc[0, 0] * 1e3,  # J / mol
        )


def transferCoefModel(temperatureKelvin):
    T = np.atleast_1d(temperatureKelvin)
    return np.c_[0.5 * np.ones(len(T)), 0.001678 * T]


def exchangeCurrentDensityModel(params: ModelParameters, temperatureKelvin):
    T = np.atleast_1d(temperatureKelvin)

    return params.exchangeCurrentDensitySIRef * np.exp(
        -params.activationEnergyExchangeCurrentDensitySI
        / GAS_CONST_SI
        * ((1 / T)[:, np.newaxis] - 1 / params.refTemperatureExchangeCurrentDensitySI)
    )


def conductivityModel(params: ModelParameters, temperatureKelvin: float):
    """ """
    T = np.atleast_1d(temperatureKelvin)

    return (
        params.preExponentialFactorConductivitySI
        / T
        * np.exp(-params.activationEnergyConductivitySI / (GAS_CONST_SI * T))
    )
