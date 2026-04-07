"""phdtools.models.ohayre_2016.py

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

Fuel cell model by R.P. O'Hayre (2016).

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de


References
----------

O'Hayre, R.P. (2016) Fuel cell fundamentals. Third edition. Hoboken, New Jersey:
    John Wiley & Sons. Available at: https://doi.org/10.1002/9781119191766.

"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import root

from phdtools.data.constants import FARADAY_CONST_SI, GAS_CONST_SI
from phdtools.data.thermochemical import (
    STST_PRESSURE_BAR,
    Compound,
    get_stdReactionGibbsEnergySI,
)

ELECTRONS_TRANSFERRED = 4
ELECTRODE_POROSITY = 0.4

# Compound = Enum("Compound", ["H2(ref)", "H2O1(g)", "O2(ref)", "N2(ref)"], start=0)


@dataclass
class ModelParameters:
    areaSpecificResistanceSI: (
        float  # ASR = A * R exponential function of temperatureKelvin
    )
    limitingCurrentDensitySI: float  #
    transferCoefficientAnode: float
    exchangeCurrentDensityAnodeSI: float  # -- j0 function of temperatureKelvin
    transferCoefficientCathode: float
    exchangeCurrentDensityCathodeSI: float  # -- j0 function of temperatureKelvin
    leakageCurrentDensitySI: float
    c: float = np.nan


def stdCellPotentialModel(
    temperatureKelvin: npt.ArrayLike, method: Literal["interpolate"] = "interpolate"
) -> np.ndarray:
    T = np.atleast_1d(temperatureKelvin)

    if method == "interpolate":
        stdReactionGibbsEnergySI = get_stdReactionGibbsEnergySI(298.15, 1000, {"HCR1"})

        stdCellPotentialSI = (
            -stdReactionGibbsEnergySI["HCR1"] / (4 * FARADAY_CONST_SI)
        ).rename("E0")

        E0 = np.interp(
            x=T,
            xp=stdCellPotentialSI.index.to_numpy(),
            fp=stdCellPotentialSI.to_numpy(),
        )
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    return E0


def reversibleCellPotentialModel(
    temperatureKelvin: npt.ArrayLike,
    activity: npt.ArrayLike,
    method: Literal["interpolate"] = "interpolate",
) -> np.ndarray:
    T = np.atleast_1d(temperatureKelvin)

    if method == "interpolate":
        E0 = stdCellPotentialModel(T, method=method)
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    return E0 - GAS_CONST_SI * T / (ELECTRONS_TRANSFERRED * FARADAY_CONST_SI) * np.log(
        activity[Compound["H2O1(g)"].value] ** 2
        / (
            activity[Compound["H2(ref)"].value] ** 2
            * activity[Compound["O2(ref)"].value]
        )
    )


def activationOverpotentialModel(
    currentDensitySI: npt.ArrayLike,
    temperatureKelvin: float,
    transferCoefficient: float,
    exchangeCurrentDensitySI: float,
    electronsTransferred: int,
    model: Literal["tafel", "butler-volmer"] = "tafel",
) -> np.ndarray:
    """Computes activation overpotential as a function of current density and temperature.

    Parameters
    ----------
    currentDensitySI : float
        current density in Ampere per meters squared
    temperatureKelvin : float
        temperature in Kelvin
    transferCoefficient : float
        transfer coefficient (dimensionless)
    exchangeCurrentDensitySI : float
        exchange current density in Ampere per meters squared

    Returns
    -------
    eta_act
        activation overpotential in Volt

    Raises
    ------

    """
    j = np.atleast_1d(currentDensitySI)
    j0 = exchangeCurrentDensitySI
    alpha = transferCoefficient

    eta_act = np.zeros(len(j))
    eta_act[:] = np.nan

    if model == "tafel":

        mask = j < 0
        if any(mask):
            eta_act[mask] = (
                -GAS_CONST_SI
                * temperatureKelvin
                / (alpha * electronsTransferred * FARADAY_CONST_SI)
                * np.log(-j[mask] / j0)
            )

        mask = j > 0
        if any(mask):
            eta_act[mask] = (
                GAS_CONST_SI
                * temperatureKelvin
                / ((1 - alpha) * electronsTransferred * FARADAY_CONST_SI)
                * np.log(j[mask] / j0)
            )

    elif model == "butler-volmer":
        x0 = activationOverpotentialModel(
            currentDensitySI,
            temperatureKelvin,
            transferCoefficient,
            exchangeCurrentDensitySI,
            electronsTransferred,
            model="tafel",
        )
        mask = j < 0
        if any(mask):
            x0[mask & (x0 >= 0)] = 0

        mask = j > 0
        if any(mask):
            x0[mask & (x0 <= 0)] = 0

        obj = (
            lambda eta: j0
            * (
                np.exp(
                    (1 - alpha)
                    * electronsTransferred
                    * FARADAY_CONST_SI
                    * eta
                    / (GAS_CONST_SI * temperatureKelvin)
                )
                - np.exp(
                    -alpha
                    * electronsTransferred
                    * FARADAY_CONST_SI
                    * eta
                    / (GAS_CONST_SI * temperatureKelvin)
                )
            )
            - j
        )
        sol = root(obj, x0=x0)
        eta_act = sol.x
    else:
        raise NotImplementedError(f"Unknown value for option model={model}")

    return eta_act


def concentrationOverpotentialModel(
    currentDensitySI: npt.ArrayLike,
    temperatureKelvin: npt.ArrayLike,
    limitingCurrentDensitySI: npt.ArrayLike,
    transferCoefficient: npt.ArrayLike,
    electronsTransferred: float = ELECTRONS_TRANSFERRED,
) -> np.ndarray:
    """Computes concentration losses as a function of current density and temeperature
    see Fuel Cell Fundamentals p. 148 Eq. (5.25) and (5.26)

    Parameters
    ----------
    currentDensitySI : float
        current density in Ampere per meters squared
    temperatureKelvin : float
        temperatureKelvin in Kelvin

    Returns
    -------
    concentration_losses
        concentration losses in Volt

    Raises
    ------

    """
    j = np.atleast_1d(currentDensitySI)
    T = np.atleast_1d(temperatureKelvin)
    jL = np.atleast_1d(limitingCurrentDensitySI)
    a = np.atleast_1d(transferCoefficient)

    const = (
        GAS_CONST_SI
        * T
        / (FARADAY_CONST_SI)
        * (1 / ELECTRONS_TRANSFERRED + 1 / electronsTransferred * 1 / a)
    )

    return const[:, np.newaxis] * np.log(1 - j / jL[:, np.newaxis])


def fuelCellVoltageModel(
    params: ModelParameters,
    currentDensitySI: float,
    partialPressureAnodeBar,
    partialPressureCathodeBar,
    temperatureKelvin: float,
) -> float:
    """Computes fuel cell voltage as a function of current density and temeperature,
    see Fuel Cell Fundamentals p.

    Parameters
    ----------
    params : ModelParameters
        fuel cell data class
    currentDensitySI : float
        current density in Ampere per meters squared
    temperatureKelvin : float
        temperatureKelvin in Kelvin

    Returns
    -------
    concentration_losses
        concentration losses in Volt

    Raises
    ------
    """
    j = np.atleast_1d(currentDensitySI)
    jgross = j  # + params.leakageCurrentDensitySI

    voltage = (
        reversibleCellPotentialModel(
            temperatureKelvin,
            partialPressureAnodeBar,
            partialPressureCathodeBar,
            method="interpolate",
        )
        + activationOverpotentialModel(
            currentDensitySI=-jgross,
            temperatureKelvin=temperatureKelvin,
            transferCoefficient=params.transferCoefficientCathode,
            exchangeCurrentDensitySI=params.exchangeCurrentDensityCathodeSI,
            model="butler-volmer",
        )
        # - ohmicLossesModel(params, j)
        # - concentrationOverpotentialModel(
        #     currentDensitySI: npt.ArrayLike,
        #     temperatureKelvin: npt.ArrayLike,
        #     limitingCurrentDensitySI: npt.ArrayLike,
        #     transferCoefficient: npt.ArrayLike,
        #     electronsTransferred: float=ELECTRONS_TRANSFERRED
        # )
    )

    return voltage
