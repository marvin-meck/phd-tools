"""phdtools.data.thermophysical.py

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

import json

import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression

from phdtools import DATA_DIR
from phdtools.data import Compound


def vapourPressureModel(
    temperatureKelvin, fname=DATA_DIR / "iapws-1995" / "water-vapour-pressure.json"
):
    """IAPWS-95 vapor–pressure equation (Wagner and Pruß, 2002, eq. (2.5))

    References
    ---------
    Wagner, W. and Pruß, A. (2002) ‘The IAPWS Formulation 1995 for the
      Thermodynamic Properties of Ordinary Water Substance for General and
      Scientific Use’, Journal of Physical and Chemical Reference Data,
      31(2), pp. 387–535. Available at: https://doi.org/10.1063/1.1461829.
    """
    T = np.atleast_1d(temperatureKelvin)

    with open(fname) as f:
        iapws95 = json.load(f)

    Tc = iapws95["Tc(K)"]
    pc = iapws95["pc(MPa)"] * 1e6
    a = iapws95["a[i]"]

    t = 1 - T / Tc

    return pc * np.exp(
        Tc
        / T
        * (
            a[0] * t
            + a[1] * t**1.5
            + a[2] * t**3
            + a[3] * t ** (3.5)
            + a[4] * t**4
            + a[5] * t ** (7.5)
        )
    )


def saturatedLiquidWaterDensityModel(
    temperatureKelvin, fname=DATA_DIR / "iapws-1995" / "water-vapour-pressure.json"
):
    """IAPWS-95 vapor–pressure equation (Wagner and Pruß, 2002, eq. (2.6))

    References
    ---------
    Wagner, W. and Pruß, A. (2002) ‘The IAPWS Formulation 1995 for the
      Thermodynamic Properties of Ordinary Water Substance for General and
      Scientific Use’, Journal of Physical and Chemical Reference Data,
      31(2), pp. 387–535. Available at: https://doi.org/10.1063/1.1461829.
    """
    T = np.atleast_1d(temperatureKelvin)

    with open(fname) as f:
        iapws95 = json.load(f)

    Tc = iapws95["Tc(K)"]
    rhoc = iapws95["rhoc(kg m-3)"]
    b = iapws95["b[i]"]

    t = 1 - T / Tc

    return rhoc * (
        1
        + b[0] * t ** (1 / 3)
        + b[1] * t ** (2 / 3)
        + b[2] * t ** (5 / 3)
        + b[3] * t ** (16 / 3)
        + b[4] * t ** (43 / 3)
        + b[5] * t ** (110 / 3)
    )


def moleFractionFromDry(dryMoleFraction, temperatureKelvin, pressureSI, relHumidity):

    vapourPressureSI = vapourPressureModel(temperatureKelvin)[0]

    moleFraction = np.zeros(len(Compound))

    moleFraction[Compound["H2O1(g)"].value] = (
        relHumidity * vapourPressureSI / pressureSI
    )
    for c in Compound:
        if not c.name == "H2O1(g)":
            moleFraction[c.value] = dryMoleFraction[c.value] * (
                1 - moleFraction[Compound["H2O1(g)"].value]
            )

    return moleFraction


def get_moleFractionH3PO4(
    weightConcentration: npt.ArrayLike,
) -> np.ndarray:
    """Computes the mole fraction of H3PO4 in solution from the concentration
    of phosphoric acid (H3PO4) in weight percent (mass fraction), see
    MacDonald and Boyack (1969, p. 383).

    References
    ----------
    MacDonald, D.I. and Boyack, J.R. (1969) “Density, electrical conductivity, and vapor
        pressure of concentrated phosphoric acid,” Journal of Chemical & Engineering Data,
        14(3), pp. 380–384. Available at: https://doi.org/10.1021/je60042a013.

    Inputs:
        weightConcentration (npt.ArrayLike): the phosphoric acid (H3PO4) mass concentration (0-1)

    """
    w = np.atleast_1d(weightConcentration)
    return w / (w + 5.43955 * (1 - w))


def fit_waterVapourPressureOverH3PO4Model(
    fname=DATA_DIR
    / "macdonald-1969"
    / "251111_tab4_VaporPressureOverConcentratedH3PO4.csv",
):

    tab4 = pd.read_csv(fname, skiprows=[1])

    temps = [129.65, 139.72, 149.88, 169.75]
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    blocks = []
    for T, (wcol, pcol) in zip(temps, pairs):
        b = tab4.iloc[:, [wcol, pcol]].dropna().copy()
        b.columns = ["W(wt. %)", "p(mmHg)"]
        b.insert(0, "T(C)", T)
        blocks.append(b)

    df = pd.concat(blocks, ignore_index=True)

    mask = (df["W(wt. %)"] >= 80) & (df["W(wt. %)"] <= 101)
    df = df[mask]

    T = df["T(C)"].to_numpy() + 273.15
    p = df["p(mmHg)"].to_numpy()
    W = df["W(wt. %)"].to_numpy()
    X = get_moleFractionH3PO4(W / 100) * 100

    X_train = np.array(
        [X**0, X**1, X**2, X**3, (X**0) / T, (X**1) / T, (X**2) / T, (X**3) / T]
    ).T

    y_train = np.log(p)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train)

    return reg


def waterVapourPressureOverH3PO4Model(
    moleFractionH3PO4: npt.ArrayLike,
    temperatureKelvin: npt.ArrayLike,
) -> np.ndarray:
    """see MacDonald and Boyack (1969, eq. 7).
    Coefficients were re-estimated using data from table IV and following
    the description for correlating the data from table V.

    References
    ----------
    MacDonald, D.I. and Boyack, J.R. (1969) “Density, electrical conductivity, and vapor
        pressure of concentrated phosphoric acid,” Journal of Chemical & Engineering Data,
        14(3), pp. 380–384. Available at: https://doi.org/10.1021/je60042a013.
    """

    x = np.atleast_1d(moleFractionH3PO4)
    T = np.atleast_1d(temperatureKelvin)

    X = np.array([(100 * x) ** 0, (100 * x) ** 1, (100 * x) ** 2, (100 * x) ** 3])

    a = np.array([1.13114730e01, 3.94874493e-01, -6.41369054e-03, 3.36245524e-05])
    b = np.array([-8.68091692e02, -1.98100881e02, 2.97646372e00, -1.62339285e-02])

    return 133.322 * np.exp(np.dot(a, X) + np.dot(b, X) * (T ** (-1))[:, np.newaxis])


def phosphoricAcidDensityModel(
    massFraction: npt.ArrayLike,
    temperatureKelvin: npt.ArrayLike,
    fname=DATA_DIR / "macdonald-1969" / "coefficients.json",
):
    """see MacDonald and Boyack (1969, eq. 1)

    References
    ----------
    MacDonald, D.I. and Boyack, J.R. (1969) “Density, electrical conductivity, and vapor
        pressure of concentrated phosphoric acid,” Journal of Chemical & Engineering Data,
        14(3), pp. 380–384. Available at: https://doi.org/10.1021/je60042a013.
    """
    w = np.atleast_1d(massFraction)
    T = np.atleast_1d(temperatureKelvin)

    W = w * 100
    t = T - 273.15

    with open(fname) as f:
        data = json.load(f)

    a1 = np.array(data["Density A[i]"])[0:2]
    a2 = np.array(data["Density A[i]"])[2:4]

    x = np.array([W**0, W**1])

    return (a1 @ x + (a2 @ x) * t[:, np.newaxis]) * 1e3


def get_molarConcentrationPhosphoricAcid(
    massFractionPhosphoricAcid: npt.ArrayLike,
    temperatureKelvin: npt.ArrayLike,
    fname=DATA_DIR / "macdonald-1969" / "coefficients.json",
):
    """
    References
    ----------
    MacDonald, D.I. and Boyack, J.R. (1969) “Density, electrical conductivity, and vapor
        pressure of concentrated phosphoric acid,” Journal of Chemical & Engineering Data,
        14(3), pp. 380–384. Available at: https://doi.org/10.1021/je60042a013.
    """
    with open(fname) as f:
        _tmp = json.load(f)

    molarMassWaterSI = _tmp["M_H2O(g/mol)"] * 1e-3
    molarMassPhosphoricAcidSI = _tmp["M_H3PO4(g/mol)"] * 1e-3
    massDensityTotalSI = phosphoricAcidDensityModel(
        massFractionPhosphoricAcid, temperatureKelvin
    )

    return (
        massDensityTotalSI * (1 - massFractionPhosphoricAcid) / molarMassWaterSI
        + massDensityTotalSI * massFractionPhosphoricAcid / molarMassPhosphoricAcidSI
    )
