"""phdtools.data.diffusion.py

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

Diffusivity models.

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

Slattery, J.C. and Bird, R.B. (1958) ‘Calculation of the diffusion coefficient of
    dilute gases and of the self‐diffusion coefficient of dense gases’, AIChE Journal,
    4(2), pp. 137–142. Available at: https://doi.org/10.1002/aic.690040205.

"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.linear_model import LinearRegression

from phdtools import DATA_DIR
from phdtools.data.constants import GAS_CONST_SI
from phdtools.data.thermochemical import Compound


def pressureDiffusivityProductModel(
    A: Compound,
    B: Compound,
    temperatureKelvin: npt.ArrayLike,
    method: str = "slattery-bird-1958",
    fname=DATA_DIR / "nist-webbook" / "critical-point.csv",
) -> np.ndarray:
    """
    Slattery, J.C. and Bird, R.B. (1958) ‘Calculation of the diffusion coefficient of dilute gases
      and of the self‐diffusion coefficient of dense gases’, AIChE Journal, 4(2), pp. 137–142.
      Available at: https://doi.org/10.1002/aic.690040205.

    Linstrom, P. (1997) “NIST Chemistry WebBook, NIST Standard Reference Database 69.” National Institute
        of Standards and Technology. Available at: https://doi.org/10.18434/T4D303.
    """
    T = np.atleast_1d(temperatureKelvin)

    if (A.name in {"H2O1(g)"}) or (B.name in {"H2O1(g)"}):
        a = 5.148e-4
        b = 2.334
    else:
        a = 3.882e-4
        b = 1.823

    data = pd.read_csv(fname, index_col=0)

    TcA = data["Tc(K)"][A.name]
    pcA = data["Pc(bar)"][A.name]
    MA = data["M(g/mol)"][A.name]

    TcB = data["Tc(K)"][B.name]
    pcB = data["Pc(bar)"][B.name]
    MB = data["M(g/mol)"][B.name]

    TcAB = (TcA * TcB) ** (1 / 2)
    pcAB = (pcA * pcB) ** (1 / 2)
    MAB = 2 * MA * MB / (MA + MB)

    pDABR = a * (T / TcAB) ** b

    val = pDABR * pcAB ** (2 / 3) * TcAB ** (5 / 6) * MAB ** (-1 / 2)  # atm * cm2 / s

    return val * 1.01325 * 1e5 * 1e-4  # return in Pa m2 / s


def oxygenDiffusionActivationEnergyModel(
    weightConcentration: npt.ArrayLike,
    method: str = "meck-2025",
) -> np.ndarray:
    """Diffusion activation energy: E in J/mol, see Mamlouk, Sousa and Scott (2011, eq. (25)).
    Note: 1 cal ^= 4.184 J (Klinedinst et al., 1974)

    Inputs:
        w (float): the phosphoric acid (H3PO4) weight concentration (0-1)
    """
    w = np.atleast_1d(weightConcentration)
    x = np.array([w**2, w**1, w**0])
    if method == "mamlouk-2008":
        a = (
            np.array([-0.011607142857, 1.9642142857, -75.376])
            * np.array([100**2, 100**1, 100**0])
            * 1e3
            * 4.184
        )
    elif method == "meck-2025":
        # First order A
        # a = -np.array([4.74422568e+05, -7.99710336e+05,  3.04584987e+05])
        # Second order A
        a = -np.array([4.74422958e05, -7.99711054e05, 3.04585316e05])
    else:
        raise ValueError(
            f'Unknown method {method}. Valid options: "meck-2025", or "mamlouk-2008"'
        )

    return np.dot(a, x)


def fitDiffusivityInPhosphoricAcid(
    method: str = "meck-2025",
    fname_fig3: Path = DATA_DIR / "klinedinst-1974" / "251107_fig3_Diffusivity.csv",
    fname_fig5: Path = DATA_DIR
    / "klinedinst-1974"
    / "251109_fig5_DiffusionActivationEnergy.csv",
) -> LinearRegression:
    """Fits a linear model to determine the coefficients for
        (1. the diffusion activation energy Ea(w) of oxygen in phosphoric acid (H3PO4), and)
        2. the pre-exponential factor A(w),
    where w denotes the fractional (0--1) mass concentration of H3PO4.
    The diffusivity of oxygen is modelled as

        D = A exp(-E/RT)

    .
    """

    fig3 = pd.read_csv(fname_fig3)
    fig5 = pd.read_csv(fname_fig5)

    w1 = fig3["w(wt.%)"].to_numpy() / 100  # phosphoric acid mass concentration [0-1]
    D = fig3["D(cm2/s)"].to_numpy() * 1e-4  # oxygen diffusivity [m–2/s]
    T = fig3["T(C)"].to_numpy() + 273.15  # temperatur [K]

    w2 = fig5["w(wt.%)"].to_numpy() / 100  # phosphoric acid mass concentration [0-1]
    E = fig5["E(kcal/mol)"].to_numpy() * 4184  # enthalpy of solution [J/mol]

    R = GAS_CONST_SI  # Gas constant [J mol-1 K-1]

    if method == "sousa-2010":
        # D = A * exp( -E/RT )
        # where
        # -E/R = a1 w**3+ a2 w**2 + a3 w + a4
        # A = exp( b1 w**3+ b2 w**2 + b3 w + b4 )
        # ---
        # --> ln D = a1 w**3/T + a2 w**2/T + a3 w/T + a4 1/T
        #               + b1 w**3+ b2 w**2 + b3 w + b4

        X_train = np.c_[
            w1**3 / T, w1**2 / T, w1**1 / T, w1**0 / T, w1**3, w1**2, w1**1, w1**0
        ]
        y_train = np.log(D)

        reg = LinearRegression()
        reg.fit(X_train, y_train)

    elif method == "mamlouk-2008":
        # D = A exp(-E/RT)
        # where
        # A = exp( a1 w + a2 )
        # ---
        # --> ln D + E/RT = a1 w + a2

        E = oxygenDiffusionActivationEnergyModel(100 * w1)

        X_train = np.c_[100 * w1**1, 100 * w1**0]
        y_train = D + E / (R * T)

        reg = LinearRegression()
        reg.fit(X_train, y_train)

    elif method == "meck-2025":
        # D = A * exp( -E/RT )
        # where
        # -E = a1 w**2 + a2 w + a3
        # A = exp( b1 w + b2 )
        # ---
        # ln D = a1/RT w**2 + a2/RT w + a3/RT \
        #           + b1 w**2 + b2 w + b3
        # -E = a1 w**3 + a2 w**2 + a3 w + a4

        X1 = np.c_[
            w1**2 / (R * T),
            w1**1 / (R * T),
            w1**0 / (R * T),
            w1**2,
            w1**1,
            w1**0,
        ]

        X2 = np.c_[
            w2**2, w2**1, w2**0, np.zeros(len(w2)), np.zeros(len(w2)), np.zeros(len(w2))
        ]

        X_train = np.r_[X1, X2]
        y_train = np.r_[np.log(D), -E]

        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, y_train)
    else:
        raise ValueError(
            f'Unknown method {method}. Valid options: "meck-2025", "mamlouk-2008", or "sousa-2010"'
        )

    return reg


def oxygenDiffusivityInH3PO4(
    temperatureKelvin: npt.ArrayLike,
    weightConcentration: npt.ArrayLike,
    method: str = "meck-2025",
) -> np.ndarray:
    """
    Inputs:
        temperatureKelvin (npt.ArrayLike): temperature in K
        weightConcentration (npt.ArrayLike): the phosphoric acid (H3PO4) mass concentration (0-1)
    """
    T = np.atleast_1d(temperatureKelvin)
    w = np.atleast_1d(weightConcentration)

    if method == "meck-2025":

        x2 = np.array([w**2, w**1, w**0])

        # First order
        # b = = np.array([-2.56140285e+01, 1.13707400e+01])
        # Second order
        b = np.array([-1.78158512e02, 2.98972616e02, -1.36278907e02])

        Ea = oxygenDiffusionActivationEnergyModel(w, method="meck-2025")

        logA = np.dot(b, x2)

        val = np.exp((-Ea / (GAS_CONST_SI * T[:, np.newaxis])) + logA)

    elif method == "mamlouk-2008":
        Ea = oxygenDiffusionActivationEnergyModel(w, method="meck-2025")
        A = 0.00000249927283 * np.exp(1.76593087 * (100 * w))
        val = A * np.exp(-Ea / (GAS_CONST_SI * T[:, np.newaxis]))
    elif method == "sousa-2010":
        val = np.exp(
            (1 / T[:, np.newaxis])
            * (-9.21e5 * w**3 + 2.47e6 * w**2 - 2.21e6 * w + 6.54e5)
            + (1.66e3 * w**3 - 4.46e3 * w**2 + 4.01e3 * w + 1.21e3)
        )
    else:
        raise ValueError(
            f'Unknown method {method}. Valid options: "meck-2025", "mamlouk-2008", or "sousa-2010"'
        )

    return val
