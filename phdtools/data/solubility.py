"""phdtools.data.solubility.py

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

Solubility models.

References
----------
Klinedinst, K. et al. (1974) “Oxygen solubility and diffusivity in hot concentrated
    H3PO4,” Journal of Electroanalytical Chemistry and Interfacial Electrochemistry,
    57(3), pp. 281–289. Available at: https://doi.org/10.1016/S0022-0728(74)80053-7.

Linstrom, P. (1997) “NIST Chemistry WebBook, NIST Standard Reference Database 69.”
    National Institute of Standards and Technology.
    Available at: https://doi.org/10.18434/T4D303.

Mamlouk, M., Sousa, T. and Scott, K. (2011) “A High Temperature Polymer Electrolyte
    Membrane Fuel Cell Model for Reformate Gas,” International Journal of Electro-
    chemistry, 2011, pp. 1–18. Available at: https://doi.org/10.4061/2011/520473.

"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from phdtools import DATA_DIR
from phdtools.data.constants import GAS_CONST_SI
from phdtools.data.thermochemical import Compound

STD_CONCENTRATION_SI = 1e3  # https://goldbook.iupac.org/terms/view/S05909
STD_MOLALITY_SI = 1  # https://goldbook.iupac.org/terms/view/S05918


def molalSolubilityInWater(
    c: Compound,
    temperatureKelvin: float,
    fname=DATA_DIR / "nist-webbook" / "henrys-law-data.csv",
):
    """Model of the molal Henry's law constant in mol/(kg*Pa) for solubility
    in water at temperature T, see (Linstrom, 1997):

        kH(T) = kH(Tr) exp(d(ln(kH))/d(1/T) ((1/T) - 1/(Tr)))

    References:
    -----------
    Linstrom, P. (1997) “NIST Chemistry WebBook, NIST Standard Reference Database 69.”
        National Institute of Standards and Technology. Available at: https://doi.org/10.18434/T4D303.
    """
    params = pd.read_csv(fname, index_col=0)

    T = np.atleast_1d(temperatureKelvin)

    return (
        params.loc[c.name, "kH[Tr](mol/(kg*bar))"]
        * 1e-5
        * np.exp(
            params.loc[c.name, "d[ln(kH)]/d[1/T](K)"]
            * ((1 / T) - 1 / (params.loc[c.name, "Tr(K)"]))
        )
    )


def fitMolarSolubilityPhosphoricAcid(
    method="meck-2025",
    fname_fig4=DATA_DIR / "klinedinst-1974" / "251108_fig4_OxygenSolubility.csv",
    fname_fig6=DATA_DIR / "klinedinst-1974" / "251109_fig6_SolutionEnthalpy.csv",
):
    """Fits a linear model to determine the coefficients for
        (1. the enthalpy of solution -dH(w) of oxygen in phosphoric acid (H3PO4), and)
        2. the pre-exponential factor B(w),
    where w denotes the fractional (0--1) mass concentration of H3PO4.
    The solubility of oxygen is modelled as

        H = c / p = B exp(-dH/RT)

    where c is the molar concentration of oxygen dissolved in H3PO4, p is the oxygen
    partial pressure and H is Henry's law constant.
    """

    fig4 = pd.read_csv(fname_fig4)
    # fig4["cO2(mol/cm3)"] = 1e3*fig4["cO2(mol/cm3)"]
    fig6 = pd.read_csv(fname_fig6)

    w1 = fig4["w(wt.%)"].to_numpy() / 100  # phosphoric acid mass concentration [0-1]
    c = fig4["cO2(mol/cm3)"].to_numpy() * 1e6  # oxygen concentration [mol m–3]
    T = fig4["T(C)"].to_numpy() + 273.15  # temperatur [K]

    w2 = fig6["w(wt.%)"].to_numpy() / 100  # phosphoric acid mass concentration [0-1]
    dH = -fig6["-dH(kcal/mol)"].to_numpy() * 4184  # enthalpy of solution [J/mol]

    H = c / 1.01325e5  # Henry's law constant [mol m–3 Pa–1]
    R = GAS_CONST_SI  # Gas constant [J mol-1 K-1]

    if method == "sousa-2010":
        # H = B * exp( -dH/RT )
        # where
        # -dH/R = a1 w + b2
        # B = exp( b1 w + b2 )
        # ---
        # --> ln H = a1/T w + a2/T + b1 w + b2

        X_train = np.c_[w1 / T, 1 / T, w1]
        y_train = np.log(H)

        reg = LinearRegression()
        reg.fit(X_train, y_train)

    elif method == "mamlouk-2008":
        # c = B exp(-dH/RT)
        # 1/B = exp(-dH/RT) / c
        #     = a * w**5 + b * w**4 + c * w**3 + d*w**2 + e * w + f
        dH = oxygenSolutionEnthalpyModel(100 * w1)

        X_train = np.c_[
            (100 - 100 * w1) ** 5,
            (100 - 100 * w1) ** 4,
            (100 - 100 * w1) ** 3,
            (100 - 100 * w1) ** 2,
            (100 - 100 * w1) ** 1,
        ]
        y_train = (c * 1e7 * 1e-6) ** (-1) * np.exp(-dH / (R * T))

        reg = LinearRegression()
        reg.fit(X_train, y_train)

    elif method == "meck-2025":
        # H = B * exp( -dH/RT )
        # where
        # dH = a1 w**3 + a2 w**2 + a3 w + a4
        # B = exp( b1 w**3 + b2 w**2 + b3 w**1 + b4 )
        # ---
        # ln H = a1/RT w**3 + a2/RT w**2 + a3/RT w + a4/RT \
        #           + b1 w**3 + b2 w**2 + b3 w + b
        # -dH = a1 w**3 + a2 w**2 + a3 w + a4

        X1 = np.c_[
            w1**3 / (R * T),
            w1**2 / (R * T),
            w1**1 / (R * T),
            w1**0 / (R * T),
            w1**3,
            w1**2,
            w1**1,
            w1**0,
        ]

        X2 = np.c_[
            w2**3,
            w2**2,
            w2**1,
            w2**0,
            np.zeros(len(w2)),
            np.zeros(len(w2)),
            np.zeros(len(w2)),
            np.zeros(len(w2)),
        ]

        X_train = np.r_[X1, X2]
        y_train = np.r_[np.log(H), -dH]

        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, y_train)

    return reg


def oxygenSolutionEnthalpyModel(w: float, method="meck-2025"):
    """Enthalpy of Solution: dH in kcal, see Mamlouk, Sousa and Scott (2011, eq. (26)).
    Note: 1 cal ^= 4.184 J (Klinedinst et al., 1974)

    Inputs:
        W (float): the phosphoric acid (H3PO4) weight concentration in
    """
    x = np.array([w**3, w**2, w**1, w**0])
    if method == "mamlouk-2008":
        a = (
            np.array([-0.003125, 0.8371429, -74.95179, 2244.786])
            * np.array([100**3, 100**2, 100**1, 1])
            * 1e3
            * 4.184
        )
    elif method == "meck-2025":
        a = np.array([-1.85671526e07, 5.02433510e07, -4.54055345e07, 1.37108499e07])
        # a = np.array([-1.85671526e+07,  5.02433510e+07, -4.54055345e+07,  1.37108499e+07])
    return -np.dot(a, x)


def molarOxygenSolubilityInH3PO4(
    temperatureKelvin, weightConcentration, method="meck-2025"
):
    """
    Inputs:
        temperatureKelvin (float): temperature in K
        weightConcentration (float): the phosphoric acid (H3PO4) mass concentration (0-1)
    """
    T = np.atleast_1d(temperatureKelvin)
    w = np.atleast_1d(weightConcentration)
    if method == "mamlouk-2008":

        W = 100 * w
        dH = oxygenSolutionEnthalpyModel(w)

        x = np.array(
            [
                (100 - W) ** 5,
                (100 - W) ** 4,
                (100 - W) ** 3,
                (100 - W) ** 2,
                (100 - W),
                np.ones(len(w)),
            ]
        )

        a = np.array(
            [0.0004444022, -0.01678248, 0.2476135, -1.714433, 5.815734, -7.662641]
        )

        B = 1 / (np.dot(a, x)) * 1e-7 * 1e6
        # B * exp(-dH/RT) is in 107 mol/cm3
        # -> convert to mol/m3 and divide
        #      1.01325 bar to obtain Henry's
        #      law coefficient in  mol m-3 Pa-1

        val = B * np.exp(-dH / (GAS_CONST_SI * T[:, np.newaxis])) / 1.01325e5

    elif method == "sousa-2010":
        val = np.exp(
            (1 / T[:, np.newaxis]) * (-1.27e4 * w + 1.23e4) + (35.2 * w - 46.6)
        )
    elif method == "mamlouk-refit":

        W = 100 * w

        dH = oxygenSolutionEnthalpyModel(W)

        a = -0.000450729
        b = 0.0190256
        c = -0.307533
        d = 2.44172
        e = -9.15357
        f = 12.9925

        B = 1 / (
            a * (100 - W) ** 5
            + b * (100 - W) ** 4
            + c * (100 - W) ** 3
            + d * (100 - W) ** 2
            + e * (100 - W)
            + f
        )
        # B * exp(-dH/RT) is in 10^7 mol/cm3
        # -> convert to mol/m3 and divide
        #      1.01325 bar to obtain Henry's
        #      law coefficient in  mol m-3 Pa-1

        val = (
            B * np.exp(-dH / (GAS_CONST_SI * T[:, np.newaxis])) * 1e-7 * 1e6 / 1.01325e5
        )

    elif method == "sousa-refit":
        # re-estimated
        a = -15513.5
        b = 14854.5
        c = 42.3555
        d = -53.021
        val = np.exp((1 / T[:, np.newaxis]) * (a * w + b) + (c * w + d))

    elif method == "meck-2025":

        x2 = np.array([w**3, w**2, w**1, w**0])

        b = np.array([5.78242197e03, -1.56303408e04, 1.41115188e04, -4.27011795e03])
        # b = np.array([5.78242197e+03, -1.56303408e+04,  1.41115188e+04, -4.26321020e+03])
        dH = oxygenSolutionEnthalpyModel(w, method="meck-2025")
        logB = np.dot(b, x2)

        val = np.exp((-dH / (GAS_CONST_SI * T[:, np.newaxis])) + logB)

    return val
