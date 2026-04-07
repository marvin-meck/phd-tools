"""phdtools.plots.fuel_cell.py

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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phdtools import DATA_DIR


def plot_stst_cell_potential(
    fname: str | os.PathLike,
) -> plt.Figure:

    stdCellPotentialSI = pd.read_csv(fname, index_col=0, comment="#")

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

    _ = stdCellPotentialSI.plot(
        ax=ax,
        xlabel="Temperature in K",
        ylabel=r"Standard cell potential in V",
        style="k.--",
    )

    _ = ax.grid(True)

    return fig


def plot_arrhenius_model_exchange_current_density(
    fname_data: str | os.PathLike,
    fname_model: str | os.PathLike,
) -> plt.Figure:

    data = pd.read_csv(fname_data, usecols=[0, 1, 2, 3, 4])
    data["T(K)"] = data["Temperature (C)"] + 273.15
    data = data.drop("Temperature (C)", axis=1).set_index("T(K)")

    model = pd.read_csv(fname_model, comment="#", index_col=0)

    fig = plt.figure()
    ax = fig.gca()

    for label, name in {
        "Anode": "i^0_H2, apparent (A cm-2)",
        "Cathode": "i^0_O2, apparent (A cm-2)",
    }.items():
        ax.plot(
            1000 / data.index,
            np.log(data[name]),
            marker=".",
            linestyle="none",
            label=f"{label}, Experiment (Zhang, 2007b)",
        )

    ax.plot(
        1000 / (model.index),
        np.log(model["j0,A(Acm-2)"]),
        marker="none",
        linestyle="-",
        color="black",
        label=f"Anode, Arrhenius model (Zhang, 2007b)",
    )
    ax.plot(
        1000 / (model.index),
        np.log(model["j0,C(Acm-2)"]),
        marker="none",
        linestyle="-",
        color="gray",
        label=f"Cathode, Arrhenius model (Zhang, 2007b)",
    )

    # _ = ax.title(r"Temperature dependence of the exchange current density at std. concentration")
    _ = ax.set_xlabel(r"Inverse temperature $1000/T$ in K")
    _ = ax.set_ylabel(
        r"Log. exchange current density $\ln{\left\{ j_{0,i}^0 \, / (1\,\mathrm{A\,cm^{-2}}) \right\}}$"
    )
    _ = ax.legend()

    _ = ax.grid()

    return fig


def plot_reversible_cell_potential_vs_temperature(
    fname: str | os.PathLike,
) -> plt.Figure:

    df = pd.read_csv(
        fname,
        comment="#",
        index_col=0,
        header=0,
        names=["T(K)", "E0", "1.01325", "2.02650", "5.06625"],
    ).drop("E0", axis=1)

    fig = plt.figure()
    ax = fig.gca()

    for col in df.columns:
        ax.plot(df.index, df[col], label=rf"$p = {col}\,\mathrm{{bar}}$")

    _ = ax.set_xlabel("Temperature in K")
    _ = ax.set_ylabel("Reversible cell potential in V")

    _ = ax.legend()
    _ = ax.grid()

    return fig


def plot_tafel_plot(
    fname_bv_hor: str | os.PathLike,
    fname_bv_orr: str | os.PathLike,
    fname_tf_hor: str | os.PathLike,
    fname_tf_orr: str | os.PathLike,
) -> plt.Figure:

    fig, ax = plt.subplots(1, 1)

    temperatureKelvin = 120 + 273.15

    # ORR -- Butler-Volmer
    bv_orr = pd.read_csv(fname_bv_orr, comment="#", index_col=0)
    mask_cathodic = bv_orr.index < 0
    mask_anodic = bv_orr.index > 0

    # for temperatureCelsius in bv_orr.columns:
    _ = ax.plot(
        np.log10(-bv_orr[mask_cathodic].index),
        bv_orr[mask_cathodic][f"{temperatureKelvin}"],
        label="ORR (cathodic)",
        color="k",
        linestyle="solid",
    )

    # ORR -- Tafel
    tf_orr = pd.read_csv(fname_tf_orr, comment="#", index_col=0)
    mask_cathodic = tf_orr.index < 0
    mask_anodic = tf_orr.index > 0
    mask_tafel_positive = tf_orr[f"{temperatureKelvin}"] >= 0
    mask_tafel_negative = tf_orr[f"{temperatureKelvin}"] <= 0

    # for temperatureCelsius in tf_orr.columns:
    _ = ax.plot(
        np.log10(-tf_orr[mask_cathodic & mask_tafel_negative].index),
        tf_orr[mask_cathodic & mask_tafel_negative][f"{temperatureKelvin}"],
        label="",
        color="k",
        linestyle="dotted",
    )

    # HOR -- Butler-Volmer
    bv_hor = pd.read_csv(fname_bv_hor, comment="#", index_col=0)
    mask_cathodic = bv_hor.index < 0
    mask_anodic = bv_hor.index > 0

    # for temperatureCelsius in bv_orr.columns:
    _ = ax.plot(
        np.log10(bv_hor[mask_anodic].index),
        bv_hor[mask_anodic][f"{temperatureKelvin}"],
        label="HOR (anodic)",
        color="gray",
        linestyle="solid",
    )

    # HOR -- Tafel
    tf_hor = pd.read_csv(fname_tf_hor, comment="#", index_col=0)
    mask_cathodic = tf_hor.index < 0
    mask_anodic = tf_hor.index > 0
    mask_tafel_positive = tf_hor[f"{temperatureKelvin}"] >= 0
    mask_tafel_negative = tf_hor[f"{temperatureKelvin}"] <= 0

    _ = ax.plot(
        np.log10(tf_hor[mask_anodic & mask_tafel_positive].index),
        tf_hor[mask_anodic & mask_tafel_positive][f"{temperatureKelvin}"],
        label="",
        color="k",
        linestyle="dotted",
    )

    _ = ax.set_xlabel(r"Current density $\log_{10}(|j|/1\,\mathrm{A\,cm^{-2}})$")
    _ = ax.set_ylabel(r"Activation overpotential $\eta_\mathrm{act}$ in V")

    _ = ax.legend()

    _ = ax.set_xlim(-4, 2)
    _ = ax.set_ylim(-0.6, 0.2)
    _ = ax.grid()

    return fig


def plot_activation_overpotential(fname, branch="cathodic"):

    df = pd.read_csv(fname, comment="#", index_col=0)

    activationOverpotentialCathodeSI = df.to_numpy().T
    currentDensityRangeSI = df.index.to_numpy() * 1e4
    temperatureRangeSI = df.columns.to_numpy(dtype=float)
    mask = currentDensityRangeSI < 0

    fig = plt.figure()
    ax = fig.gca()

    for num, temperatureKelvin in enumerate(temperatureRangeSI):
        if branch == "cathodic":
            _ = ax.plot(
                abs(currentDensityRangeSI[mask]) * 1e-4,
                activationOverpotentialCathodeSI[num, mask],
                label=f"{temperatureKelvin-273.15} °C",
            )
        elif branch == "anodic":
            _ = ax.plot(
                abs(currentDensityRangeSI[~mask]) * 1e-4,
                activationOverpotentialCathodeSI[num, ~mask],
                label=f"{temperatureKelvin} K",
            )

    _ = ax.set_xlabel(r"Current density $|j|$")
    _ = ax.set_ylabel(r"Activation overpotential $\eta_{\mathrm{act},C}$")

    _ = ax.legend()
    _ = ax.grid()

    _ = ax.set_xlim(0, 2)
    # _ = ax.set_ylim(0,-.5)

    return fig


def plot_conductivity_vs_temperature(
    fname: str | os.PathLike,
) -> plt.Figure:
    model = pd.read_csv(fname, comment="#")

    fig, ax = plt.subplots(1, 1)
    _ = ax.plot(model["T(K)"], model["s(S/cm)"])

    _ = ax.set_xlim(model["T(K)"].min(), model["T(K)"].max())
    _ = ax.set_ylim(0.2, 0.7)

    _ = ax.set_xlabel(r"Temperature in $\mathrm{K}$")
    _ = ax.set_ylabel(r"Conductivity in $\mathrm{S/cm}$")

    ax.grid()

    return fig


def plot_internal_resistance_vs_temperature(fname: str | os.PathLike) -> plt.Figure:

    model = pd.read_csv(fname, comment="#", index_col=0)

    fig, ax = plt.subplots(1, 1)

    for temperatureCelsius in model.columns:
        _ = ax.plot(
            model.index, model[temperatureCelsius], label=f"{temperatureCelsius} °C"
        )

    _ = ax.set_xlim(model.index.min(), model.index.max())
    _ = ax.set_ylim(0, 0.4)

    _ = ax.set_xlabel(r"Current density in $\mathrm{A/cm}^2$")
    _ = ax.set_ylabel(r"Ohmic losses in $V$")
    _ = ax.legend()

    ax.grid()

    return fig


def plot_mole_fractions_gdl(
    fname: str | os.PathLike,
) -> plt.Figure:

    moleFraction = pd.read_csv(fname, comment="#", index_col=0)
    coordinateRange = moleFraction.index.to_numpy()

    fig = plt.figure()
    ax = fig.gca()

    for c in moleFraction.columns:
        _ = ax.plot(coordinateRange, moleFraction[c], label=f"{c}")

    # _ = ax.set_title(
    #     "Maxwell-Stefan diffusion model (cathode)\n"
    #     + r"$j = {}\,\mathrm{{A/cm^2}}$; $T = {}\,\mathrm{{^\circ C}}$; $p = {}\,\mathrm{{bar}}$".format(
    #         currentDensitySI*1e-4,
    #         temperatureKelvin-273.15,
    #         pressureBar
    #     )
    # )
    _ = ax.set_xlabel(r"Coordinate $z_+ := z / \delta$")
    _ = ax.set_ylabel(r"Mole fraction $x_{i}$")

    _ = ax.set_xlim(0, 1)
    _ = ax.set_ylim(0, 1)

    _ = ax.grid()
    _ = ax.legend()

    return fig


def plot_interface_mole_fractions(
    fname_hydrogen: str | os.PathLike, fname_oxygen: str | os.PathLike
) -> plt.Figure:

    moleFractionHydrogenInterface = pd.read_csv(
        fname_hydrogen, comment="#", index_col=0
    )
    currentDensityRangeSI = moleFractionHydrogenInterface.index.to_numpy() * 1e4

    fig, axs = plt.subplots(2, 1)

    for temperatureCelsius in moleFractionHydrogenInterface.columns:
        _ = axs[0].plot(
            currentDensityRangeSI * 1e-4,
            moleFractionHydrogenInterface[temperatureCelsius],
            label=f"T = {temperatureCelsius} C",
        )

    moleFractionOxygenInterface = pd.read_csv(fname_oxygen, comment="#", index_col=0)
    currentDensityRangeSI = moleFractionOxygenInterface.index.to_numpy() * 1e4

    for temperatureCelsius in moleFractionOxygenInterface.columns:
        _ = axs[1].plot(
            currentDensityRangeSI * 1e-4,
            moleFractionOxygenInterface[temperatureCelsius],
            label=f"T = {temperatureCelsius} C",
        )

    _ = axs[0].set_xlabel(r"Current density $j$ in $\mathrm{A/cm^2}$")
    _ = axs[0].set_ylabel(r"$x^\ast_{\mathrm{H_2}} / x^0_{\mathrm{H_2}}$")
    # _ = axs[0].legend()

    _ = axs[1].set_xlabel(r"Current density $j$ in $\mathrm{A/cm^2}$")
    _ = axs[1].set_ylabel(r"$x^\ast_{\mathrm{O_2}} / x^0_{\mathrm{O_2}}$")
    _ = axs[1].legend()

    for ax in axs:
        _ = ax.set_xlim(
            1e-4 * currentDensityRangeSI.min(), 1e-4 * currentDensityRangeSI.max()
        )
        _ = ax.set_ylim(0, 1)
        _ = ax.grid()

    return fig


def plot_concentration_overpotential(fname: str | os.PathLike) -> plt.Figure:

    df = pd.read_csv(fname, comment="#", index_col=0)

    temperatureRangeSI = df.columns.to_numpy(dtype=float) + 273.15
    currentDensityRangeSI = df.index.to_numpy() * 1e4
    concentrationOverpotentialValuesSI = df.to_numpy().T

    fig, ax = plt.subplots(1, 1)

    _ = ax.plot(
        currentDensityRangeSI * 1e-4,
        concentrationOverpotentialValuesSI.T,
        label=[
            f"{temperatureKelvin-273.15} °C" for temperatureKelvin in temperatureRangeSI
        ],
    )

    _ = ax.set_xlabel(r"Current density $j$ in $\mathrm{A/cm^2}$")
    _ = ax.set_ylabel(r"Concentration losses $\eta_\mathrm{conc}$ in $V$")
    _ = ax.legend()

    _ = ax.set_xlim(
        currentDensityRangeSI.min() * 1e-4, currentDensityRangeSI.max() * 1e-4
    )
    _ = ax.set_ylim(-0.1, 0)
    _ = ax.grid()

    return fig


def plot_polarization_curve(
    fname: str | os.PathLike,
    experiment: str | os.PathLike = DATA_DIR
    / "zhang-2007"
    / "251020_polarization_curve.csv",
) -> plt.Figure:

    model = pd.read_csv(fname, comment="#")
    data = pd.read_csv(experiment)

    fig = plt.figure()
    ax = fig.gca()

    _ = ax.plot(model["j(A/cm2)"], model["T = 393.15 K"], label="Model (120 C)")
    _ = ax.plot(model["j(A/cm2)"], model["T = 413.15 K"], label="Model (140 C)")
    _ = ax.plot(model["j(A/cm2)"], model["T = 433.15 K"], label="Model (160 C)")
    _ = ax.plot(model["j(A/cm2)"], model["T = 453.15 K"], label="Model (180 C)")
    _ = ax.plot(model["j(A/cm2)"], model["T = 473.15 K"], label="Model (200 C)")

    for temperatureCelsius in data["T(C)"].unique():
        mask = data["T(C)"] == temperatureCelsius
        _tmp = data[mask]

        _ = ax.plot(
            _tmp["j(A/cm2)"],
            _tmp["U(V)"],
            ".",
            label=rf"$T = {temperatureCelsius} \, \mathrm{{^\circ C}}$",
        )

    _ = ax.set_xlabel(r"Current density in $\mathrm{A/cm}^2$")
    _ = ax.set_ylabel(r"Voltage in V")
    _ = ax.legend()

    _ = ax.set_xlim(model["j(A/cm2)"].min(), model["j(A/cm2)"].max())
    _ = ax.set_ylim(
        0,
    )

    _ = ax.grid()

    return fig


def plot_fuel_cell_characteristic(
    fname: str | os.PathLike,
) -> plt.Figure:

    df = pd.read_csv(fname, comment="#")

    fig, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()

    _ = ax1.plot(df["j(A/m2)"] / 1e4, df["U(V)"], label="Cell voltage")
    _ = ax2.plot(
        df["j(A/m2)"] / 1e4, df["P(W/m2)"] / 1e4, "gray", label="Power density"
    )

    _ = ax1.set_xlim(0, 2)
    _ = ax1.set_ylim(0, 1.2)

    _ = ax2.set_xlim(0, 2)
    _ = ax2.set_ylim(0, 1.4)

    _ = ax1.set_xlabel(r"Current density in $\mathrm{A/cm}^2$")
    _ = ax1.set_ylabel(r"Voltage in V")

    _ = ax2.set_xlabel(r"Current density in $\mathrm{A/cm}^2$")
    _ = ax2.set_ylabel(r"Power denisty in $\mathrm{W/cm^2}$", color="gray")

    _ = ax2.tick_params(axis="y", colors="gray")
    _ = ax2.spines["right"].set_color("gray")

    _ = ax1.grid()

    return fig
