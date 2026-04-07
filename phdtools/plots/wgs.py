"""phdtools.plots.wgs.py

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

import matplotlib.pyplot as plt
import pandas as pd

from phdtools import DATA_DIR

markerStyle = {
    120: {
        "marker": "s",
        "color": "black",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
    },
    155: {
        "marker": "s",
        "color": "black",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
    },
    175: {
        "marker": "^",
        "color": "black",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
    },
    190: {
        "marker": "^",
        "color": "black",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
    },
    220: {
        "marker": "o",
        "color": "black",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
    },
    250: {
        "marker": "o",
        "color": "black",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
    },
}


def plot_equilibrium_shift_reactor(fname):

    fig = plt.figure()
    ax = fig.gca()

    df = pd.read_csv(fname, comment="#", index_col=0)

    _ = ax.plot(df.index, df["C1O1(g)"], "k")

    _ = ax.set_xlim(200 + 273.15, 450 + 273.15)
    _ = ax.set_ylim(0, 0.05)

    _ = ax.set_xlabel(r"Temperature in K")
    _ = ax.set_ylabel(r"Mole fraction $x_\mathrm{CO}$")
    _ = ax.grid(True)

    return fig


def plot_conversion_vs_steam_to_carbon_experiment(
    fname=DATA_DIR
    / "choi-stenger-2003"
    / "250723_figure_2_experimental_choi_stenger_2003.csv",
    ax=None,
    ls="--",
    **kwargs,
):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    fig = ax.get_figure()

    data = pd.read_csv(fname, comment="#", names=["H2O/CO", "XCO", "T(C)"])
    temperatureRange = data["T(C)"].unique()

    for temperatureCelsius in temperatureRange:
        mask = data["T(C)"] == temperatureCelsius
        _ = ax.plot(
            data[mask]["H2O/CO"],
            data[mask]["XCO"],
            label=f"T = {temperatureCelsius} °C",
            linestyle=ls,
            **markerStyle[temperatureCelsius],
        )

    _ = ax.set_xlim(0, 5)
    _ = ax.set_ylim(0, 1)

    _ = ax.set_xlabel(r"Steam to carbon ratio $\mathrm{H_2O} / \mathrm{CO}$")
    _ = ax.set_ylabel(r"Conversion $X_\mathrm{CO}$")

    _ = ax.legend()
    _ = ax.grid(True)

    return fig


def plot_conversion_vs_space_velocity(
    fname=DATA_DIR
    / "choi-stenger-2003"
    / "250723_figure_3_experimental_choi_stenger_2003.csv",
    ax=None,
    ls="--",
    **kwargs,
):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    fig = ax.get_figure()

    data = pd.read_csv(fname, comment="#", names=["GHSV(1/h)", "XCO", "T(C)"])
    temperatureRange = data["T(C)"].unique()

    for temperatureCelsius in temperatureRange:
        mask = data["T(C)"] == temperatureCelsius
        _ = ax.plot(
            data[mask]["GHSV(1/h)"],
            data[mask]["XCO"],
            label=f"T = {temperatureCelsius} °C",
            linestyle=ls,
            **markerStyle[temperatureCelsius],
        )

    _ = ax.set_xlim(
        0,
    )
    _ = ax.set_ylim(0, 1)

    _ = ax.set_xlabel(r"GHSV in $\mathrm{h}^{-1}$")
    _ = ax.set_ylabel(r"Conversion $X_\mathrm{CO}$")

    _ = ax.legend()
    _ = ax.grid(True)

    return fig


def plot_conversion_vs_space_time_converted(
    data,
    # fname = DATA_DIR / "choi-stenger-2003" / "250723_figure_3_experimental_choi_stenger_2003.csv",
    ax=None,
    ls="--",
    **kwargs,
):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    fig = ax.get_figure()

    # data = pd.read_csv(fname, comment="#", names=["GHSV(1/h)","XCO","T(C)"])
    temperatureRange = data["T(C)"].unique()

    for temperatureCelsius in temperatureRange:
        mask = data["T(C)"] == temperatureCelsius
        _ = ax.plot(
            data[mask]["W/F_CO,0(kg*s/mol)"] / 3.6,
            data[mask]["XCO"],
            label=f"T = {temperatureCelsius} °C",
            linestyle=ls,
            **markerStyle[temperatureCelsius],
        )

    _ = ax.set_xlim(
        0,
    )
    _ = ax.set_ylim(0, 1)

    _ = ax.set_xlabel(r"$W/F_{\mathrm{CO},0}$ in $\mathrm{g(cat)\,h/mol}$")
    _ = ax.set_ylabel(r"Conversion $X_{\mathrm{CO}}$")
    _ = ax.legend()
    _ = ax.grid(True)

    return fig


def plot_eq_constant_temperature_dependence(file_model, file_tabulated, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    logEqulibriumConstModelValues = pd.read_csv(file_model, comment="#", index_col=0)
    logEquilibriumConstTabulated = pd.read_csv(file_tabulated, comment="#", index_col=0)

    _ = ax.plot(
        1000 / logEqulibriumConstModelValues.index.to_numpy(),
        logEqulibriumConstModelValues["choi11"],
        label="Chinchen et al. (1988)",
    )
    _ = ax.plot(
        1000 / logEqulibriumConstModelValues.index.to_numpy(),
        logEqulibriumConstModelValues["choi12"],
        label="Moe (1961)",
    )
    _ = ax.plot(
        1000 / logEqulibriumConstModelValues.index.to_numpy(),
        logEqulibriumConstModelValues["vantHoff"],
        label=r"van't Hoff equation ($T_m = 500 \,\mathrm{K}$)",
    )
    _ = ax.plot(
        1000 / logEquilibriumConstTabulated["WGS"].index.to_numpy(),
        logEquilibriumConstTabulated["WGS"].to_numpy(),
        "k.",
        label="NIST (1998)",
    )

    _ = ax.legend()
    _ = ax.set_title(r"Temperature dependence of the equlibrium constant")
    _ = ax.set_xlabel(r"$1000/T$ in $\mathrm{1/K}$")
    _ = ax.set_ylabel(r"$\log K$")

    _ = ax.set_xlim(1, 3.5)
    _ = ax.set_ylim(0, 12)
    _ = ax.grid(True)

    return fig


def plot_space_time_conversion_mendes_2010(file_experiment, file_model, model, ax=None):

    experiment_frame = pd.read_csv(file_experiment, comment="#")
    model_frame = pd.read_csv(
        file_model,
        comment="#",
        header=None,
        index_col=0,
        names=[
            "W/F_CO,0(gcat*h/mol)",
            "180(C)",
            "190(C)",
            "200(C)",
            "230(C)",
            "250(C)",
            "300(C)",
        ],
    )

    markerStyle = {
        180: {
            "marker": "^",
            "color": "black",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        190: {
            "marker": "v",
            "color": "black",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        200: {
            "marker": "o",
            "color": "black",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        230: {
            "marker": "s",
            "color": "black",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        250: {
            "marker": "D",
            "color": "black",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        300: {
            "marker": "h",
            "color": "black",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
    }

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    ax = model_frame.plot(ax=ax, style="k-")
    for line in ax.get_lines():
        line.set_label(f"Model ({model})")

    for temperatureCelsius in experiment_frame["T(C)"].unique():
        mask = experiment_frame["T(C)"] == temperatureCelsius
        x = experiment_frame[mask]["W/F_CO,0(gcat*h/mol)"]
        y = experiment_frame[mask]["X_CO"]

        ax.plot(
            x,
            y,
            linestyle="none",
            label=f"T = {temperatureCelsius} °C",
            **markerStyle[temperatureCelsius],
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    _ = ax.set_xlabel(r"Space-time $W/F_{\mathrm{CO},0}$ in $\mathrm{g(cat)\,h / mol}$")
    _ = ax.set_ylabel(r"Conversion $X_{\mathrm{CO}}$")

    _ = ax.set_xlim(0, 75)
    _ = ax.set_ylim(0, 1)
    _ = ax.grid(True)

    return fig


def plot_parity_rate_of_conversion_mendes_2010(fname, method="central"):

    fig, axs = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))

    df = pd.read_csv(fname, comment="#")
    # df.dropna()
    # df["T(K)"] = temperatureValues
    # df["X_CO"] = conversionValues
    # df["W/F_CO,0(gcat*h/mol)"] = spaceTimeValues

    # mask = abs(df["X_CO"]) >= 1e-6
    # df = df[mask]

    styles = [
        {"marker": "o", "mec": "k", "mfc": "white", "linestyle": "none"},
        {"marker": "s", "mec": "k", "mfc": "white", "linestyle": "none"},
    ]

    for num, model in enumerate(["Moe", "Power law"]):
        if model != "Combined":
            axs[0].plot(
                -1 * df[f"Experiment ({method})"],
                -1 * df[model],
                label=f"{model}",
                **styles[num],
            )

    styles = [
        {"marker": "o", "mec": "k", "mfc": "white", "linestyle": "none"},
        {"marker": "^", "mec": "k", "mfc": "white", "linestyle": "none"},
        {"marker": "s", "mec": "k", "mfc": "white", "linestyle": "none"},
    ]
    for num, model in enumerate(["LH1", "LH2", "Redox"]):
        if model != "Combined":
            axs[1].plot(
                -1 * df[f"Experiment ({method})"],
                -1 * df[model],
                label=f"{model}",
                **styles[num],
            )

    for ax in axs:

        ax.plot([0, 0.14], [0, 0.14], "k-")

        ax.set_xlim(0, 0.14)
        ax.set_ylim(0, 0.14)

        ax.legend()
        ax.grid(True)

        ax.set_xlabel(
            r"Experimental reaction rate $- r_\mathrm{exp}$ in $\mathrm{mol/(g(cat)\,h)}$"
        )
        ax.set_ylabel(
            r"Predicted reaction rate $- r_\mathrm{pred}$ in $\mathrm{mol/(g(cat)\,h)}$"
        )

    return fig
