"""phdtools.plots.smr.py

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phdtools import DATA_DIR


def plot_eq_constant_temperature_dependence(
    file_model, file_tabulated, temperatureRange=np.linspace(600, 1600)
):

    logEquilibriumConst = pd.read_csv(file_model, index_col="T(K)", comment="#")
    logEquilibriumConstModelValues = pd.read_csv(
        file_tabulated,
        header=None,
        skiprows=1,
        names=["T(K)", "I", "II", "III"],
        index_col=0,
        comment="#",
    )

    # TODO: filter temperature range
    # mask = logEquilibriumConstModelValues.index >= temperatureRange.min()

    fig = plt.figure()
    ax = fig.gca()

    mask = (logEquilibriumConst.index >= 600) & (logEquilibriumConst.index <= 1600)
    _ = ax.plot(
        1000 / logEquilibriumConstModelValues.index,
        logEquilibriumConstModelValues.to_numpy(),
        label=["Model for log10K1", "Model for log10K2", "Model for log10K3"],
    )
    _ = ax.plot(
        1000 / logEquilibriumConst[mask].index,
        logEquilibriumConst[mask].to_numpy(),
        ".k",
        label=["NIST-JANAF, Allison (2013)", "", ""],
    )
    _ = ax.set_xlabel(r"1000/T in 1/K")
    _ = ax.set_ylabel(r"Equilibrium constant $\log K_i$")

    _ = ax.legend()
    _ = ax.set_xlim(1000 / temperatureRange.max(), 1000 / temperatureRange.min())
    _ = ax.grid(True)

    return fig


def figure_two_xu_froment(
    file_model,
    file_experiment=DATA_DIR
    / "xu-froment-1989"
    / "241007_figure_2_experiment_xu_froment.csv",
):

    fig = plt.figure()
    ax = fig.gca()

    df1 = pd.read_csv(
        file_experiment,
        sep=",",
        comment="#",
        names=["T(K)", "W/F_CH4,0", "X_CH4"],
        index_col=0,
    )
    df2 = pd.read_csv(
        file_model,
        sep=",",
        comment="#",
        header=None,
        names=["W/F_CH4,0", "X_CH4", "T(K)"],
    )

    _ = ax.plot(df1.iloc[:, 0], df1.iloc[:, 1], "k.")
    for t in pd.unique(df2.iloc[:, 2]):
        mask = df2.iloc[:, 2] == t
        _ = ax.plot(df2[mask].iloc[:, 0], df2[mask].iloc[:, 1], "-", label=f"T = {t}K")

    _ = ax.legend()
    _ = ax.set_xlabel(r"$W/F_\mathrm{CH_4}^\mathrm{in}$ in g(cat) hr / mol")
    _ = ax.set_ylabel(r"Conversion $X_\mathrm{CH_4}$")

    _ = ax.set_xlim(0, 0.5)
    _ = ax.set_ylim(0, 0.15)
    _ = ax.grid(True)

    return fig


def plot_equilibrium_space_time(fname_conversion, fname_equilibrium):

    df1 = pd.read_csv(fname_conversion, comment="#", index_col=0)
    df2 = pd.read_csv(fname_equilibrium, comment="#", header=None, index_col=0)

    fig = plt.figure()
    ax = fig.gca()

    # Plot every 5th column and label only the first line
    lines = df1[df1.columns[::5]].plot(ax=ax, style="k", legend=False)
    for i, line in enumerate(ax.get_lines()[: len(df1.columns[::5])]):
        if i == 0:
            line.set_label("PFR model (Xu and Froment, 1989)")
        else:
            line.set_label("_nolegend_")  # Skip in legend

    # Plot equilibrium data, label just once
    eq_lines = df2.plot(ax=ax, style="--", legend=False)
    for i, line in enumerate(ax.get_lines()[len(df1.columns[::5]) :]):
        if i == 0:
            line.set_label("Equilibrium")
        else:
            line.set_label("_nolegend_")

    _ = ax.set_xlabel(r"$W/F_\mathrm{CH_4}^\mathrm{in}$ in g(cat) hr / mol")
    _ = ax.set_ylabel(r"Conversion $X_\mathrm{CH_4}$")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    _ = ax.set_xlim(0, 0.5)
    _ = ax.set_ylim(0, 1)
    _ = ax.grid(True)

    return fig


def plot_molefraction_vs_conversion(fname):

    fig = plt.figure()
    ax = fig.gca()

    moleFractions = pd.read_csv(fname, comment="#", index_col=0)

    moleFractions.plot(ax=ax)

    _ = ax.set_xlabel(r"Conversion $X_\mathrm{CH_4}$")
    _ = ax.set_ylabel(r"Mole fractions $x_{i}$")

    _ = ax.set_xlim(0, 1)
    _ = ax.set_ylim(0, 1)
    _ = ax.grid(True)
