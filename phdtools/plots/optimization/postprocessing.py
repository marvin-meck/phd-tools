"""phdtools.plots.optimization.postprocessing.py

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
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.integrate import solve_ivp

from phdtools.optimization import (
    THERMAL_EFFICIENCY_STATUS_QUO,
    SHIFT_MASS_OF_SOLIDS_GRAM_UB,
)

from phdtools.optimization.preprocessing import (
    get_reactorCostValues,
)

from phdtools.optimization.postprocessing import (
    get_optimization_results_space_time_reforming,
    get_optimization_results_space_time_shift,
    get_reference_values_fuel_cell,
)


def plot_optimization_result_fuel_cell(block):

    model_ref = get_reference_values_fuel_cell(block)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    _ = axs[0].plot(
        model_ref["currentDensityRangeSI"] * 1e-4,
        model_ref["fuelCellVoltageValues"],
        "k-",
    )
    _ = axs[1].plot(
        model_ref["currentDensityRangeSI"] * 1e-4,
        model_ref["powerDenstiyValues"] * 1e-4,
        "k-",
    )

    _ = axs[0].plot(
        pyo.value(block.currentDensitySI) * 1e-4,
        pyo.value(block.cellPotentialSI),
        marker="o",
        markerfacecolor="w",
        markeredgecolor="k",
    )
    _ = axs[1].plot(
        pyo.value(block.currentDensitySI) * 1e-4,
        pyo.value(block.powerDensitySI) * 1e-4,
        marker="o",
        markerfacecolor="w",
        markeredgecolor="k",
    )

    _ = axs[0].set_xlabel("Current density in A/cm2")
    _ = axs[0].set_ylabel("Voltage in V")

    _ = axs[1].set_xlabel("Current density in A/cm2")
    _ = axs[1].set_ylabel("Power density in W/cm2")

    for ax in axs:
        _ = ax.grid(True)
        _ = ax.set_xlim(0, np.ceil(pyo.value(block.currentDensityUpperBoundSI) * 1e-4))
        _ = ax.set_ylim(0, np.ceil(pyo.value(block.reversibleCellPotentialSI)))

    return fig


def plot_optimization_result_space_time_reforming(block):

    temperatureKelvin = pyo.value(block.temperatureKelvin)
    pressureBar = pyo.value(block.pressureBar)

    data_pyomo, data_scipy = get_optimization_results_space_time_reforming(block)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(
        label=f"Space-time vs. conversion;\nT = {pyo.value(temperatureKelvin):.2f} K; p = {pyo.value(pressureBar):.2f} bar"
    )

    _ = ax.plot(
        data_scipy["spaceTimeSI"] / 3.6,
        data_scipy["conversion"],
        "k",
        label="Scipy RK45",
    )

    _ = ax.plot(
        data_pyomo["spaceTimeSI"] / 3.6,
        data_pyomo["conversion"],
        "o--",
        color="k",
        mfc="white",
        mec="black",
        label="Pyomo (backward Euler)",
    )

    _ = ax.set_xlabel(r"Space-time $W/F_{\mathrm{CH4},0}$ in g(cat) hr / mol")
    _ = ax.set_ylabel(r"Conversion $X_{\mathrm{CH4}}$")
    _ = ax.legend()
    _ = ax.grid(True)

    return fig


def plot_optimization_result_space_time_shift(block):

    temperatureKelvin = pyo.value(block.temperatureKelvin)
    pressureBar = pyo.value(block.pressureBar)

    data_pyomo, data_scipy = get_optimization_results_space_time_shift(block)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(
        label=f"Space-time vs. conversion;\nT = {pyo.value(temperatureKelvin):.2f} K; p = {pyo.value(pressureBar):.2f} bar"
    )

    _ = ax.plot(
        data_scipy["spaceTimeSI"] / 3.6,
        data_scipy["conversion"],
        "k",
        label="Scipy RK45",
    )

    _ = ax.plot(
        data_pyomo["spaceTimeSI"] / 3.6,
        data_pyomo["conversion"],
        "o--",
        color="k",
        mfc="white",
        mec="black",
        label="Pyomo (backward Euler)",
    )

    _ = ax.set_xlabel(r"Space-time $W/F_{\mathrm{CO},0}$ in g(cat) hr / mol")
    _ = ax.set_ylabel(r"Conversion $X_{\mathrm{CO}}$")
    _ = ax.legend()
    _ = ax.grid(True)

    return fig


def plot_optimization_result_reactor_costs(block, fname_cost_coefs):

    df = get_reactorCostValues(fname_cost_coefs)

    catalystMassSI = df.loc[:, "catalystMassSI"]
    costValuesReformer = df.loc[:, "Reformer"]
    costValuesShift = df.loc[:, "Shift"]

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(catalystMassSI * 1e3, costValuesReformer, label="Reformer (R1)", color="k")
    ax.plot(
        catalystMassSI * 1e3, costValuesShift, label="Shift reactor (R2)", color="gray"
    )

    ax.plot(
        1e3 * pyo.value(block.reformer.massCatalystSI),
        pyo.value(
            block.costCoef["R1", "a1"]
            + block.costCoef["R1", "a2"]
            * block.reformer.massCatalystSI ** block.costCoef["R1", "k"]
        ),
        "o",
        mec="k",
        mfc="white",
    )
    ax.plot(
        1e3 * pyo.value(block.shift.massCatalystSI),
        pyo.value(
            block.costCoef["R2", "a1"]
            + block.costCoef["R2", "a2"]
            * block.shift.massCatalystSI ** block.costCoef["R2", "k"]
        ),
        "o",
        mec="gray",
        mfc="white",
    )

    ax.set_xlabel(r"Catalyst mass $W_i$ in $\mathrm{g}$")
    ax.set_ylabel(r"Cost $c_i$ in $\mathrm{EUR}$")
    ax.legend()

    ax.grid()

    ax.set_xlim(0, SHIFT_MASS_OF_SOLIDS_GRAM_UB)
    ax.set_ylim(1800, 2100)

    return fig


def create_power_cost_efficiency_plot(fname: str | os.PathLike) -> plt.Figure:

    frame = pd.read_csv(fname, index_col=0, comment="#")

    fig = plt.figure()
    ax = fig.gca()
    twin = ax.twinx()

    (p1,) = ax.plot(
        frame["Electrical power (kW)"],
        frame["Variable costs (Euro)"],
        "o-",
        mfc="white",
    )

    (p2,) = twin.plot(
        frame["Electrical power (kW)"],
        frame["Thermal efficiency"],
        "o-",
        color="gray",
        mec="gray",
        mfc="white",
    )
    _ = twin.plot(
        [250, 1050],
        [THERMAL_EFFICIENCY_STATUS_QUO, THERMAL_EFFICIENCY_STATUS_QUO],
        linestyle="--",
        color="gray",
    )
    _ = twin.annotate(
        r"Reference efficiency $\eta^{(\mathrm{SQ})}_\mathrm{th}$",
        (750, 0.81125),
        color="gray",
    )

    ax.set_xlabel("Electrical power in kW")
    ax.set_ylabel("Variable production costs in Euro")
    twin.set_ylabel("Thermal efficiency")

    # ax.set_xlim(3900,4300)

    # # map limits to twin axis range
    tmin, tmax = twin.dataLim.intervaly
    amin, amax = ax.dataLim.intervaly

    scale = (tmax - tmin) / (amax - amin)

    ax.set_xlim(0.250, 1.050)
    ax.set_ylim(8900, 9300)

    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()

    twin.set_ylim(tmin + (y1 - amin) * scale, tmin + (y2 - amin) * scale)

    twin.yaxis.label.set_color(p2.get_color())
    twin.tick_params(axis="y", colors=p2.get_color())
    # twin.yaxis.set_inverted(True)

    ax.grid()

    return fig


def plot_power_vs_cost_vs_price(
    fname_markup: str | os.PathLike, fname_contribution: str | os.PathLike
) -> plt.Figure:

    markup = pd.read_csv(
        fname_markup,
        comment="#",
        index_col=0,
    )

    contribution = pd.read_csv(
        fname_contribution,
        comment="#",
        index_col=0,
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(
        markup["Electrical power (kW)"],
        markup["Variable costs (Euro)"],
        marker=".",
        label="Variable costs",
    )
    axs[0].plot(
        markup["Electrical power (kW)"],
        markup["Markup = 1.75"],
        marker=".",
        label="Markup = 1.75",
    )
    axs[0].plot(
        markup["Electrical power (kW)"],
        markup["Markup = 2.00"],
        marker=".",
        label="Markup = 2.00",
    )
    axs[0].plot(
        markup["Electrical power (kW)"],
        markup["Markup = 2.25"],
        marker=".",
        label="Markup = 2.25",
    )

    axs[1].plot(
        contribution["Electrical power (kW)"],
        contribution["Variable costs (Euro)"],
        marker=".",
        label="Variable costs",
    )
    axs[1].plot(
        contribution["Electrical power (kW)"],
        contribution["Contribution = 7000.00 (Euro)"],
        marker=".",
        label="CM = 7000 (Euro)",
    )
    axs[1].plot(
        contribution["Electrical power (kW)"],
        contribution["Contribution = 10000.00 (Euro)"],
        marker=".",
        label="CM = 10000 (Euro)",
    )
    axs[1].plot(
        contribution["Electrical power (kW)"],
        contribution["Contribution = 12000.00 (Euro)"],
        marker=".",
        label="CM = 12500 (Euro)",
    )

    axs[0].set_xlabel(r"Electrical power $P_N$ in kW")
    axs[1].set_xlabel(r"Electrical power $P_N$ in kW")

    axs[0].set_ylabel(r"Price $P^\ast$ in Euro")
    axs[1].set_ylabel(r"Price $P^\ast$ in Euro")

    for ax in axs:
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(8000, 22500)
        ax.grid()
        ax.legend()

    return fig


def plot_cost_minimization_price_vs_market_share(
    fname_contribution, powerValuesSI=None, choice_model="mnl", ax=None
):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # Constant contribution
    df_contribution = pd.read_csv(fname_contribution, comment="#")

    df_contribution.loc[:, "Contribution margin (Euro)"] = (
        np.round(1e4 * df_contribution["Contribution margin (Euro)"]) * 1e-4
    )
    df_contribution.loc[:, "Electrical power (kW)"] = (
        np.round(1e4 * df_contribution["Electrical power (kW)"]) * 1e-4
    )

    if powerValuesSI is None:
        powerValuesSI = df_contribution["Electrical power (kW)"].unique() * 1e3

    for electricalPowerSI in np.flip(powerValuesSI):
        mask = np.isclose(
            df_contribution["Electrical power (kW)"], 1e-3 * electricalPowerSI
        )

        df = df_contribution[mask]
        if choice_model == "mnl":
            ax.plot(
                df["Price (Euro)"] - df["Variable costs (Euro)"],
                df["Market share, MNL (percent)"],
                linestyle="solid",
                label=rf"$ P = {electricalPowerSI*1e-3:.2f}\,\mathrm{{kW}}$",
            )
        elif choice_model == "mxl":
            ax.plot(
                df["Price (Euro)"] - df["Variable costs (Euro)"],
                df["Market share, MXL (percent)"],
                linestyle="solid",
                label=rf"$ P = {electricalPowerSI*1e-3:.2f}\,\mathrm{{kW}}$",
            )

    return fig


def plot_cost_minimization_price_vs_total_contribution(
    fname_contribution, powerValuesSI=None, choice_model="mnl", ax=None
):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # Constant contribution
    df_contribution = pd.read_csv(fname_contribution, comment="#")

    df_contribution.loc[:, "Contribution margin (Euro)"] = (
        np.round(1e4 * df_contribution["Contribution margin (Euro)"]) * 1e-4
    )
    df_contribution.loc[:, "Electrical power (kW)"] = (
        np.round(1e4 * df_contribution["Electrical power (kW)"]) * 1e-4
    )

    if powerValuesSI is None:
        powerValuesSI = df_contribution["Electrical power (kW)"].unique() * 1e3

    for electricalPowerSI in np.flip(powerValuesSI):
        mask = np.isclose(
            df_contribution["Electrical power (kW)"], 1e-3 * electricalPowerSI
        )

        df = df_contribution[mask]

        if choice_model == "mnl":
            ax.plot(
                df["Price (Euro)"] - df["Variable costs (Euro)"],
                df["Normalized total contribution, MNL (Euro)"],
                linestyle="solid",
                label=rf"$ P = {electricalPowerSI*1e-3:.2f}\,\mathrm{{kW}}$",
            )
        elif choice_model == "mxl":
            ax.plot(
                df["Price (Euro)"] - df["Variable costs (Euro)"],
                df["Normalized total contribution, MXL (Euro)"],
                linestyle="solid",
                label=rf"$ P = {electricalPowerSI*1e-3:.2f}\,\mathrm{{kW}}$",
            )

    return fig
