"""phdtools.plots.optimization.preprocessing.py

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
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_floor_area_by_year_bar_chart(
    fname: str | os.PathLike,
) -> plt.Figure:

    table = pd.read_csv(fname, index_col=0, comment="#")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.autofmt_xdate(rotation=45)

    M = table.iloc[:-1, :-1].to_numpy(dtype=float) / table.loc["Total", "Total"]

    bottom = np.zeros(len(table.index[:-1]))
    for j, year in enumerate(table.columns[:-1]):
        x = table.index[:-1]
        heights = M[:, j]
        _ = axs[0].bar(x, heights, bottom=bottom, label=year)
        bottom += heights

    bottom = np.zeros(len(table.columns[:-1]))
    for j, floor in enumerate(table.index[:-1]):
        x = table.columns[:-1]
        heights = M[j, :]
        _ = axs[1].bar(x, heights, bottom=bottom, label=floor)
        bottom += heights

    for ax in axs:
        _ = ax.legend()
        _ = ax.set_ylabel("Frequency")

    _ = ax.grid(True)

    return fig


def plot_specific_heating_demand_by_year_and_type(
    fname: str | os.PathLike,
) -> plt.Figure:

    frame = pd.read_csv(
        fname,
        skiprows=[0, 1],
        names=[
            "Detached house",
            "Semi-detached house",
            "Apartment building (3 - 6 dwellings)",
            "Apartment building (7 - 12 dwellings)",
            "Apartment building (13 - 20 dwellings)",
            "Apartment building (> 20 dwellings)",
            "Non-residential buildings with residential units",
        ],
        comment="#",
    )

    fig, ax = plt.subplots(1, 1)
    fig.autofmt_xdate(rotation=45)

    for c in frame.columns:
        if c not in {"Non-residential buildings with residential units"}:
            ax.plot(frame.index, frame[c], marker=".", linestyle="--", label=c)

    _ = ax.legend()
    _ = ax.set_ylabel(r"Specific heating requirement in $\mathrm{kWh / (a \, m^2)}$")

    _ = ax.grid(True)
    _ = ax.set_ylim(30, 170)

    return fig


def plot_energy_cost_savings_across_sample(
    thermalEfficiency: float,
    nominalThermalPowerSI: float,
    nominalPowerSI: float,
    matchingFactor: float,
    fnames: Dict[int, str | os.PathLike],
) -> plt.Figure:

    fig = plt.figure()
    ax = fig.gca()

    for sample_size, fname in fnames.items():
        df = pd.read_csv(fname, comment="#", index_col=0)

        ax.plot(
            df.iloc[:, 0], df.iloc[:, 1], ".--", label=f"N = {len(df)} ({sample_size})"
        )

    ax.set_title(
        rf"$\eta_\mathrm{{th}} = {thermalEfficiency:.2f}$; $\dot{{Q}}_\mathrm{{N}} = {nominalThermalPowerSI/1000:.2f}\,\mathrm{{kW}}$; "
        rf"$P_\mathrm{{N}} = {nominalPowerSI/1000:.2f}\,\mathrm{{kW}}$; $k = {matchingFactor:.2f}$"
    )
    ax.set_xlabel(r"Energy cost savings relative to status quo $Z$")
    ax.set_ylabel(r"Cumulative freq. of $z_{n,2} \leq Z$")
    ax.legend()

    ax.grid(True)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    return fig


def plot_carbon_dioxide_savings_across_sample(
    thermalEfficiency: float,
    nominalThermalPowerSI: float,
    nominalPowerSI: float,
    fnames: Dict[int, str | os.PathLike],
) -> plt.Figure:

    fig = plt.figure()
    ax = fig.gca()

    for sample_size, fname in fnames.items():
        df = pd.read_csv(fname, comment="#", index_col=0)

        ax.plot(
            df.iloc[:, 0], df.iloc[:, 1], ".--", label=f"N = {len(df)} ({sample_size})"
        )

    ax.set_title(
        rf"$\eta_\mathrm{{th}} = {thermalEfficiency:.2f}$; $\dot{{Q}}_\mathrm{{N}} = {nominalThermalPowerSI/1000:.2f}\,\mathrm{{kW}}$; "
        rf"$P_\mathrm{{N}} = {nominalPowerSI/1000:.2f}\,\mathrm{{kW}}$"
    )
    ax.set_xlabel(r"Carbon dioxide emissions reductions relative to status quo $Z$")
    ax.set_ylabel(r"Cumulative freq. of $z_{n,3} \leq Z$")
    ax.legend()

    ax.grid(True)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    return fig
