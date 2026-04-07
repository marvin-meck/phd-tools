"""phdtools.plots.cost_modelling.py

Copyright 2026 Marvin Meck

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
import pandas as pd

from phdtools import DATA_DIR


def plot_heat_exchanger_cost(
    fname_u_tube: str | os.PathLike = DATA_DIR
    / "peters-2004"
    / "260210_fig_14_17_u_tube_hex.csv",
    fname_fixed_tube_sheet: str | os.PathLike = DATA_DIR
    / "peters-2004"
    / "260210_fig_14_18_fixed_tube_sheet_hex.csv",
) -> plt.Figure:

    fig = plt.figure()
    ax = fig.gca()

    u_tube = pd.read_csv(fname_u_tube)

    x = u_tube["Surface area (m2)"]
    y1 = u_tube["Carbon steel"]
    y2 = u_tube["Stainless steel"]

    ax.plot(x, y1, "o--", color="black", label="U-tube; Carbon steel")
    ax.plot(x, y2, "v--", color="black", label="U-tube; Stainless steel")

    fixed_tube_sheet = pd.read_csv(fname_fixed_tube_sheet)

    x = fixed_tube_sheet["Surface area (m2)"]
    y1 = fixed_tube_sheet["Carbon-steel tubes"]
    y2 = fixed_tube_sheet["304 Stainless-steel tubes"]
    y3 = fixed_tube_sheet["316 Stainless-steel tubes"]

    ax.plot(x, y1, "^--", color="black", label="Fixed-tube-sheet; carbon-steel tubes")
    ax.plot(
        x, y2, "s--", color="black", label="Fixed-tube-sheet; 304 Stainless-steel tubes"
    )
    ax.plot(
        x, y3, "d--", color="black", label="Fixed-tube-sheet; 316 Stainless-steel tubes"
    )

    ax.set_xlabel(r"Surface area $S$ in $\mathrm{m}^2$")
    ax.set_ylabel(r"Purchased cost $c$ in $\$$")

    ax.set_xlim(1e0, 1e3)
    ax.set_ylim(1e3, 1e6)
    ax.legend(frameon=True)

    ax.loglog()
    ax.set_aspect("equal")
    ax.grid()

    return fig


def plot_direct_fired_heater_costs(
    fname: str | os.PathLike = DATA_DIR
    / "peters-2004"
    / "260210_fig_14_38_direct_fired_heaters.csv",
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.gca()

    heater = pd.read_csv(fname)

    x = heater["Heat duty (kW)"]
    y1 = heater["Carbon-steel tubes (690 kPa)"]
    y2 = heater["Carbon-steel tubes (3450 kPa)"]
    y3 = heater["Stainless-steel tubes (10340 kPA)"]
    y4 = heater["Chrome/moly tubes (6895 kPa)"]

    ax.plot(x, y1, "o--", color="black", label="Carbon-steel tubes (690 kPa)")
    ax.plot(x, y2, "v--", color="black", label="Carbon-steel tubes (3450 kPa)")
    ax.plot(x, y3, "^--", color="black", label="Stainless-steel tubes (10340 kPa)")
    ax.plot(x, y4, "d--", color="black", label="Chrome/moly tubes (6895 kPa)")

    ax.set_xlabel(r"Heat duty $\dot{Q}$ in $\mathrm{kW}$")
    ax.set_ylabel(r"Purchased $c$ cost in $\$$")

    ax.set_xlim(1e2, 1e4)
    ax.set_ylim(1e4, 1e6)
    ax.legend(frameon=True)

    ax.loglog()
    ax.set_aspect("equal")
    ax.grid()

    return fig


def plot_cost_model(
    fname_data: str | os.PathLike,
    fname_model: str | os.PathLike,
    x_name_data: str,
    y_name_data: str,
    title: str | None = None,
    label: str | None = None,
):

    fig = plt.figure()
    ax = fig.gca()

    if label is None:
        label = y_name_data

    data = pd.read_csv(fname_data)
    model = pd.read_csv(
        fname_model, comment="#", names=["Surface area", "Model A", "Model B"]
    )

    _ = ax.plot(data[x_name_data], data[y_name_data], ".", color="black", label=label)
    _ = ax.plot(
        model.iloc[:, 0],
        model.iloc[:, 1],
        "-",
        color="gray",
        label=r"$c/c_0 = (S/S_0)^k$",
    )
    _ = ax.plot(
        model.iloc[:, 0],
        model.iloc[:, 2],
        "-",
        color="black",
        label=r"$c/c_0 = a_1 + a_2 (S/S_0)^k$",
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel(r"Surface area $S$ in $\mathrm{m}^2$")
    ax.set_ylabel(r"Purchased cost $c$ in $\$$")

    ax.set_xlim(1e0, 1e3)
    ax.set_ylim(1e3, 1e6)
    ax.legend(frameon=True)

    ax.loglog()
    ax.set_aspect("equal")
    ax.grid()

    return fig
