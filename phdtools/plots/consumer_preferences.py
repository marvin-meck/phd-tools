"""phdtools.plots.consumer_preferences.py

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


def plot_sample_average_choice_probability(
    fname_mxl_csav: str | os.PathLike = None,
    fname_mxl_co2sav: str | os.PathLike = None,
    fname_logit_csav: str | os.PathLike = None,
    fname_logit_co2sav: str | os.PathLike = None,
) -> plt.Figure:

    if not fname_mxl_csav is None:
        mxl_sample_avg_prob_csav = pd.read_csv(fname_mxl_csav, comment="#")

    if not fname_mxl_co2sav is None:
        mxl_sample_avg_prob_co2sav = pd.read_csv(fname_mxl_co2sav, comment="#")

    if not fname_logit_csav is None:
        logit_sample_avg_prob_csav = pd.read_csv(fname_logit_csav, comment="#")

    if not fname_logit_co2sav is None:
        logit_sample_avg_prob_co2sav = pd.read_csv(fname_logit_co2sav, comment="#")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle(
        # "Sample average choice probability according to Rommel and Sagebiel (2017)\n"
        "Feed-in tarrif 0.08€ (FIT = 8); one-time investment (ITYPE = 0); contract duration 5a (DUR = 5)"
    )

    if not fname_mxl_csav is None:
        axs[0].plot(
            mxl_sample_avg_prob_csav["ICOST"],
            mxl_sample_avg_prob_csav["CSAV=10.0"],
            linestyle="none",
            marker="^",
            mec="k",
            mfc="white",
            label="CSAV = 10%",
        )
        axs[0].plot(
            mxl_sample_avg_prob_csav["ICOST"],
            mxl_sample_avg_prob_csav["CSAV=20.0"],
            linestyle="none",
            marker="o",
            mec="k",
            mfc="black",
            label="CSAV = 20%",
        )
        axs[0].plot(
            mxl_sample_avg_prob_csav["ICOST"],
            mxl_sample_avg_prob_csav["CSAV=30.0"],
            linestyle="none",
            marker="v",
            mec="k",
            mfc="white",
            label="CSAV = 30%",
        )

    if not fname_logit_csav is None:
        axs[0].plot(
            mxl_sample_avg_prob_csav["ICOST"],
            logit_sample_avg_prob_csav["CSAV=10.0"],
            linestyle="-",
            color="k",
            label="Logit approximation",
        )
        axs[0].plot(
            mxl_sample_avg_prob_csav["ICOST"],
            logit_sample_avg_prob_csav["CSAV=20.0"],
            linestyle="-",
            color="k",
        )
        axs[0].plot(
            mxl_sample_avg_prob_csav["ICOST"],
            logit_sample_avg_prob_csav["CSAV=30.0"],
            linestyle="-",
            color="k",
        )

    if not fname_mxl_co2sav is None:
        axs[1].plot(
            mxl_sample_avg_prob_co2sav["ICOST"],
            mxl_sample_avg_prob_co2sav["CO2SAV=1.0"],
            linestyle="none",
            marker="^",
            mec="k",
            mfc="white",
            label="CO2SAV = 10%",
        )
        axs[1].plot(
            mxl_sample_avg_prob_co2sav["ICOST"],
            mxl_sample_avg_prob_co2sav["CO2SAV=2.0"],
            linestyle="none",
            marker="o",
            mec="k",
            mfc="black",
            label="CO2SAV = 20%",
        )
        axs[1].plot(
            mxl_sample_avg_prob_co2sav["ICOST"],
            mxl_sample_avg_prob_co2sav["CO2SAV=3.0"],
            linestyle="none",
            marker="v",
            mec="k",
            mfc="white",
            label="CO2SAV = 30%",
        )

    if not fname_logit_co2sav is None:
        axs[1].plot(
            mxl_sample_avg_prob_co2sav["ICOST"],
            logit_sample_avg_prob_co2sav["CO2SAV=1.0"],
            linestyle="-",
            color="k",
            label="Logit approximation",
        )
        axs[1].plot(
            mxl_sample_avg_prob_co2sav["ICOST"],
            logit_sample_avg_prob_co2sav["CO2SAV=2.0"],
            linestyle="-",
            color="k",
        )
        axs[1].plot(
            mxl_sample_avg_prob_co2sav["ICOST"],
            logit_sample_avg_prob_co2sav["CO2SAV=3.0"],
            linestyle="-",
            color="k",
        )

    for ax in axs:
        ax.set_xlabel(r"Investment cost ICOST in k€")
        ax.set_ylabel(r"Sample average probability")
        ax.legend()

        ax.set_ylim(0, 1)
        ax.grid(True)

    return fig
