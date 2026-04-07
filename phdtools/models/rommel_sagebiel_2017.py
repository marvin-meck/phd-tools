"""phdtools.models.rommel_sagebiel_2017.py

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

Implementation of a Random Parameters Logit (RPL) model with error component
by Rommel and Sagebiel (2017).

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de

References
----------

Rommel, K. and Sagebiel, J. (2017) 'Preferences for micro-cogeneration in
    Germany: Policy implications for grid expansion from a discrete choice
    experiment', Applied Energy, 206, pp. 612–622. Available at:
    https://doi.org/10.1016/j.apenergy.2017.08.216.
"""

from datetime import datetime

import pandas as pd
import numpy as np
import scipy.stats

from phdtools import DATA_DIR
from phdtools.rdm import auto_create_path, write_metadata

TODAY = datetime.today().strftime("%y%m%d")

NUM_AGENTS = 512
SAMPLE_SIZE = 512
SAMPLING_TECHNIQUE = "lhs"

MEAN_AGE = 48.65
MEAN_FLATSIZE = 91.99

DICT_INTERACTIONS = {
    "ICOST": ["HEATSYS"],
    "CO2SAV": ["AGE"],
    "CSAV": ["AGE"],
    "ITYPE": ["AGE"],
    "DUR": ["SEX"],
    "FIT": ["SEX", "FLATSIZE"],
}


def _create_design(dim, sample_size, seed=None):
    if SAMPLING_TECHNIQUE == "uniform":
        rng = np.random.default_rng(seed=seed)
        design = rng.random((sample_size, dim))
    else:
        if SAMPLING_TECHNIQUE == "lhs":
            sampler = scipy.stats.qmc.LatinHypercube(d=dim, seed=seed)
        elif SAMPLING_TECHNIQUE == "sobol":
            sampler = scipy.stats.qmc.Sobol(d=dim, scramble=True, seed=seed)
        elif SAMPLING_TECHNIQUE == "halton":
            sampler = scipy.stats.qmc.Halton(d=dim, scramble=True, seed=seed)

        design = sampler.random(n=sample_size)

    return design


def _create_coefficient_sample(
    sample_size,
    seed=None,
    fname=DATA_DIR
    / "rommel-sagebiel-2017"
    / r"data_230420_RPL_Model_CHP_Table_4_Rommel_Sagebiel.csv",
):

    design = _create_design(dim=7, sample_size=sample_size, seed=seed)

    parameters = pd.read_csv(
        fname,
        index_col=0,
        usecols=[0, 1, 3],
        skiprows=[0, 1, 3, 5, 7, 9, 11, 12, 14, 16, 17, 18, 19],
        names=["parameter", "loc", "scale"],
        engine="python",
    )

    # ERR_COMP: scale parameter is negative. Why? Authors don't comment on it.
    #   "Our specification includes an error component to account for unobserved effects regarding the opt-out choice [31]."
    parameters.loc["ERR_COMP", "scale"] = np.abs(parameters.loc["ERR_COMP", "scale"])

    random_variables = [
        scipy.stats.norm(loc=parameters["loc"][name], scale=parameters["scale"][name])
        for name in parameters.index
    ]

    sample_raw = [
        random_variables[idx].ppf(design[:, idx])
        for idx in range(len(random_variables))
    ]

    idx = list(parameters.index).index("ICOST")

    sample_raw[idx] = -1 * np.exp(sample_raw[idx])
    sample = np.column_stack(sample_raw)

    rand_coefs = pd.DataFrame(
        data=sample,
        columns=parameters.index.to_list(),
        index=pd.RangeIndex(1, sample_size + 1, 1, name="REALIZATION"),
    )

    return rand_coefs


def _create_socio_demographic_sample(
    num_agents,
    seed=None,
):

    # see Table 1:  Summary statistics of socio-demographic and energy usage related variables. (Rommel and Sagebiel, 2017)
    # AGE: Age in years
    # SEX: Female = 0 Male = 1
    # INCOME: scale ranging from 1 = less than 500 Euros to 13 = more than 6000 Euros
    # ENECOST: Monthly Expenditure for electricity 1 = 15–30 Euros 6 = more than 90 Euros
    # FLATSIZE: in m2
    # HEATSYS

    design = _create_design(dim=6, sample_size=num_agents, seed=seed)

    # TODO: get data from a file
    frame = pd.DataFrame(
        data={
            "AGE": {"mean": MEAN_AGE, "std": 17.48, "min": 18, "max": 91},
            "SEX": {"mean": 0.48, "std": 0.5, "min": 0, "max": 1},
            "INCOME": {"mean": 6.76, "std": 3.89, "min": 1, "max": 13},
            "ENECOST": {"mean": 3.58, "std": 1.39, "min": 1, "max": 6},
            "FLATSIZE": {"mean": MEAN_FLATSIZE, "std": 36.50, "min": 12, "max": 230},
            "HEATSYS": {"mean": 0.43, "std": 0.5, "min": 0, "max": 1},
        }
    ).T

    random_variables = [
        (
            scipy.stats.truncnorm(
                a=(frame.loc[name, "min"] - frame.loc[name, "mean"])
                / frame.loc[name, "std"],
                b=(frame.loc[name, "max"] - frame.loc[name, "mean"])
                / frame.loc[name, "std"],
                loc=frame.loc[name, "mean"],
                scale=frame.loc[name, "std"],
            )
            if name in ("AGE", "INCOME", "ENECOST", "FLATSIZE")
            else scipy.stats.bernoulli(p=frame.loc[name, "mean"])
        )
        for name in frame.index
    ]

    sample_raw = [random_variables[idx].ppf(design[:, idx]) for idx in range(6)]

    for idx, name in enumerate(frame.index):
        if name in ("AGE", "INCOME", "ENECOST", "FLATSIZE"):
            # standardize
            z = (sample_raw[idx] - np.mean(sample_raw[idx])) / np.std(sample_raw[idx])
            # rescale
            s = frame.loc[name, "mean"] + z * frame.loc[name, "std"]
            # clip
            sample_raw[idx] = np.clip(s, frame.loc[name, "min"], frame.loc[name, "max"])

    sample = np.column_stack(sample_raw)

    socio_demographic_attributes = pd.DataFrame(
        data=sample,
        columns=["AGE", "SEX", "INCOME", "ENECOST", "FLATSIZE", "HEATSYS"],
        index=pd.RangeIndex(1, num_agents + 1, 1, name="AGENT"),
    )

    # "All interactions except the two binary variables sex and heating system were demeaned,
    #   so that the main effects can be interpreted for the sample mean of these variables."
    socio_demographic_attributes["AGE"] = socio_demographic_attributes["AGE"] - MEAN_AGE
    socio_demographic_attributes["FLATSIZE"] = (
        socio_demographic_attributes["FLATSIZE"] - MEAN_FLATSIZE
    )

    return socio_demographic_attributes


def wtp(attr, fname_a, fname_b, cond=None, heatsys=0):

    a = pd.read_csv(
        fname_a,  # support_id.get_path(fail_exists=False) / f"250718_deterministic_coefficients.csv"
        comment="#",
        index_col=0,
    ).iloc[:, 0]

    b = pd.read_csv(
        fname_b,  # support_id.get_path(fail_exists=False) / f"250718_random_coefficients.csv",
        comment="#",
        index_col=0,
    )

    if attr in DICT_INTERACTIONS.keys():
        if cond is None:
            fmt = "{} has interaction(s) but no socio-demographic values were given.\n\tSet cond={}"
            raise ValueError(
                fmt.format(
                    attr,
                    "{"
                    + ",".join(
                        f"{key}:'value {num+1}'"
                        for num, key in enumerate(DICT_INTERACTIONS[attr])
                    ),
                )
                + "}"
            )

    num = lambda key: (
        b[key]
        + sum(cond[inter] * a[f"{key} x {inter}"] for inter in DICT_INTERACTIONS[key])
        if key != "ASC"
        else a["ASC"]
    )

    den = b["ICOST"] + heatsys * a["ICOST x HEATSYS"]

    return 1e3 * (num(attr) / den)


def median_wtp(attr, fname_a, fname_b, cond=None, heatsys=0):

    a = pd.read_csv(
        fname_a,  # support_id.get_path(fail_exists=False) / f"250718_deterministic_coefficients.csv",
        comment="#",
        index_col=0,
    ).iloc[:, 0]

    b = pd.read_csv(
        fname_b,  # support_id.get_path(fail_exists=False) / f"250718_random_coefficients.csv",
        comment="#",
        index_col=0,
    )

    if attr in DICT_INTERACTIONS.keys():
        if cond is None:
            fmt = "{} has interaction(s) but no socio-demographic values were given.\n\tSet cond={}"
            raise ValueError(
                fmt.format(
                    attr,
                    "{"
                    + ",".join(
                        f"{key}:'value {num+1}'"
                        for num, key in enumerate(DICT_INTERACTIONS[attr])
                    ),
                )
                + "}"
            )

    num = lambda key: (
        b[key].median()
        + sum(cond[inter] * a[f"{key} x {inter}"] for inter in DICT_INTERACTIONS[key])
        if key != "ASC"
        else a["ASC"]
    )

    den = b["ICOST"].median() + heatsys * a["ICOST x HEATSYS"]

    return 1e3 * num(attr) / den


def sample_average_wtp(attr, fname_a, fname_b, fname_s, wtp_func=median_wtp):

    s = pd.read_csv(
        fname_s,  # support_id.get_path(fail_exists=False) / f"250718_socio_demographic_attributes.csv",
        comment="#",
        index_col=0,
    )

    if attr in DICT_INTERACTIONS.keys():
        cond = {key: s[key] for key in DICT_INTERACTIONS[attr]}
    else:
        cond = None

    wtp = wtp_func(
        attr, fname_a=fname_a, fname_b=fname_b, cond=cond, heatsys=s["HEATSYS"]
    )

    return wtp.mean()


# def delta_method_ci(num, den):

#     if np.isscalar(num):
#         mean_num = num
#     else:
#         mean_num = np.mean(num,axis=0)

#     mean_den = np.mean(den,axis=0)

#     wtp_mean = mean_num / mean_den

#     print(wtp_mean)
#     if np.isscalar(num):
#         num = num * np.ones(den.shape)

#     cov = np.cov(num, den, rowvar=False)
#     grad = np.array([1 / mean_den, -mean_num / mean_den**2])

#     # print(grad.shape, cov.shape)
#     var = grad @ cov @ grad.T
#     se = np.sqrt(var)

#     return wtp_mean - 1.96 * se, wtp_mean + 1.96 * se


def recreate_table5_rommel2017(ostream, fname_a, fname_b, fname_s, wtp_func=median_wtp):

    header = [
        "Attribute",
        "Socio-demographics",
        "WTP(HEATSYS = 0)",
        "WTP(HEATSYS = 1)",
        "Average WTP",
    ]

    col_widths = [17, 18, 18, 18, 18]

    def make_rule():
        return "" + "-+-".join(["-" * w for w in col_widths]) + "\n"

    fmt = "" + " | ".join([f"{{:{w}}}" for w in col_widths]) + "\n"

    ostream.write(make_rule())
    ostream.write(fmt.format(*header))

    rows = [
        [
            "ASC",
            "ASC",
            wtp_func("ASC", fname_a=fname_a, fname_b=fname_b, heatsys=0),
            wtp_func("ASC", fname_a=fname_a, fname_b=fname_b, heatsys=1),
            sample_average_wtp(
                "ASC",
                fname_a=fname_a,
                fname_b=fname_b,
                fname_s=fname_s,
                wtp_func=wtp_func,
            ),
        ],
        [
            "Investment type",
            "ITYPE60y",
            wtp_func(
                "ITYPE",
                cond={"AGE": 60 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "ITYPE",
                cond={"AGE": 60 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            sample_average_wtp(
                "ITYPE",
                fname_a=fname_a,
                fname_b=fname_b,
                fname_s=fname_s,
                wtp_func=wtp_func,
            ),
        ],
        [
            "Investment type",
            "ITYPE25y",
            wtp_func(
                "ITYPE",
                cond={"AGE": 25 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "ITYPE",
                cond={"AGE": 25 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            np.nan,
        ],
        [
            "Contract duration",
            "DURFemale",
            wtp_func(
                "DUR", cond={"SEX": 0}, fname_a=fname_a, fname_b=fname_b, heatsys=0
            ),
            wtp_func(
                "DUR", cond={"SEX": 0}, fname_a=fname_a, fname_b=fname_b, heatsys=1
            ),
            sample_average_wtp(
                "DUR",
                fname_a=fname_a,
                fname_b=fname_b,
                fname_s=fname_s,
                wtp_func=wtp_func,
            ),
        ],
        [
            "Contract duration",
            "DURMale",
            wtp_func(
                "DUR", cond={"SEX": 1}, fname_a=fname_a, fname_b=fname_b, heatsys=0
            ),
            wtp_func(
                "DUR", cond={"SEX": 1}, fname_a=fname_a, fname_b=fname_b, heatsys=1
            ),
            np.nan,
        ],
        [
            "CO2 savings",
            "CO2SAV60y",
            wtp_func(
                "CO2SAV",
                cond={"AGE": 60 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "CO2SAV",
                cond={"AGE": 60 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            sample_average_wtp(
                "CO2SAV",
                fname_a=fname_a,
                fname_b=fname_b,
                fname_s=fname_s,
                wtp_func=wtp_func,
            ),
        ],
        [
            "CO2 savings",
            "CO2SAV25y",
            wtp_func(
                "CO2SAV",
                cond={"AGE": 25 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "CO2SAV",
                cond={"AGE": 25 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            np.nan,
        ],
        [
            "Cost savings",
            "CSAV60y",
            wtp_func(
                "CSAV",
                cond={"AGE": 60 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "CSAV",
                cond={"AGE": 60 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            sample_average_wtp(
                "CSAV",
                fname_a=fname_a,
                fname_b=fname_b,
                fname_s=fname_s,
                wtp_func=wtp_func,
            ),
        ],
        [
            "Cost savings",
            "CSAV25y",
            wtp_func(
                "CSAV",
                cond={"AGE": 25 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "CSAV",
                cond={"AGE": 25 - MEAN_AGE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            np.nan,
        ],
        [
            "Feed-in tariff",
            "FITFemale110",
            wtp_func(
                "FIT",
                cond={"SEX": 0, "FLATSIZE": 110 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "FIT",
                cond={"SEX": 0, "FLATSIZE": 110 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            sample_average_wtp(
                "FIT",
                fname_a=fname_a,
                fname_b=fname_b,
                fname_s=fname_s,
                wtp_func=wtp_func,
            ),
        ],
        [
            "Feed-in tariff",
            "FITMale110",
            wtp_func(
                "FIT",
                cond={"SEX": 1, "FLATSIZE": 110 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "FIT",
                cond={"SEX": 1, "FLATSIZE": 110 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            np.nan,
        ],
        [
            "Feed-in tariff",
            "FITFemale50",
            wtp_func(
                "FIT",
                cond={"SEX": 0, "FLATSIZE": 50 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "FIT",
                cond={"SEX": 0, "FLATSIZE": 50 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            np.nan,
        ],
        [
            "Feed-in tariff",
            "FITMale50",
            wtp_func(
                "FIT",
                cond={"SEX": 1, "FLATSIZE": 50 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=0,
            ),
            wtp_func(
                "FIT",
                cond={"SEX": 1, "FLATSIZE": 50 - MEAN_FLATSIZE},
                fname_a=fname_a,
                fname_b=fname_b,
                heatsys=1,
            ),
            np.nan,
        ],
    ]

    fmt1 = "{:17s} | {:18s} | {:>18.1f} | {:>18.1f} | {:>18.1f}\n"

    prev = None
    for row in rows:
        if row[0] != prev:
            ostream.write(make_rule())
        ostream.write((fmt1.format(row[0] if row[0] != prev else "", *row[1:])))
        prev = row[0]


# def compute_interactions(x, s):
#     frame = pd.DataFrame(
#         columns=[f"{key} x {val}" for key in DICT_INTERACTIONS.keys() for val in DICT_INTERACTIONS[key]]
#     )

#     frame["ICOST x HEATSYS"] = x["ICOST"] * s["HEATSYS"]
#     frame["CO2SAV x AGE"] = x["CO2SAV"] * s["AGE"]
#     frame["CSAV x AGE"] = x["CSAV"] * s["AGE"]

#     frame["ITYPE x AGE"] = x["ITYPE"] * s["AGE"]
#     frame["DUR x SEX"] = x["DUR"] * s["SEX"]
#     frame["FIT x SEX"] = x["FIT"] * s["SEX"]
#     frame["FIT x FLATSIZE"] = x["FIT"] * s["FLATSIZE"]

#     return frame


def compute_interactions_old(attributes_alternative, fname_s):

    s = pd.read_csv(
        fname_s,  # support_id.get_path(fail_exists=False) / f"250718_socio_demographic_attributes.csv",
        comment="#",
        index_col=0,
    )

    index = pd.MultiIndex.from_product(
        [s.index, attributes_alternative.index], names=["AGENT", "ALTERNATIVE"]
    )

    frame = pd.DataFrame(
        index=index,
        columns=[
            f"{key} x {val}"
            for key in DICT_INTERACTIONS.keys()
            for val in DICT_INTERACTIONS[key]
        ],
    )

    for alt in DICT_INTERACTIONS.keys():
        for soc in DICT_INTERACTIONS[alt]:
            frame[f"{alt} x {soc}"] = attributes_alternative[alt].reindex(
                index, level="ALTERNATIVE"
            ) * s[soc].reindex(index, level="AGENT")

    return frame


def compute_interactions(attributes_alternative, fname_s):

    s = pd.read_csv(
        fname_s,
        comment="#",
        index_col=0,
    )

    index = attributes_alternative.index

    frame = pd.DataFrame(
        index=index,
        columns=[
            f"{key} x {val}"
            for key in DICT_INTERACTIONS.keys()
            for val in DICT_INTERACTIONS[key]
        ],
    )

    for alt in DICT_INTERACTIONS.keys():
        for soc in DICT_INTERACTIONS[alt]:
            frame[f"{alt} x {soc}"] = attributes_alternative[alt].mul(
                s[soc], level="AGENT"
            )

    return frame


# def mxl_prob(x, z, a, b):

#     v_chp = pd.DataFrame(
#         np.add.outer(
#             b.drop("ERR_COMP", axis=1).dot(x).values,
#             z.dot(a.drop("ASC")).values,
#         ),
#         index=b.index,
#         columns=z.index,
#     )

#     v_status_quo = a["ASC"] + b["ERR_COMP"]

#     exp_chp = np.exp(v_chp.values)
#     exp_status_quo = np.exp(v_status_quo.values)

#     num = exp_chp
#     den = exp_chp + exp_status_quo[:, np.newaxis]

#     Pr = pd.DataFrame(num / den, index=v_chp.index, columns=v_chp.columns)

#     return Pr.mean(axis=0)


def mxl_prob_old(x, z, a, b):
    # random part: REALIZATION, ALTERNATIVE
    random_part = b.drop("ERR_COMP", axis=1).dot(x.T).stack()
    random_part.name = "b^T x"
    random_part = random_part.reset_index()

    # fixed part: (AGENT, ALTERNATIVE)
    fixed_part = z.dot(a.drop("ASC"))
    fixed_part.name = "a^T z"
    fixed_part = fixed_part.reset_index()

    df = pd.merge(random_part, fixed_part, on="ALTERNATIVE", how="left")
    df["V_CHP"] = df["b^T x"] + df["a^T z"]

    df = df.merge(b["ERR_COMP"].reset_index(), on="REALIZATION", how="left")
    df["ASC"] = a["ASC"]
    df["V_SQ"] = df["ASC"] + df["ERR_COMP"]

    df = df.set_index(["AGENT", "ALTERNATIVE", "REALIZATION"])

    exp_chp = np.exp(df["V_CHP"])
    exp_status_quo = np.exp(df["V_SQ"])

    num = exp_chp
    den = exp_chp + exp_status_quo

    logit_prob = (
        pd.Series(num / den, index=df.index, name="Pr")
        .reset_index()
        .pivot(columns="REALIZATION", values="Pr", index=["AGENT", "ALTERNATIVE"])
    )
    Pr = logit_prob.mean(axis=1)
    Pr.name = "Pr"

    return Pr.reset_index().pivot(index="AGENT", columns="ALTERNATIVE", values="Pr")


def mxl_prob(attributes_alternative, interactions, fname_a, fname_b):
    """ """
    a = pd.read_csv(
        fname_a,  # support_id.get_path(fail_exists=False) / f"250718_deterministic_coefficients.csv"
        comment="#",
        index_col=0,
    ).iloc[:, 0]

    b = pd.read_csv(
        fname_b,  # support_id.get_path(fail_exists=False) / f"250718_random_coefficients.csv",
        comment="#",
        index_col=0,
    )

    b.loc[:, a.index] = a.values

    # X = pd.merge(attributes_alternative.reset_index(), interactions.reset_index(), on="ALTERNATIVE", how="right").set_index(["AGENT","ALTERNATIVE"])
    X = pd.concat([attributes_alternative, interactions], axis=1)
    X["ASC"] = -1
    X["ERR_COMP"] = -1

    t = X.dot(b.T)

    Pr = (1 / (1 + np.exp(-t))).mean(axis=1)
    Pr.name = "Pr"

    return Pr.reset_index().pivot(index="AGENT", columns="ALTERNATIVE", values="Pr")


def compute_sample_avg_mxl_prob(attributes_alternative, fname_a, fname_b, fname_s):

    interactions = compute_interactions(attributes_alternative, fname_s)

    sample_average_prob = mxl_prob(
        attributes_alternative, interactions, fname_a, fname_b
    )

    avg_prob = sample_average_prob.mean()
    avg_prob.name = "SAMPLE_AVG_MXL_PROB"

    return avg_prob


def logit_prob(attributes_alternative, interactions, coefs):

    X_train = pd.concat([attributes_alternative, interactions], axis=1)
    X_train["ASC"] = -1

    Pr = 1 / (1 + np.exp(-1 * X_train.dot(coefs)))
    Pr.name = "Pr"

    return Pr.reset_index().pivot(index="AGENT", columns="ALTERNATIVE", values="Pr")


def compute_sample_avg_logit_prob(attributes_alternative, fname_s, fname_c):

    interactions = compute_interactions(attributes_alternative, fname_s)

    logit_coefs = pd.read_json(fname_c.as_posix(), typ="series")

    sample_average_prob = logit_prob(
        attributes_alternative,
        interactions,
        logit_coefs,
    )

    avg_prob = sample_average_prob.mean()
    avg_prob.name = "SAMPLE_AVG_LOGIT_PROB"

    return avg_prob


@auto_create_path
def create_socio_demographic_samples(
    path, seed=None, sample_sizes=[16, 32, 64, 128, 256, 512, 1024]
):

    for sample_size in sample_sizes:
        socio_demographic_attributes = _create_socio_demographic_sample(
            num_agents=sample_size, seed=seed
        )

        description = (
            "Synthetic sample of socio-demographic and energy usage related variables\n"
            "created to approximate the empirical sample used by Rommel and Sagebiel (2017)\n"
            "based on summary statstics reported in Rommel and Sagebiel (2017, table 1).\n"
            "\n"
            f"Created using scipy.stats.truncnorm and scipy.stats.bernoulli (scipy {scipy.__version__}):\n"
            f"sampling technique={SAMPLING_TECHNIQUE}; sample size={sample_size}\n"
            "\n"
            "References:\n"
            "-----------\n"
            "Rommel, K. and Sagebiel, J. (2017) 'Preferences for micro-cogeneration in\n"
            "    Germany: Policy implications for grid expansion from a discrete choice\n"
            "    experiment', Applied Energy, 206, pp. 612–622. Available at:\n"
            "    https://doi.org/10.1016/j.apenergy.2017.08.216. "
        )

        with open(
            path / f"{TODAY}_socio_demographic_attributes_{sample_size}.csv", "w+"
        ) as f:
            write_metadata(f, description=description)
            socio_demographic_attributes.to_csv(f)


@auto_create_path
def create_coefficient_samples(
    path, seed=None, sample_sizes=[16, 32, 64, 128, 256, 512, 1024]
):

    deterministic_coefficients = (
        pd.read_csv(
            DATA_DIR
            / "rommel-sagebiel-2017"
            / r"data_230420_RPL_Model_CHP_Table_4_Rommel_Sagebiel.csv",
            index_col=0,
            usecols=[0, 1],
            skiprows=[0, 2, 4, 6, 8, 10, 13, 15, 16, 17, 18, 19],
            names=["coefficient", "value"],
            engine="python",
        )
        .iloc[:, 0]
        .rename("deterministic coefficients")
    )

    description = (
        "Deterministic coefficients used in the mixed logit model by \n"
        "Rommel and Sagebiel (2017) based Rommel and Sagebiel (2017, table 4).\n"
        "\n"
        "References:\n"
        "-----------\n"
        "Rommel, K. and Sagebiel, J. (2017) 'Preferences for micro-cogeneration in\n"
        "    Germany: Policy implications for grid expansion from a discrete choice\n"
        "    experiment', Applied Energy, 206, pp. 612–622. Available at:\n"
        "    https://doi.org/10.1016/j.apenergy.2017.08.216.\n"
    )

    with open(path / f"{TODAY}_deterministic_coefficients.csv", "w+") as f:
        write_metadata(f, description)
        deterministic_coefficients.to_csv(f)

    for sample_size in sample_sizes:
        random_coefficients = _create_coefficient_sample(
            sample_size=sample_size, seed=seed
        )

        description = (
            "Realizations for the random coefficients used in the mixed logit model by \n"
            "Rommel and Sagebiel (2017) based Rommel and Sagebiel (2017, table 4).\n"
            "\n"
            f"Created using scipy.stats.norm (scipy {scipy.__version__}):\n"
            f"sampling technique={SAMPLING_TECHNIQUE}; sample size={sample_size}\n"
            "\n"
            "References:\n"
            "-----------\n"
            "Rommel, K. and Sagebiel, J. (2017) 'Preferences for micro-cogeneration in\n"
            "    Germany: Policy implications for grid expansion from a discrete choice\n"
            "    experiment', Applied Energy, 206, pp. 612–622. Available at:\n"
            "    https://doi.org/10.1016/j.apenergy.2017.08.216.\n"
        )

        with open(path / f"{TODAY}_random_coefficients_{sample_size}.csv", "w+") as f:
            write_metadata(f, description)
            random_coefficients.to_csv(f)


if __name__ == "__main__":
    pass
