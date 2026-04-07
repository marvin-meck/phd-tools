#!/usr/bin/env -S python3 -W ignore 
"""scripts/cep_case_builder.py

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

from argparse import ArgumentParser
from itertools import product
import logging
import os
from pathlib import Path
import re
import sqlite3

# from typing import Dict, Set, LiteralString
import uuid

import numpy as np
import pandas as pd
import yaml

from phdtools.models.white_dantzig_1958 import pyomo_create_model
from phdtools.io.write_datacmds import write_data_commands

logger = logging.getLogger(__name__)

template = """
SELECT * FROM [std_gibbs_free_energy]
WHERE "T(K)" BETWEEN 500 AND 2000
AND "FORMULA" IN ({})
;
"""


def get_gibbs(dbfile: Path, set_compounds: set, temperature_kelvin: float):
    query = template.format(", ".join(f'"{c}"' for c in set_compounds))

    with sqlite3.connect(dbfile) as con:
        df = pd.read_sql(query, con)

    df = df.pivot(index="T(K)", columns="FORMULA", values="G")

    if not temperature_kelvin in df.index:
        df.loc[temperature_kelvin, :] = np.nan
        df = df.sort_index()

    std_gibbs_free_energy = (
        df.interpolate(method="index").loc[temperature_kelvin].to_dict()
    )
    return std_gibbs_free_energy


def create_data_dict(
    set_compounds: set,
    temperature_kelvin: float,
    pressure_atm: float,
    amount_element: dict,
    dbfile: Path,
):
    pattern = re.compile(r"[A-Z][a-z]?|\d+")

    set_elements = set()
    number_of_atoms = dict()

    for compound in set_compounds:
        res = re.findall(pattern, compound)
        number_of_atoms[compound] = {
            res[idx]: res[idx + 1] for idx in range(0, len(res), 2)
        }

    for vals in number_of_atoms.values():
        for key in vals.keys():
            set_elements.add(key)

    std_gibbs_free_energy = get_gibbs(dbfile, set_compounds, temperature_kelvin)

    data_dict = {
        "SetCompounds": {None: list(set_compounds)},
        "SetElements": {None: list(set_elements)},
        "number_of_atoms": {
            (j, i): k
            for i in number_of_atoms.keys()
            for j, k in number_of_atoms[i].items()
        },  # defaults to 0
        "temperature": {None: temperature_kelvin},
        "pressure": {None: pressure_atm},
        "amount_element": amount_element,
        "std_gibbs_free_energy": std_gibbs_free_energy,
    }

    return data_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--log",
        dest="LOG_LEVEL",
        default="warning",
        choices=["info", "debug", "warning"],
        help="sets log level",
    )
    parser.add_argument("--log-file", dest="LOG_FILE", default=None, help="log file")
    parser.add_argument(
        "--data-base", dest="DBFILE", required=True, help="NIST thermochemical tables"
    )
    parser.add_argument(
        "-o", "--out-dir", dest="OUT_DIR", default="tmp", help="output directory"
    )
    parser.add_argument("CONFIG_FILE")

    args = parser.parse_args()

    loglevel = getattr(logging, args.LOG_LEVEL.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError(f"Invalid log level: {args.LOG_LEVEL}")
    logging.basicConfig(level=loglevel)

    if not args.LOG_FILE is None:
        logging.basicConfig(filename=args.LOG_FILE, encoding="utf-8")

    with open(args.CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    list_compounds = config["compounds"]
    list_compounds.sort()
    set_compounds = set(list_compounds)
    logger.debug("compounds: {}".format(",".join(set_compounds)))

    param_ranges = {"temperature": None, "pressure": None}
    for param in param_ranges.keys():
        if isinstance(config[param], dict):
            vals = np.arange(
                config[param]["start"],
                config[param]["stop"],
                config[param]["step"],
            )
        elif isinstance(config[param], list):
            vals = np.array(config[param])
        elif isinstance(config[param], (int, float)):
            vals = np.array([config[param]])
        else:
            raise ValueError("invalid parameter: {}".format(config[param]))

        logger.debug("{}: {}".format(param, vals))

        param_ranges[param] = vals

    model = pyomo_create_model()

    if isinstance(config["amount_element"], list):
        param_ranges["amount_element"] = np.arange(len(config["amount_element"]))
    elif isinstance(config["amount_element"], dict):
        config["amount_element"] = [config["amount_element"]]
        param_ranges["amount_element"] = np.arange(1)

    logger.debug("amount_element: {}".format(config["amount_element"]))

    gen = product(
        param_ranges["temperature"],
        param_ranges["pressure"],
        param_ranges["amount_element"],
    )

    if not os.path.exists(args.OUT_DIR):
        os.makedirs(args.OUT_DIR, exist_ok=False)
    else:
        if os.listdir(args.OUT_DIR):
            raise IOError("Directory {} is not empty".format(args.OUT_DIR))

    for temperature, pressure, idx_amount_element in gen:
        amount_element = config["amount_element"][idx_amount_element]

        _uid = uuid.uuid1().hex
        datacmd_file = os.path.join(args.OUT_DIR, f"{_uid}.dat")

        logger.info("writing {}".format(datacmd_file))
        data_dict = create_data_dict(
            set_compounds, temperature, pressure, amount_element, args.DBFILE
        )
        logger.debug(data_dict)
        with open(datacmd_file, "w+") as f:
            write_data_commands(
                model=model, ostream=f, data_dict=data_dict, parenthensize_tuples=False
            )
