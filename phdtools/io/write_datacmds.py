"""phdtools.io.write_datacmds.py

Copyright 2023 Technical University Darmstadt

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

from collections import defaultdict
from typing import Dict
import warnings

import pyomo.environ as pyo


def _item_format(item):
    if isinstance(item, int):
        cmd = f"\t{item}"
    elif isinstance(item, str):
        cmd = f'\t"{item}"'
    elif isinstance(item, tuple):
        cmd = "\t({})".format(
            ",".join(f"{sub}" if isinstance(sub, int) else f'"{sub}"' for sub in item)
        )
    return cmd


def _as_set(name, data):
    cmd = ""
    for key in data.keys():
        if key is None:
            cmd += f"set {name} := \n"
            cmd += "{}".format("\n".join(_item_format(item) for item in data[key]))
            # cmd += "{}".format(
            #     "\n".join(
            #         f"\t{item}" if isinstance(item, int) else f'\t"{item}"'
            #         for item in data[key]
            #     )
            # )
        else:
            cmd += f"set {name}[{key}] := \n"
            cmd += "{}".format("\n".join(_item_format(item) for item in data[key]))
            # cmd += "{}".format(
            #     "\n".join(
            #         f"\t{item}" if isinstance(item, int) else f'\t"{item}"'
            #         for item in data[key]
            #     )
            # )
    cmd += "\n;\n"
    return cmd


def _as_param(name, data, parenthensize_tuples=True):
    cmd = f"param {name} := "
    for key in data.keys():
        if key is None:
            cmd += "{}".format(data[key])
        else:
            cmd += "\n"
            if isinstance(key, int):
                cmd += "\t{}\t{}".format(key, data[key])
            elif isinstance(key, str):
                cmd += '\t"{}"\t{}'.format(key, data[key])
            elif isinstance(key, tuple):
                if len(key) == 1:
                    cmd += "\t{}\t{}".format(
                        "\t".join(
                            f"{item}" if isinstance(item, int) else f'"{item}"'
                            for item in key
                        ),
                        data[key],
                    )
                elif len(key) >= 1:
                    if parenthensize_tuples:
                        cmd += "\t({})\t{}".format(
                            ",".join(
                                f"{item}" if isinstance(item, int) else f'"{item}"'
                                for item in key
                            ),
                            data[key],
                        )
                    else:
                        cmd += "\t{}\t{}".format(
                            "\t".join(
                                f"{item}" if isinstance(item, int) else f'"{item}"'
                                for item in key
                            ),
                            data[key],
                        )
                else:
                    raise ValueError("Should be a dead branch!")
            else:
                raise ValueError(f"Unknown index type {type(key)}")

    if not key is None:
        cmd += "\n"
    cmd += ";\n"

    return cmd


def write_data_commands(
    model=None, ostream=None, data_dict: Dict = defaultdict(), parenthensize_tuples=True
):

    for name in model.component_map(pyo.Set):
        if name in data_dict.keys():
            ostream.write(_as_set(name=name, data=data_dict[name]))
            ostream.write("\n")
        else:
            warnings.warn(
                f"Warning: {name} not in `data_dict.keys()`. You can ignore this warning if {name} is an automatically created index set."
            )

    for name in model.component_map(pyo.Param):
        if name in data_dict.keys():
            ostream.write(_as_param(name, data_dict[name], parenthensize_tuples))
            ostream.write("\n")
        else:
            warnings.warn(
                f"Warning: {name} not in `data_dict.keys()`. Skipping... Your datacommand-file may not work with your model."
            )
