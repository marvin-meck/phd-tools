"""phdtools.optimization.pyomo.__init__.py

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

from enum import Enum

import pyomo.environ as pyo
from pyomo.core.expr.visitor import polynomial_degree

from phdtools.optimization.pyomo._reformer_block import reformer_block_rule
from phdtools.optimization.pyomo._shift_block import shift_block_rule
from phdtools.optimization.pyomo._fuel_cell_block import fuel_cell_block_rule

from phdtools.optimization.pyomo._reformer_warmstart import warmstart_reformer
from phdtools.optimization.pyomo._shift_warmstart import warmstart_shift

from phdtools.optimization.pyomo._base_model import BaseModel
from phdtools.optimization.pyomo._choice_model import add_consumer_preference_model

SteamReformingCompounds = Enum(
    "Compound", ["C1H4(g)", "C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"], start=0
)


def pyomo_print_constraint_residuals(block):
    for c in block.component_data_objects(pyo.Constraint, active=True):
        if c.body is None:
            continue

        val = abs(pyo.value(c.body))
        print(c.name, val)


def create_pyomo_problem_statistics(model):
    stats = {
        "variables": {"binary": 0, "integer": 0, "continuous": 0, "total": 0},
        "constraints": {
            "linear": 0,
            "quadratic": 0,
            "polynomial": 0,
            "non-polynomial nonlinear": 0,
            "total": 0,
        },
    }

    # Variable statistics
    for v in model.component_data_objects(pyo.Var, active=True):
        if v.is_binary():
            stats["variables"]["binary"] += 1
        elif v.is_integer():
            stats["variables"]["integer"] += 1
        else:
            stats["variables"]["continuous"] += 1

        stats["variables"]["total"] += 1

    # Constraint statistics
    for c in model.component_data_objects(pyo.Constraint, active=True):
        deg = polynomial_degree(c.body)
        if deg is None:
            stats["constraints"]["non-polynomial nonlinear"] += 1
        elif deg == 1:
            stats["constraints"]["linear"] += 1
        elif deg == 2:
            stats["constraints"]["quadratic"] += 1
        elif deg > 2:
            stats["constraints"]["polynomial"] += 1

        stats["constraints"]["total"] += 1

    return stats
