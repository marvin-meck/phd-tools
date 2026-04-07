"""phdtools.optimization.pyomo._fuel_cell_block.py

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

import pyomo.environ as pyo
from phdtools.models.meck_2025 import get_fuelCellParameters
from phdtools.optimization import FUEL_CELL_TEMPERATURE_SI, FUEL_CELL_PRESSURE_SI
from phdtools.optimization.postprocessing import get_fuelCellEfficiency

PARAMS = get_fuelCellParameters(
    temperatureKelvin=FUEL_CELL_TEMPERATURE_SI, pressureBar=FUEL_CELL_PRESSURE_SI * 1e-5
)


def fuel_cell_block_rule(block):

    # Parameter declarations
    block.GAS_CONST_SI = pyo.Param()
    block.FARADAY_CONST_SI = pyo.Param()

    block.temperatureKelvin = pyo.Param()
    block.reversibleCellPotentialSI = pyo.Param()
    block.exchangeCurrentDensityCathodeSI = pyo.Param()
    block.transferCoefficientCathode = pyo.Param()
    block.areaSpecificResistanceSI = pyo.Param()
    block.limitingCurrentDensitySI = pyo.Param()

    block.currentDensityUpperBoundSI = pyo.Param()
    block.powerDensityUpperBoundSI = pyo.Param()

    block.totalActiveAreaLowerBoundSI = pyo.Param()
    block.totalActiveAreaUpperBoundSI = pyo.Param()

    # Variable declarations
    block.currentDensityScaled = pyo.Var(bounds=(0, 1))

    @block.Expression()
    def currentDensitySI(block):
        return block.currentDensityUpperBoundSI * block.currentDensityScaled

    block.cellPotentialScaled = pyo.Var(bounds=(0, 1))

    @block.Expression()
    def cellPotentialSI(block):
        return block.reversibleCellPotentialSI * block.cellPotentialScaled

    block.powerDensityScaled = pyo.Var(
        bounds=(0, 1),
    )

    @block.Expression()
    def powerDensitySI(block):
        return block.powerDensityUpperBoundSI * block.powerDensityScaled

    block.totalActiveAreaScaled = pyo.Var(
        bounds=(
            pyo.value(
                block.totalActiveAreaLowerBoundSI / block.totalActiveAreaUpperBoundSI
            ),
            1.0,
        ),
    )

    @block.Expression()
    def totalActiveAreaSI(block):
        return block.totalActiveAreaUpperBoundSI * block.totalActiveAreaScaled

    block.totalChargeTransferRateScaled = pyo.Var(bounds=(0, 1))

    @block.Expression()
    def totalChargeTransferRateSI(block):
        return (
            block.totalChargeTransferRateScaled
            * block.currentDensityUpperBoundSI
            * block.totalActiveAreaUpperBoundSI
        )

    @block.Expression()
    def activationOverpotentialScaled(block):
        E0 = block.reversibleCellPotentialSI
        a = block.transferCoefficientCathode
        R = block.GAS_CONST_SI
        F = block.FARADAY_CONST_SI
        T = block.temperatureKelvin

        return (
            -1
            / E0
            * 1
            / a
            * R
            * T
            / F
            * pyo.log(
                block.currentDensityUpperBoundSI
                / block.exchangeCurrentDensityCathodeSI
                * block.currentDensityScaled
            )
        )

    @block.Expression()
    def concentrationOverpotentialScaled(block):
        E0 = block.reversibleCellPotentialSI
        R = block.GAS_CONST_SI
        F = block.FARADAY_CONST_SI
        T = block.temperatureKelvin

        a = block.transferCoefficientCathode

        return (
            1
            / E0
            * R
            * T
            / F
            * (1 / 4 + 1 / a)
            * pyo.log(
                1
                - block.currentDensityUpperBoundSI
                / block.limitingCurrentDensitySI
                * block.currentDensityScaled
            )
        )

    @block.Expression()
    def ohmicLossesScaled(block):
        return (
            block.areaSpecificResistanceSI
            * block.currentDensityUpperBoundSI
            / block.reversibleCellPotentialSI
            * block.currentDensityScaled
        )

    @block.Constraint()
    def _fc_voltage_constr(block):
        # return (
        #     block.cellPotentialSI
        #     == block.reversibleCellPotentialSI
        #     + block.activationOverpotentialSI
        #     + block.concentrationOverpotentialSI
        #     - block.ohmicLossesSI
        # )
        return (
            block.cellPotentialScaled
            == 1
            + block.activationOverpotentialScaled
            + block.concentrationOverpotentialScaled
            - block.ohmicLossesScaled
        )

    @block.Constraint()
    def _fc_power_density_const(block):
        # return block.powerDensitySI - block.powerDensitySI * block.totalActiveAreaSI == 0
        return (
            block.powerDensityUpperBoundSI
            / (block.currentDensityUpperBoundSI * block.reversibleCellPotentialSI)
            * block.powerDensityScaled
            == block.cellPotentialScaled * block.currentDensityScaled
        )

    @block.Constraint()
    def _fc_charge_transfer_rate_constr(block):
        return (
            block.totalChargeTransferRateScaled
            == block.currentDensityScaled * block.totalActiveAreaScaled
        )

    @block.Expression()
    def electricalPowerSI(block):
        return block.powerDensitySI * block.totalActiveAreaSI

    return block


def pyomo_create_model(**kwargs):

    model = pyo.AbstractModel("Feasibility problem: fuel cell")

    model.fuel_cell = pyo.Block(rule=fuel_cell_block_rule)

    @model.Objective(sense=pyo.minimize)
    def obj(model):
        return 0

    return model


def pyomo_print_result(**kwargs):
    model = kwargs["model"]
    eta = get_fuelCellEfficiency(model.fuel_cell.cellPotentialSI)
    print(
        f"Current density (A/cm2):\t{pyo.value(model.fuel_cell.currentDensitySI)*1e-4:.2f}"
    )
    print(
        f"Power density (W/cm2):\t\t{pyo.value(model.fuel_cell.powerDensitySI)*1e-4:.2f}"
    )
    print(f"Cell potential (V):\t\t{pyo.value(model.fuel_cell.cellPotentialSI):.2f}")
    print(f"Electrical efficiency:\t\t{pyo.value(eta):.2f}")
