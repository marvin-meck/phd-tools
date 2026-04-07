"""phdtools.optimization.data_preparation.py

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
from typing import Dict, TextIO

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from phdtools.data import ISO_STD_REF_PRESSURE_SI, ISO_STD_REF_TEMPERATURE_SI, Compound

from phdtools.data.constants import FARADAY_CONST_SI, GAS_CONST_SI
from phdtools.data.thermochemical import (
    stoichiometricNumber,
    get_stdEnthalpyFromIntegration,
)
from phdtools.data.thermophysical import vapourPressureModel

from phdtools.models.xu_froment_1989 import (
    STST_PRESSURE_BAR as STST_PRESSURE_BAR_REFORMER,
    ModelParameters as SteamReformingParameters,
    Reaction as SteamReformingReactions,
    rateConstModel as rateConstModelSteamReforming,
    equilibriumConstModel as equilibriumConstModelSteamReforming,
    adsorptionCoefModel as adsorptionCoefModelSteamReforming,
)

from phdtools.models.mendes_2010 import (
    STST_PRESSURE_BAR as STST_PRESSURE_BAR_SHIFT,
    ModelParameters as WaterGasShiftParameters,
    rateConstModel as rateConstModelWaterGasShift,
    equilibriumConstModel as equilibriumConstModelWaterGasShift,
    adsorptionCoefModel as adsorptionCoefModelWaterGasShift,
)

from phdtools.models.meck_2025 import get_fuelCellParameters

from phdtools.optimization import (
    EPSILON,
    NUM_FINITE_ELEMENTS_REFORMER,
    NUM_FINITE_ELEMENTS_SHIFT,
    NOMINAL_THERMAL_POWER_SI,
    FUEL_CELL_POWER_SI_LB,
    FUEL_CELL_POWER_SI_UB,
    THERMAL_EFFICIENCY_LB,
    THERMAL_EFFICIENCY_UB,
    FLOW_TEMPERATURE_HEATING_SI,
    TERMINAL_TEMPERATURE_DIFFERENCE_SI,
    GROSS_CALORIFIC_VALUE_METHANE_SI,
    OXYGEN_TO_CARBON_RATIO_LB,
    OXYGEN_TO_CARBON_RATIO_UB,
    REFORMER_STEAM_TO_CARBON_LB,
    REFORMER_STEAM_TO_CARBON_UB,
    REFORMER_TEMPERATURE_SI_LB,
    REFORMER_TEMPERATURE_SI_UB,
    REFORMER_PRESSURE_SI,
    REFORMER_MASS_OF_SOLIDS_GRAM_LB,
    REFORMER_MASS_OF_SOLIDS_GRAM_UB,
    REFORMER_METHANE_CONVERSION_LB,
    SHIFT_TEMPERATURE_SI_LB,
    SHIFT_TEMPERATURE_SI_UB,
    SHIFT_PRESSURE_SI,
    SHIFT_MASS_OF_SOLIDS_GRAM_LB,
    SHIFT_MASS_OF_SOLIDS_GRAM_UB,
    FUEL_CELL_TEMPERATURE_SI,
    FUEL_CELL_PRESSURE_SI,
    FUEL_CELL_CARBON_MONOXIDE_TOLERANCE,
    FUEL_CELL_TOTAL_ACTIVE_AREA_SI_LB,
    FUEL_CELL_TOTAL_ACTIVE_AREA_SI_UB,
    REFORMER_BULK_DENSITY_CATALYST_SI,
    REFORMER_VOID_FRACTION_CATALYST_BED,
    REFORMER_TUBE_DIAMETER_SI,
    REFORMER_TUBE_WALL_THICKNESS_SI,
    SHIFT_BULK_DENSITY_CATALYST_SI,
    SHIFT_VOID_FRACTION_CATALYST_BED,
    SHIFT_TUBE_DIAMETER_SI,
    SHIFT_TUBE_WALL_THICKNESS_SI,
    FUEL_CELL_PRICE_EURO_PER_SQUARE_METER,
    INVESTMENT_TYPE,
    FEED_IN_TARIFF_EUR_PER_KWH,
    CONTRACT_DURATION_YEARS,
    CONTRIBUTION_MARGIN_LOWER_BOUND_EURO,
    CONTRIBUTION_MARGIN_UPPER_BOUND_EURO,
    INDIRECT_COSTS_EURO,
    GAS_PRICE_EUR_PER_KWH,
    ELECTRICITY_PRICE_EUR_PER_KWH,
    THERMAL_EFFICIENCY_STATUS_QUO,
    MATCHING_FACTOR,
    CO2_EMISSION_FACTOR_MOL_PER_KWH,
)


from phdtools.optimization.preprocessing import (
    get_currentDensityUpperBound,
    get_powerDensityUpperBound,
    get_moleFractionsAir,
    get_molarFlowRateBoundsSI,
    get_plugFlowReactorCostCoefficients,
    get_variableCostsBounds,
    get_energyCostSavingsLowerBound,
    get_energyCostSavingsUpperBound,
    get_carbonDioxideEmissionReductionsLowerBound,
    get_carbonDioxideEmissionReductionsUpperBound,
)

from phdtools.optimization.pyomo import (
    BaseModel,
    SteamReformingCompounds,
    reformer_block_rule,
    shift_block_rule,
    fuel_cell_block_rule,
    add_consumer_preference_model,
)

from phdtools.io.write_datacmds import write_data_commands


def create_reformer_data_dict(namespace: str | None = None) -> Dict:

    params = SteamReformingParameters.init()

    def _rate_const_bounds_rule(r, params):
        return (
            rateConstModelSteamReforming(REFORMER_TEMPERATURE_SI_LB, params)[0][
                r.value
            ],
            rateConstModelSteamReforming(REFORMER_TEMPERATURE_SI_UB, params)[0][
                r.value
            ],
        )

    def _equilibrium_const_bounds_rule(r, params):
        a = equilibriumConstModelSteamReforming(REFORMER_TEMPERATURE_SI_LB, params)[0][
            r.value
        ]
        b = equilibriumConstModelSteamReforming(REFORMER_TEMPERATURE_SI_UB, params)[0][
            r.value
        ]
        return (a, b) if a <= b else (b, a)

    def _adsorption_coef_bounds_rule(k, params):
        a = adsorptionCoefModelSteamReforming(REFORMER_TEMPERATURE_SI_LB, params)[0][
            SteamReformingCompounds[k].value
        ]
        b = adsorptionCoefModelSteamReforming(REFORMER_TEMPERATURE_SI_UB, params)[0][
            SteamReformingCompounds[k].value
        ]
        return (a, b) if a <= b else (b, a)

    def _molar_flow_rate_bounds_rule(k, t):

        molarFlowRateMethaneInUpperBound = (
            THERMAL_EFFICIENCY_LB ** (-1)
            * NOMINAL_THERMAL_POWER_SI
            / np.abs(GROSS_CALORIFIC_VALUE_METHANE_SI)
        )
        molarFlowRateSteamInUpperBound = (
            REFORMER_STEAM_TO_CARBON_UB * molarFlowRateMethaneInUpperBound
        )

        if k == "C1H4(g)":
            ub = molarFlowRateMethaneInUpperBound
        elif k == "C1O1(g)":
            ub = (
                -1
                * molarFlowRateMethaneInUpperBound
                * stoichiometricNumber.loc[k, "SMR"]
                / stoichiometricNumber.loc["C1H4(g)", "SMR"]
            )
        elif k == "C1O2(g)":
            ub = (
                -1
                * molarFlowRateMethaneInUpperBound
                * stoichiometricNumber.loc[k, "DSR"]
                / stoichiometricNumber.loc["C1H4(g)", "DSR"]
            )
        elif k == "H2O1(g)":
            ub = molarFlowRateSteamInUpperBound
        elif k == "H2(ref)":
            ub = (
                -1
                * molarFlowRateSteamInUpperBound
                * stoichiometricNumber.loc[k, "DSR"]
                / stoichiometricNumber.loc["C1H4(g)", "DSR"]
            )
        else:
            raise ValueError(f"There should be no index {k}")
            # ub = molarFlowRateMethaneInUpperBound + molarFlowRateSteamInUpperBound

        return EPSILON, ub

    data_dict = {
        namespace: {
            "NUM_FINITE_ELEMENTS": {None: NUM_FINITE_ELEMENTS_REFORMER},
            "STST_PRESSURE_BAR": {None: STST_PRESSURE_BAR_REFORMER},
            "GAS_CONST_SI": {None: GAS_CONST_SI},
            "setReactingCompounds": {None: [c.name for c in SteamReformingCompounds]},
            "setSteamReformingReactions": {
                None: [r.name for r in SteamReformingReactions]
            },
            "setIndependentComponents": {None: ["C1H4(g)", "C1O2(g)"]},
            "setCompoundsAdsorption": {
                None: ["C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"]
            },
            "stoichiometricNumber": {
                (r.name, c.name): int(stoichiometricNumber.loc[c.name, r.name])
                for r in SteamReformingReactions
                for c in SteamReformingCompounds
            },
            "pressureBar": {None: REFORMER_PRESSURE_SI * 1e-5},
            "rateConstantRefSI": {
                r.name: float(params.rateConstantRefSI[r.value])
                for r in SteamReformingReactions
            },
            "activationEnergySI": {
                r.name: float(params.activationEnergySI[r.value])
                for r in SteamReformingReactions
            },
            "equilibriumConstRef": {
                r.name: float(params.equilibriumConstRef[r.value])
                for r in SteamReformingReactions
            },
            "enthalpyReactionSI": {
                r.name: float(params.enthalpyReactionSI[r.value])
                for r in SteamReformingReactions
            },
            "adsorptionCoefRef": {
                c.name: float(params.adsorptionCoefRef[c.value])
                for c in SteamReformingCompounds
                if c.name in ["C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"]
            },
            "enthalpyAdsorptionSI": {
                c.name: float(params.enthalpyAdsorptionSI[c.value])
                for c in SteamReformingCompounds
                if c.name in ["C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"]
            },
            "refTemperatureEquilibriumSI": {
                r.name: float(params.refTemperatureEquilibriumSI[r.value])
                for r in SteamReformingReactions
            },
            "refTemperatureAdsorptionSI": {
                c.name: float(params.refTemperatureAdsorptionSI[c.value])
                for c in SteamReformingCompounds
                if c.name in ["C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"]
            },
            "refTemperatureRateSI": {
                r.name: float(params.refTemperatureRateSI[r.value])
                for r in SteamReformingReactions
            },
            "temperatureLowerBoundSI": {None: REFORMER_TEMPERATURE_SI_LB},
            "temperatureUpperBoundSI": {None: REFORMER_TEMPERATURE_SI_UB},
            "rateConstLowerBound": {
                r.name: float(_rate_const_bounds_rule(r, params)[0])
                for r in SteamReformingReactions
            },
            "rateConstUpperBound": {
                r.name: float(_rate_const_bounds_rule(r, params)[1])
                for r in SteamReformingReactions
            },
            "equilibriumConstLowerBound": {
                r.name: float(_equilibrium_const_bounds_rule(r, params)[0])
                for r in SteamReformingReactions
            },
            "equilibriumConstUpperBound": {
                r.name: float(_equilibrium_const_bounds_rule(r, params)[1])
                for r in SteamReformingReactions
            },
            "adsorptionCoefLowerBound": {
                k: float(_adsorption_coef_bounds_rule(k, params)[0])
                for k in ["C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"]
            },
            "adsorptionCoefUpperBound": {
                k: float(_adsorption_coef_bounds_rule(k, params)[1])
                for k in ["C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"]
            },
            "massCatalystLowerBoundSI": {None: 1e-3 * REFORMER_MASS_OF_SOLIDS_GRAM_LB},
            "massCatalystUpperBoundSI": {None: 1e-3 * REFORMER_MASS_OF_SOLIDS_GRAM_UB},
            "molarFlowRateUpperBoundSI": {
                (k.name, t): float(_molar_flow_rate_bounds_rule(k.name, t)[1])
                for k in SteamReformingCompounds
                for t in range(0, NUM_FINITE_ELEMENTS_REFORMER + 1, 1)
            },
        }
    }

    return data_dict


def create_shift_data_dict(namespace: str | None = None) -> Dict:

    params = WaterGasShiftParameters.init(model="LH1")

    def _rate_const_bounds_rule(params):
        return (
            rateConstModelWaterGasShift(SHIFT_TEMPERATURE_SI_LB, params)[0][0],
            rateConstModelWaterGasShift(SHIFT_TEMPERATURE_SI_UB, params)[0][0],
        )

    def _equilibrium_const_bounds_rule(params):
        a = equilibriumConstModelWaterGasShift(SHIFT_TEMPERATURE_SI_LB, params)[0][0]
        b = equilibriumConstModelWaterGasShift(SHIFT_TEMPERATURE_SI_UB, params)[0][0]
        return (a, b) if a <= b else (b, a)

    def _adsorption_coef_bounds_rule(k, params):
        a = adsorptionCoefModelWaterGasShift(SHIFT_TEMPERATURE_SI_LB, params)[0][
            SteamReformingCompounds[k].value
        ]
        b = adsorptionCoefModelWaterGasShift(SHIFT_TEMPERATURE_SI_UB, params)[0][
            SteamReformingCompounds[k].value
        ]
        return (a, b) if a <= b else (b, a)

    def _molar_flow_rate_bounds_rule(k, t):

        molarFlowRateMethaneInUpperBound = (
            THERMAL_EFFICIENCY_LB ** (-1)
            * NOMINAL_THERMAL_POWER_SI
            / np.abs(GROSS_CALORIFIC_VALUE_METHANE_SI)
        )
        molarFlowRateSteamInUpperBound = (
            REFORMER_STEAM_TO_CARBON_UB * molarFlowRateMethaneInUpperBound
        )

        if k == "C1H4(g)":
            ub = molarFlowRateMethaneInUpperBound
        elif k == "C1O1(g)":
            ub = (
                -1
                * molarFlowRateMethaneInUpperBound
                * stoichiometricNumber.loc[k, "SMR"]
                / stoichiometricNumber.loc["C1H4(g)", "SMR"]
            )
        elif k == "C1O2(g)":
            ub = (
                -1
                * molarFlowRateMethaneInUpperBound
                * stoichiometricNumber.loc[k, "DSR"]
                / stoichiometricNumber.loc["C1H4(g)", "DSR"]
            )
        elif k == "H2O1(g)":
            ub = molarFlowRateSteamInUpperBound
        elif k == "H2(ref)":
            ub = (
                -1
                * molarFlowRateSteamInUpperBound
                * stoichiometricNumber.loc[k, "DSR"]
                / stoichiometricNumber.loc["C1H4(g)", "DSR"]
            )
        else:
            raise ValueError(f"There should be no index {k}")
            # ub = molarFlowRateMethaneInUpperBound + molarFlowRateSteamInUpperBound

        return EPSILON, ub

    data_dict = {
        namespace: {
            "NUM_FINITE_ELEMENTS": {None: NUM_FINITE_ELEMENTS_SHIFT},
            "setReactingCompounds": {None: [c.name for c in SteamReformingCompounds]},
            # "setInertCompounds": {None:[]},
            "setCompoundsAdsorption": {
                None: ["C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"]
            },
            "STST_PRESSURE_BAR": {None: STST_PRESSURE_BAR_SHIFT},
            "GAS_CONST_SI": {None: GAS_CONST_SI},
            "stoichiometricNumber": {
                c.name: int(stoichiometricNumber.loc[c.name, "WGS"])
                for c in SteamReformingCompounds
            },
            "pressureBar": {None: SHIFT_PRESSURE_SI * 1e-5},
            "rateConstantFactorSI": {None: float(params.rateConstantFactorSI)},
            "activationEnergySI": {None: float(params.activationEnergySI)},
            "equilibriumConstRef": {None: float(params.equilibriumConstRef)},
            "enthalpyReactionSI": {None: float(params.enthalpyReactionSI)},
            "adsorptionCoefFactor": {
                c.name: float(params.adsorptionCoefFactor[c.value])
                for c in SteamReformingCompounds
                if c.name in ["C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"]
            },
            "enthalpyAdsorptionSI": {
                c.name: float(params.enthalpyAdsorptionSI[c.value])
                for c in SteamReformingCompounds
                if c.name in ["C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"]
            },
            "refTemperatureEquilibriumSI": {
                None: float(params.refTemperatureEquilibriumSI)
            },
            "temperatureLowerBoundSI": {None: SHIFT_TEMPERATURE_SI_LB},
            "temperatureUpperBoundSI": {None: SHIFT_TEMPERATURE_SI_UB},
            "rateConstLowerBound": {None: float(_rate_const_bounds_rule(params)[0])},
            "rateConstUpperBound": {None: float(_rate_const_bounds_rule(params)[1])},
            "equilibriumConstLowerBound": {
                None: float(_equilibrium_const_bounds_rule(params)[0])
            },
            "equilibriumConstUpperBound": {
                None: float(_equilibrium_const_bounds_rule(params)[1])
            },
            "adsorptionCoefLowerBound": {
                k: float(_adsorption_coef_bounds_rule(k, params)[0])
                for k in ["C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"]
            },
            "adsorptionCoefUpperBound": {
                k: float(_adsorption_coef_bounds_rule(k, params)[1])
                for k in ["C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"]
            },
            "massCatalystLowerBoundSI": {None: 1e-3 * SHIFT_MASS_OF_SOLIDS_GRAM_LB},
            "massCatalystUpperBoundSI": {None: 1e-3 * SHIFT_MASS_OF_SOLIDS_GRAM_UB},
            "molarFlowRateUpperBoundSI": {
                (k.name, t): float(_molar_flow_rate_bounds_rule(k.name, t)[1])
                for k in SteamReformingCompounds
                for t in range(0, NUM_FINITE_ELEMENTS_SHIFT + 1, 1)
            },
        }
    }

    return data_dict


def create_fuel_cell_data_dict(namespace: str | None = None) -> Dict:
    params = get_fuelCellParameters(
        temperatureKelvin=FUEL_CELL_TEMPERATURE_SI,
        pressureBar=FUEL_CELL_PRESSURE_SI * 1e-5,
    )

    currentDensityUpperBoundSI = get_currentDensityUpperBound(params)
    powerDensityUpperBoundSI = get_powerDensityUpperBound(
        params, currentDensityUpperBoundSI
    )

    data_dict = {
        namespace: {
            "GAS_CONST_SI": {None: GAS_CONST_SI},
            "FARADAY_CONST_SI": {None: FARADAY_CONST_SI},
            **{key: {None: val} for key, val in params.__dict__.items()},
            "currentDensityUpperBoundSI": {None: float(currentDensityUpperBoundSI)},
            "powerDensityUpperBoundSI": {None: float(powerDensityUpperBoundSI)},
            "totalActiveAreaLowerBoundSI": {None: FUEL_CELL_TOTAL_ACTIVE_AREA_SI_LB},
            "totalActiveAreaUpperBoundSI": {None: FUEL_CELL_TOTAL_ACTIVE_AREA_SI_UB},
        }
    }

    return data_dict


def create_base_data_dict(
    fname_cost_coefs: str | os.PathLike, namespace: str | None = None
) -> Dict:

    data_dict = {namespace: dict()}

    stdEnthalpySI = pd.concat(
        [
            get_stdEnthalpyFromIntegration(
                temperatureKelvin=ISO_STD_REF_TEMPERATURE_SI
            ),
            get_stdEnthalpyFromIntegration(
                temperatureKelvin=FLOW_TEMPERATURE_HEATING_SI
                + TERMINAL_TEMPERATURE_DIFFERENCE_SI
            ),
        ],
        axis=1,
    )

    moleFractionsAir = get_moleFractionsAir()

    costCoefficientsReformer = get_plugFlowReactorCostCoefficients(
        bulkDensityCatalysSI=REFORMER_BULK_DENSITY_CATALYST_SI,
        voidFractionCatalystBed=REFORMER_VOID_FRACTION_CATALYST_BED,
        tubeDiameterSI=REFORMER_TUBE_DIAMETER_SI,
        tubeWallThicknessSI=REFORMER_TUBE_WALL_THICKNESS_SI,
        fname_coefs=fname_cost_coefs,
    )

    costCoefficientsShift = get_plugFlowReactorCostCoefficients(
        bulkDensityCatalysSI=SHIFT_BULK_DENSITY_CATALYST_SI,
        voidFractionCatalystBed=SHIFT_VOID_FRACTION_CATALYST_BED,
        tubeDiameterSI=SHIFT_TUBE_DIAMETER_SI,
        tubeWallThicknessSI=SHIFT_TUBE_WALL_THICKNESS_SI,
        fname_coefs=fname_cost_coefs,
    )

    molarFlowRateLowerBoundSI, molarFlowRateUpperBoundSI = get_molarFlowRateBoundsSI()
    variableCostLowerBoundEuro, variableCostUpperBoundEuro = get_variableCostsBounds(
        fname_coefs=fname_cost_coefs
    )

    data_dict = {
        namespace: {
            "setCompounds": {None: [c.name for c in Compound]},
            "setCompoundsIn": {
                None: ["C1H4(g)", "H2O1(g)", "H2O1(l)", "N2(ref)", "O2(ref)"]
            },
            "setCompoundsOut": {
                None: ["C1O2(g)", "H2O1(g)", "H2O1(l)", "N2(ref)", "O2(ref)"]
            },
            "setCompoundsAir": {None: ["N2(ref)", "O2(ref)", "H2O1(g)"]},
            "costCoefIndex": {
                None: [
                    ("R1", "a1"),
                    ("R1", "a2"),
                    ("R1", "k"),
                    ("R2", "a1"),
                    ("R2", "a2"),
                    ("R2", "k"),
                    ("FC", "a1"),
                ]
            },
            "costCoef": {
                ("R1", "a1"): float(costCoefficientsReformer["a1"]),
                ("R1", "a2"): float(costCoefficientsReformer["a2"]),
                ("R1", "k"): float(costCoefficientsReformer["k"]),
                ("R2", "a1"): float(costCoefficientsShift["a1"]),
                ("R2", "a2"): float(costCoefficientsShift["a2"]),
                ("R2", "k"): float(costCoefficientsShift["k"]),
                ("FC", "a1"): float(FUEL_CELL_PRICE_EURO_PER_SQUARE_METER),
            },
            "variableCostLowerBoundEuro": {None: variableCostLowerBoundEuro},
            "variableCostUpperBoundEuro": {None: variableCostUpperBoundEuro},
            "INDIRECT_COSTS_EURO": {None: INDIRECT_COSTS_EURO},
            "NOMINAL_THERMAL_POWER_SI": {None: NOMINAL_THERMAL_POWER_SI},
            "FUEL_CELL_POWER_SI_LB": {None: FUEL_CELL_POWER_SI_LB},
            "FUEL_CELL_POWER_SI_UB": {None: FUEL_CELL_POWER_SI_UB},
            "OXYGEN_TO_CARBON_RATIO_LB": {None: OXYGEN_TO_CARBON_RATIO_LB},
            "OXYGEN_TO_CARBON_RATIO_UB": {None: OXYGEN_TO_CARBON_RATIO_UB},
            "REFORMER_STEAM_TO_CARBON_LB": {None: REFORMER_STEAM_TO_CARBON_LB},
            "REFORMER_STEAM_TO_CARBON_UB": {None: REFORMER_STEAM_TO_CARBON_UB},
            "FUEL_CELL_CARBON_MONOXIDE_TOLERANCE": {
                None: FUEL_CELL_CARBON_MONOXIDE_TOLERANCE
            },
            "pressureSI": {None: ISO_STD_REF_PRESSURE_SI},
            "stoichiometricNumber": {
                k: float(stoichiometricNumber.loc[k, "MCR1"])
                for k in ["C1H4(g)", "O2(ref)", "C1O2(g)", "H2O1(g)"]
            },
            "moleFractionsAir": {
                "N2(ref)": float(moleFractionsAir[Compound["N2(ref)"].value]),
                "O2(ref)": float(moleFractionsAir[Compound["O2(ref)"].value]),
                "H2O1(g)": float(moleFractionsAir[Compound["H2O1(g)"].value]),
            },
            "vapourPressureGasOutSI": {
                None: float(
                    vapourPressureModel(
                        temperatureKelvin=FLOW_TEMPERATURE_HEATING_SI
                        + TERMINAL_TEMPERATURE_DIFFERENCE_SI
                    )[0]
                )
            },
            "stdEnthalpyInSI": {
                **{
                    k: float(stdEnthalpySI.loc[k, ISO_STD_REF_TEMPERATURE_SI])
                    for k in ["C1H4(g)", "H2O1(g)", "N2(ref)", "O2(ref)"]
                },
                "H2O1(l)": float(
                    stdEnthalpySI.loc[
                        "H2O1(l)",
                        FLOW_TEMPERATURE_HEATING_SI
                        + TERMINAL_TEMPERATURE_DIFFERENCE_SI,
                    ]
                ),
            },
            "stdEnthalpyOutSI": {
                k: float(
                    stdEnthalpySI.loc[
                        k,
                        FLOW_TEMPERATURE_HEATING_SI
                        + TERMINAL_TEMPERATURE_DIFFERENCE_SI,
                    ]
                )
                for k in ["C1O2(g)", "H2O1(g)", "H2O1(l)", "N2(ref)", "O2(ref)"]
            },
            "grossCalorificValueMethaneSI": {
                None: float(GROSS_CALORIFIC_VALUE_METHANE_SI)
            },
            "molarFlowRateInLowerBoundSI": {
                c.name: float(molarFlowRateLowerBoundSI[c.value])
                for c in Compound
                if c.name in {"C1H4(g)", "H2O1(l)", "N2(ref)", "O2(ref)", "H2O1(g)"}
            },
            "molarFlowRateInUpperBoundSI": {
                c.name: float(molarFlowRateUpperBoundSI[c.value])
                for c in Compound
                if c.name in {"C1H4(g)", "H2O1(l)", "N2(ref)", "O2(ref)", "H2O1(g)"}
            },
            "methaneConversionLowerBound": {
                None: float(REFORMER_METHANE_CONVERSION_LB)
            },
        }
    }
    return data_dict


def create_logit_model_data_dict(
    fname_logit_coefs: str | os.PathLike,
    fname_s: str | os.PathLike,
    namespace: str | None = None,
) -> Dict:

    data_dict = {namespace: {}}

    logit_coefs = pd.read_json(fname_logit_coefs, typ="series")

    socio_demographic_attributes = pd.read_csv(fname_s, comment="#", index_col=0)

    mask = socio_demographic_attributes["HEATSYS"] == 1
    socio_demographic_attributes = socio_demographic_attributes[mask].drop(
        ["ENECOST", "INCOME"], axis=1
    )

    logitVariablesMainEffects = ["ICOST", "CSAV", "CO2SAV", "FIT", "ITYPE", "DUR"]

    data_dict[namespace] = {
        "setAgents": {None: list(socio_demographic_attributes.index)},
        "logitVariablesIndex": {None: list(logit_coefs.index)},
        "socioDemographicAttributesIndex": {
            None: list(socio_demographic_attributes.columns)
        },
        "logitVariablesMainEffectsIndex": {None: logitVariablesMainEffects},
        "logitCoefs": logit_coefs.to_dict(),
        "socioDemographicAttributes": {
            (n, j): float(socio_demographic_attributes.loc[n, j])
            for n in socio_demographic_attributes.index
            for j in socio_demographic_attributes.columns
        },
        "FEED_IN_TARIFF_EUR_PER_KWH": {None: FEED_IN_TARIFF_EUR_PER_KWH},
        "INVESTMENT_TYPE": {None: INVESTMENT_TYPE},
        "CONTRACT_DURATION_YEARS": {None: CONTRACT_DURATION_YEARS},
    }

    return data_dict


def create_reformer_datafile(ostream: TextIO) -> None:

    ns = None
    data_dict = {ns: dict()}

    data_dict[ns].update(create_reformer_data_dict(namespace="reformer"))

    model = pyo.AbstractModel()
    model.reformer = pyo.Block(rule=reformer_block_rule)
    instance = model.create_instance(data_dict)

    write_data_commands(instance.reformer, ostream, data_dict[ns]["reformer"])


def create_shift_datafile(ostream: TextIO) -> None:

    ns = None
    data_dict = {ns: dict()}

    data_dict[ns].update(create_shift_data_dict(namespace="shift"))

    model = pyo.AbstractModel()
    model.shift = pyo.Block(rule=shift_block_rule)
    instance = model.create_instance(data_dict)

    write_data_commands(instance.shift, ostream, data_dict[ns]["shift"])


def create_fuel_cell_datafile(ostream: TextIO) -> None:

    ns = None
    data_dict = {ns: dict()}

    data_dict[ns].update(create_fuel_cell_data_dict(namespace="fuel_cell"))

    model = pyo.AbstractModel()
    model.fuel_cell = pyo.Block(rule=fuel_cell_block_rule)
    instance = model.create_instance(data_dict)

    write_data_commands(instance.fuel_cell, ostream, data_dict[ns]["fuel_cell"])


def create_base_datafile(ostream: TextIO, fname_cost_coefs: str | os.PathLike) -> None:

    ns = None
    data_dict = {None: dict()}

    data_dict[ns].update(create_base_data_dict(fname_cost_coefs, ns)[ns])

    model = BaseModel("Cost minimization model")

    write_data_commands(model=model, ostream=ostream, data_dict=data_dict[ns])


def create_consumer_preference_model_datafile(
    ostream: TextIO,
    fname_logit_coefs: str | os.PathLike,
    fname_s: str | os.PathLike,
    fname_heat: str | os.PathLike,
    fname_electricity: str | os.PathLike,
    fname_cost_coefs: str | os.PathLike,
) -> None:

    ns = None

    data_dict = create_logit_model_data_dict(
        fname_logit_coefs=fname_logit_coefs, fname_s=fname_s, namespace=ns
    )

    annualHeatDemandSI = pd.read_csv(fname_heat, comment="#", index_col=0)
    annualElectricityDemandSI = pd.read_csv(fname_electricity, comment="#", index_col=0)
    variableCostsLowerBound, variableCostsUpperBound = get_variableCostsBounds(
        fname_coefs=fname_cost_coefs
    )

    data_dict[ns].update(annualHeatDemandSI.to_dict())
    data_dict[ns].update(annualElectricityDemandSI.to_dict())
    data_dict[ns].update(
        {
            "GAS_PRICE_EUR_PER_KWH": {None: GAS_PRICE_EUR_PER_KWH},
            "ELECTRICITY_PRICE_EUR_PER_KWH": {None: ELECTRICITY_PRICE_EUR_PER_KWH},
            "THERMAL_EFFICIENCY_STATUS_QUO": {None: THERMAL_EFFICIENCY_STATUS_QUO},
            "MATCHING_FACTOR": {None: MATCHING_FACTOR},
            "CO2_EMISSION_FACTOR_MOL_PER_KWH": {None: CO2_EMISSION_FACTOR_MOL_PER_KWH},
            "priceLowerBoundEuro": {
                None: float(
                    variableCostsLowerBound + CONTRIBUTION_MARGIN_LOWER_BOUND_EURO
                )
            },
            "priceUpperBoundEuro": {
                None: float(
                    variableCostsUpperBound + CONTRIBUTION_MARGIN_UPPER_BOUND_EURO
                )
            },
            "energyCostSavingsLowerBound": get_energyCostSavingsLowerBound(
                fname_heat=fname_heat, fname_electricity=fname_electricity
            ),
            "energyCostSavingsUpperBound": get_energyCostSavingsUpperBound(
                fname_heat=fname_heat, fname_electricity=fname_electricity
            ),
            "carbonDioxideEmissionReductionsUpperBound": get_carbonDioxideEmissionReductionsUpperBound(
                fname_heat=fname_heat, fname_electricity=fname_electricity
            ),
            "carbonDioxideEmissionReductionsLowerBound": get_carbonDioxideEmissionReductionsLowerBound(
                fname_heat=fname_heat, fname_electricity=fname_electricity
            ),
            "carbonDioxideEmissionReductionsUpperBound": get_carbonDioxideEmissionReductionsUpperBound(
                fname_heat=fname_heat, fname_electricity=fname_electricity
            ),
        }
    )

    model = pyo.AbstractModel()
    add_consumer_preference_model(model)

    write_data_commands(model=model, ostream=ostream, data_dict=data_dict[ns])


def create_cost_minimization_datafile(
    ostream: TextIO, epsElectricalPowerSI: float
) -> None:
    ostream.write("include base.dat;\n\n")
    # ostream.write(f"param epsThermalEfficiency := {epsThermalEfficiency};\n\n")
    ostream.write(f"param epsElectricalPowerSI := {epsElectricalPowerSI};\n\n")
    for ns in ["reformer", "shift", "fuel_cell"]:
        ostream.write(f"namespace {ns} {{\n")
        ostream.write(f"    include {ns}.dat;\n")
        ostream.write("}\n\n")


def create_demand_maximization_datafile(
    ostream: TextIO, markup: float | None = None, contribution: float | None = None
) -> None:

    if (markup is None) and (contribution is None):
        raise ValueError(
            "Cannot interpret input. Values for both markup and contribution given, expected only one"
        )
    elif (markup is not None) and (contribution is not None):
        raise ValueError("variant must be either 'markup' OR 'contribution'")

    ostream.write("include base.dat;\n\n")
    ostream.write("include consumer_preferences.dat;\n\n")

    if markup:
        ostream.write(f"param MARKUP:= {markup};\n\n")
    elif contribution:
        ostream.write(f"param CONTRIBUTION_MARGIN_EURO:= {contribution};\n\n")

    for ns in ["reformer", "shift", "fuel_cell"]:
        ostream.write(f"namespace {ns} {{\n")
        ostream.write(f"    include {ns}.dat;\n")
        ostream.write("}\n\n")


def create_profit_maximization_datafile(ostream: TextIO) -> None:
    ostream.write("include base.dat;\n\n")
    ostream.write("include consumer_preferences.dat;\n\n")
    ostream.write(
        f"param contributionMarginLowerBoundEuro := {CONTRIBUTION_MARGIN_LOWER_BOUND_EURO};\n"
    )
    ostream.write(
        f"param contributionMarginUpperBoundEuro := {CONTRIBUTION_MARGIN_UPPER_BOUND_EURO};\n\n"
    )

    ostream.write(f"param normalizedTotalContributionLowerBoundEuro := {0};\n")
    ostream.write(
        f"param normalizedTotalContributionUpperBoundEuro := {CONTRIBUTION_MARGIN_UPPER_BOUND_EURO};\n\n"
    )

    for ns in ["reformer", "shift", "fuel_cell"]:
        ostream.write(f"namespace {ns} {{\n")
        ostream.write(f"    include {ns}.dat;\n")
        ostream.write("}\n\n")
