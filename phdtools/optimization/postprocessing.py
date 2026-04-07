"""phdtools.optimization.postprocessing.py

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
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverResults
from scipy.integrate import solve_ivp

from phdtools.rdm import DataID

from phdtools.data import Compound
from phdtools.data.constants import FARADAY_CONST_SI
from phdtools.data.thermochemical import get_stdReactionEnthalpyFromKirchhoffsLaw

from phdtools.models.meck_2025 import (
    ELECTRONS_TRANSFERRED,
    fuelCellVoltageModel,
    ModelParameters as FuelCellParameters,
)

from phdtools.models.mendes_2010 import (
    initialValueProblemSpaceTime as shiftIVP,
    ModelParameters as WaterGasShiftParameters,
)

from phdtools.models.rommel_sagebiel_2017 import (
    compute_sample_avg_mxl_prob,
    compute_sample_avg_logit_prob,
)

from phdtools.models.xu_froment_1989 import (
    initialValueProblemSpaceTime as reformerIVP,
    ModelParameters as SteamReformingParameters,
)

from phdtools.optimization import (
    EPSILON,
    FEED_IN_TARIFF_EUR_PER_KWH,
    INVESTMENT_TYPE,
    CONTRACT_DURATION_YEARS,
    ISO_STD_REF_TEMPERATURE_SI,
)

from phdtools.optimization.preprocessing import (
    energyCostSavingsModel,
    carbonDioxideEmissionReductionsModel,
)

from phdtools.optimization.preprocessing import get_attributes_alternative
from phdtools.optimization.preprocessing import (
    energyCostSavingsModel,
    carbonDioxideEmissionReductionsModel,
)


def get_sampleAverageMixedLogitProbability(
    priceEuro,
    thermalEfficiency,
    powerIndex,
    matchingFactor,
    data_id_socio_demographic_attributes,
    data_id_mxl_coefs,
    data_id_heating_demands,
    data_id_electricity_demands,
    file_date_demands,
    file_date_agents,
    file_date_sample,
    num_agents,
    mxl_coefs_sample_size,
    fit=100 * FEED_IN_TARIFF_EUR_PER_KWH,
    itype=INVESTMENT_TYPE,
    dur=CONTRACT_DURATION_YEARS,
):
    designs = pd.DataFrame(
        np.c_[priceEuro, thermalEfficiency, powerIndex, matchingFactor],
        columns=["PRICE_EURO", "THERMAL_EFFICIENCY", "POWER_INDEX", "MATCHING_FACTOR"],
    )
    designs.index.name = "ALTERNATIVE"

    attributes_alternative = get_attributes_alternative(
        designs,
        data_id_socio_demographic_attributes=data_id_socio_demographic_attributes,
        data_id_heating_demands=data_id_heating_demands,
        data_id_electricity_demands=data_id_electricity_demands,
        file_date=file_date_demands,
        file_date_agents=file_date_agents,
        sample_size=num_agents,
        fit=fit,
        itype=itype,
        dur=dur,
    )

    sample_average_mxl_probabilities = compute_sample_avg_mxl_prob(
        attributes_alternative,
        fname_a=data_id_mxl_coefs.get_path(fail_exists=False)
        / f"{file_date_sample}_deterministic_coefficients.csv",
        fname_b=data_id_mxl_coefs.get_path(fail_exists=False)
        / f"{file_date_sample}_random_coefficients_{mxl_coefs_sample_size}.csv",
        fname_s=data_id_socio_demographic_attributes.get_path(fail_exists=False)
        / f"{file_date_agents}_socio_demographic_attributes_{num_agents}.csv",
    )

    return sample_average_mxl_probabilities


def get_sampleAverageLogitProbability(
    priceEuro,
    thermalEfficiency,
    powerIndex,
    matchingFactor,
    data_id_socio_demographic_attributes,
    data_id_logit_coefs,
    data_id_heating_demands,
    data_id_electricity_demands,
    file_date_demands,
    file_date_agents,
    file_date_logit_regression,
    num_agents,
    fit=100 * FEED_IN_TARIFF_EUR_PER_KWH,
    itype=INVESTMENT_TYPE,
    dur=CONTRACT_DURATION_YEARS,
):
    designs = pd.DataFrame(
        np.c_[priceEuro, thermalEfficiency, powerIndex, matchingFactor],
        columns=["PRICE_EURO", "THERMAL_EFFICIENCY", "POWER_INDEX", "MATCHING_FACTOR"],
    )
    designs.index.name = "ALTERNATIVE"

    attributes_alternative = get_attributes_alternative(
        designs,
        data_id_socio_demographic_attributes=data_id_socio_demographic_attributes,
        data_id_heating_demands=data_id_heating_demands,
        data_id_electricity_demands=data_id_electricity_demands,
        file_date=file_date_demands,
        file_date_agents=file_date_agents,
        sample_size=num_agents,
        fit=fit,
        itype=itype,
        dur=dur,
    )

    sample_average_logit_probabilities = compute_sample_avg_logit_prob(
        attributes_alternative,
        fname_s=data_id_socio_demographic_attributes.get_path(fail_exists=False)
        / f"{file_date_agents}_socio_demographic_attributes_{num_agents}.csv",
        fname_c=data_id_logit_coefs.get_path(fail_exists=False)
        / f"{file_date_logit_regression}_logit_coefficients.json",
    )

    return sample_average_logit_probabilities


def get_fuelCellEfficiency(cellPotentialSI):

    stdReactionEnthalpySI = get_stdReactionEnthalpyFromKirchhoffsLaw(
        temperatureKelvin=ISO_STD_REF_TEMPERATURE_SI,
        Tmin=298.15,
        Tmax=298.15,
        reactions={"HCR2"},
    )["HCR2"]

    eta = (
        -ELECTRONS_TRANSFERRED
        * FARADAY_CONST_SI
        * cellPotentialSI
        / stdReactionEnthalpySI
    )

    return eta


def get_optimization_results_space_time_reforming(block):

    molarFlowRateInSI = np.zeros(len(Compound))

    for c in Compound:
        if c.name in block.setReactingCompounds:
            molarFlowRateInSI[c.value] = pyo.value(block.molarFlowRateSI[c.name, 0])

    temperatureKelvin = pyo.value(block.temperatureKelvin)
    pressureBar = pyo.value(block.pressureBar)

    spaceTimePyomoSI = (
        pyo.value(block.massCatalystSI)
        * 1
        / block.setTimeSteps.last()
        * np.linspace(
            block.setTimeSteps.first(),
            block.setTimeSteps.last(),
            len(block.setTimeSteps),
        )
        / (pyo.value(block.molarFlowRateSI["C1H4(g)", 0]))
    )
    conversionPyomo = 1 - np.array(
        pyo.value(block.molarFlowRateSI["C1H4(g)", :])
    ) / pyo.value(block.molarFlowRateSI["C1H4(g)", 0])

    massCatalystSI = pyo.value(block.massCatalystSI)
    spaceTimeSI = np.linspace(
        0, massCatalystSI / molarFlowRateInSI[Compound["C1H4(g)"].value], 250
    )

    params = SteamReformingParameters.init()

    sol = solve_ivp(
        fun=reformerIVP,
        t_span=np.array([0, spaceTimeSI.max()]),
        y0=np.array([0, 0]),
        method="RK45",
        t_eval=spaceTimeSI,
        dense_output=False,
        events=None,
        vectorized=True,
        args=(molarFlowRateInSI, temperatureKelvin, pressureBar, params),
    )

    conversionScipy = sol.y[0]

    data_scipy = pd.DataFrame(
        {"spaceTimeSI": spaceTimeSI, "conversion": conversionScipy}
    )

    data_pyomo = pd.DataFrame(
        {"spaceTimeSI": spaceTimePyomoSI, "conversion": conversionPyomo}
    )

    return data_pyomo, data_scipy


def get_optimization_results_space_time_shift(block):

    molarFlowRateInSI = EPSILON * np.ones(len(Compound))

    for c in Compound:
        if c.name in block.setReactingCompounds:
            molarFlowRateInSI[c.value] = pyo.value(block.molarFlowRateSI[c.name, 0])

    temperatureKelvin = pyo.value(block.temperatureKelvin)
    pressureBar = pyo.value(block.pressureBar)

    spaceTimePyomoSI = (
        pyo.value(block.massCatalystSI)
        * 1
        / block.setTimeSteps.last()
        * np.linspace(
            block.setTimeSteps.first(),
            block.setTimeSteps.last(),
            len(block.setTimeSteps),
        )
        / (pyo.value(block.molarFlowRateSI["C1O1(g)", 0]))
    )
    conversionPyomo = 1 - np.array(
        pyo.value(block.molarFlowRateSI["C1O1(g)", :])
    ) / pyo.value(block.molarFlowRateSI["C1O1(g)", 0])

    massCatalystSI = pyo.value(block.massCatalystSI)
    spaceTimeSI = np.linspace(
        0, massCatalystSI / molarFlowRateInSI[Compound["C1O1(g)"].value], 250
    )

    moleFractionIn = molarFlowRateInSI / molarFlowRateInSI.sum()

    params = WaterGasShiftParameters.init(model="LH1")

    # Solve initial value problem
    sol = solve_ivp(
        fun=shiftIVP,
        t_span=[spaceTimeSI.min(), spaceTimeSI.max()],
        y0=[0],
        method="RK45",
        t_eval=spaceTimeSI,
        dense_output=False,
        events=None,
        vectorized=True,
        args=(moleFractionIn, temperatureKelvin, pressureBar, params),
    )

    conversionScipy = sol.y[0]

    data_scipy = pd.DataFrame(
        {"spaceTimeSI": spaceTimeSI, "conversion": conversionScipy}
    )

    data_pyomo = pd.DataFrame(
        {"spaceTimeSI": spaceTimePyomoSI, "conversion": conversionPyomo}
    )

    return data_pyomo, data_scipy


def get_reference_values_fuel_cell(block):

    params = FuelCellParameters(
        temperatureKelvin=pyo.value(block.temperatureKelvin),
        reversibleCellPotentialSI=pyo.value(block.reversibleCellPotentialSI),
        exchangeCurrentDensityCathodeSI=pyo.value(
            block.exchangeCurrentDensityCathodeSI
        ),
        transferCoefficientCathode=pyo.value(block.transferCoefficientCathode),
        areaSpecificResistanceSI=pyo.value(block.areaSpecificResistanceSI),
        limitingCurrentDensitySI=pyo.value(block.limitingCurrentDensitySI),
    )

    currentDensityRangeSI = np.linspace(
        EPSILON, pyo.value(block.currentDensityUpperBoundSI), 250
    )

    fuelCellVoltageValues = fuelCellVoltageModel(
        currentDensitySI=currentDensityRangeSI,
        params=params,
    )

    powerDenstiyValues = currentDensityRangeSI * fuelCellVoltageValues

    df = pd.DataFrame(
        {
            "currentDensityRangeSI": currentDensityRangeSI,
            "fuelCellVoltageValues": fuelCellVoltageValues,
            "powerDenstiyValues": powerDenstiyValues,
        }
    )

    return df


def pyomo_load_solution(
    fname,
    model: pyo.AbstractModel | pyo.ConcreteModel,
    modeldata: pyo.DataPortal | None = None,
):

    if isinstance(model, pyo.AbstractModel):
        if modeldata is None:
            raise ValueError(
                "'model' is of type 'AbstractModel', 'modeldata' cannot be 'None'!"
            )

        model.reformer.construct(modeldata.data(namespace="reformer"))
        model.shift.construct(modeldata.data(namespace="shift"))
        model.fuel_cell.construct(modeldata.data(namespace="fuel_cell"))
        instance = model.create_instance(modeldata, namespace=None)

        instance = model.create_instance(data=modeldata)

    elif isinstance(model, pyo.ConcreteModel):
        instance = model
    else:
        raise TypeError("Model must be 'AbstactModel' or 'ConcreteModel'")

    results = SolverResults()
    results.read(filename=fname)

    # fix the solution object, otherwise results.solutions.load_from(...) won't work
    results.solution(0)._cuid = False
    results.solution.Constraint = {}

    instance.solutions.load_from(results)

    # default_variable_value=0 doesn't work because smap_id = None,
    # so we set them manually
    for var in instance.component_data_objects(pyo.Var):
        if var.value is None:
            var.value = 0
        elif type(var.value):
            var.value = float(var.value)
        else:
            pass

    return instance


def calculate_consumer_preferences_cost_optimization_constant_markup(
    fname_results_summary: DataID,
    markup,
    fname_heat,
    fname_electricity,
    data_id_socio_demographic_attributes,
    data_id_mxl_coefs,
    data_id_logit_coefs,
    data_id_heating_demands,
    data_id_electricity_demands,
    file_date_agents,
    file_date_sample,
    file_date_demands,
    file_date_logit_regression,
    num_agents,
    mxl_coefs_sample_size,
    matchingFactor=0.5,
):

    solution_data = pd.read_csv(
        fname_results_summary,
        index_col=0,
        comment="#",
    )

    priceEuro = markup * solution_data.loc["Variable costs (Euro)"].to_numpy()

    thermalEfficiency = solution_data.loc["Thermal efficiency"].to_numpy()
    powerIndex = (
        solution_data.loc["Electrical power (kW)"].to_numpy()
        / solution_data.loc["Thermal power (kW)"].to_numpy()
    )
    matchingFactor = matchingFactor * np.ones(len(priceEuro))

    sample_average_mxl_probabilities = get_sampleAverageMixedLogitProbability(
        priceEuro,
        thermalEfficiency,
        powerIndex,
        matchingFactor,
        data_id_socio_demographic_attributes=data_id_socio_demographic_attributes,
        data_id_mxl_coefs=data_id_mxl_coefs,
        data_id_heating_demands=data_id_heating_demands,
        data_id_electricity_demands=data_id_electricity_demands,
        file_date_demands=file_date_demands,
        file_date_agents=file_date_agents,
        file_date_sample=file_date_sample,
        num_agents=num_agents,
        mxl_coefs_sample_size=mxl_coefs_sample_size,
    )

    sample_average_logit_probabilities = get_sampleAverageLogitProbability(
        priceEuro,
        thermalEfficiency,
        powerIndex,
        matchingFactor,
        data_id_socio_demographic_attributes=data_id_socio_demographic_attributes,
        data_id_logit_coefs=data_id_logit_coefs,
        data_id_heating_demands=data_id_heating_demands,
        data_id_electricity_demands=data_id_electricity_demands,
        file_date_demands=file_date_demands,
        file_date_agents=file_date_agents,
        file_date_logit_regression=file_date_logit_regression,
        num_agents=16,
    )

    energyCostSavings = np.full(len(thermalEfficiency), np.nan)
    for num in range(len(energyCostSavings)):
        energyCostSavings[num] = energyCostSavingsModel(
            thermalEfficiency=thermalEfficiency[num],
            powerIndex=powerIndex[num],
            matchingFactor=matchingFactor[num],
            fname_heat=fname_heat,
            fname_electricity=fname_electricity,
        ).mean()

    carbonDioxideEmissionReductions = np.full(len(thermalEfficiency), np.nan)
    for num in range(len(carbonDioxideEmissionReductions)):
        carbonDioxideEmissionReductions[num] = carbonDioxideEmissionReductionsModel(
            thermalEfficiency=thermalEfficiency[num],
            powerIndex=powerIndex[num],
            fname_heat=fname_heat,
            fname_electricity=fname_electricity,
        ).mean()

    solution_data.loc["Normalized total contribution, MXL (Euro)"] = (
        sample_average_mxl_probabilities.values
        * (priceEuro - solution_data.loc["Variable costs (Euro)"])
    )
    solution_data.loc["Normalized total contribution, MNL (Euro)"] = (
        sample_average_logit_probabilities.values
        * (priceEuro - solution_data.loc["Variable costs (Euro)"])
    )

    solution_data.loc["Market share, MXL (percent)"] = (
        100 * sample_average_mxl_probabilities.values
    )
    solution_data.loc["Market share, MNL (percent)"] = (
        100 * sample_average_logit_probabilities.values
    )

    solution_data.loc["Price (Euro)"] = priceEuro

    solution_data.loc["Contribution margin (Euro)"] = (
        priceEuro - solution_data.loc["Variable costs (Euro)"]
    )
    solution_data.loc["Markup"] = priceEuro / solution_data.loc["Variable costs (Euro)"]

    solution_data.loc["Sample mean energy cost savings (percent)"] = (
        100 * energyCostSavings
    )
    solution_data.loc["Sample mean CO2 savings (percent)"] = (
        100 * carbonDioxideEmissionReductions
    )

    solution_data = solution_data.reindex(
        [
            *solution_data.index[-9:-4],
            solution_data.index[0],
            solution_data.index[-4],
            solution_data.index[-3],
            solution_data.index[-2],
            solution_data.index[-1],
            *list(solution_data.index[1:-9]),
        ]
    )

    return solution_data


def calculate_consumer_preferences_cost_optimization_constant_contribution(
    fname_results_summary: str | os.PathLike,
    contribution,
    fname_heat,
    fname_electricity,
    data_id_socio_demographic_attributes,
    data_id_mxl_coefs,
    data_id_logit_coefs,
    data_id_heating_demands,
    data_id_electricity_demands,
    file_date_agents,
    file_date_sample,
    file_date_demands,
    file_date_logit_regression,
    num_agents,
    mxl_coefs_sample_size,
    matchingFactor=0.5,
):

    solution_data = pd.read_csv(
        fname_results_summary,
        index_col=0,
        comment="#",
    )

    priceEuro = solution_data.loc["Variable costs (Euro)"].to_numpy() + contribution

    thermalEfficiency = solution_data.loc["Thermal efficiency"].to_numpy()
    powerIndex = (
        solution_data.loc["Electrical power (kW)"].to_numpy()
        / solution_data.loc["Thermal power (kW)"].to_numpy()
    )
    matchingFactor = matchingFactor * np.ones(len(priceEuro))

    sample_average_mxl_probabilities = get_sampleAverageMixedLogitProbability(
        priceEuro,
        thermalEfficiency,
        powerIndex,
        matchingFactor,
        data_id_socio_demographic_attributes=data_id_socio_demographic_attributes,
        data_id_mxl_coefs=data_id_mxl_coefs,
        data_id_heating_demands=data_id_heating_demands,
        data_id_electricity_demands=data_id_electricity_demands,
        file_date_demands=file_date_demands,
        file_date_agents=file_date_agents,
        file_date_sample=file_date_sample,
        num_agents=num_agents,
        mxl_coefs_sample_size=mxl_coefs_sample_size,
    )

    sample_average_logit_probabilities = get_sampleAverageLogitProbability(
        priceEuro,
        thermalEfficiency,
        powerIndex,
        matchingFactor,
        data_id_socio_demographic_attributes=data_id_socio_demographic_attributes,
        data_id_logit_coefs=data_id_logit_coefs,
        data_id_heating_demands=data_id_heating_demands,
        data_id_electricity_demands=data_id_electricity_demands,
        file_date_demands=file_date_demands,
        file_date_agents=file_date_agents,
        file_date_logit_regression=file_date_logit_regression,
        num_agents=16,
    )

    energyCostSavings = np.full(len(thermalEfficiency), np.nan)
    for num in range(len(energyCostSavings)):
        energyCostSavings[num] = energyCostSavingsModel(
            thermalEfficiency=thermalEfficiency[num],
            powerIndex=powerIndex[num],
            matchingFactor=matchingFactor[num],
            fname_heat=fname_heat,
            fname_electricity=fname_electricity,
        ).mean()

    carbonDioxideEmissionReductions = np.full(len(thermalEfficiency), np.nan)
    for num in range(len(carbonDioxideEmissionReductions)):
        carbonDioxideEmissionReductions[num] = carbonDioxideEmissionReductionsModel(
            thermalEfficiency=thermalEfficiency[num],
            powerIndex=powerIndex[num],
            fname_heat=fname_heat,
            fname_electricity=fname_electricity,
        ).mean()

    solution_data.loc["Normalized total contribution, MXL (Euro)"] = (
        sample_average_mxl_probabilities.values
        * (priceEuro - solution_data.loc["Variable costs (Euro)"])
    )
    solution_data.loc["Normalized total contribution, MNL (Euro)"] = (
        sample_average_logit_probabilities.values
        * (priceEuro - solution_data.loc["Variable costs (Euro)"])
    )

    solution_data.loc["Market share, MXL (percent)"] = (
        100 * sample_average_mxl_probabilities.values
    )
    solution_data.loc["Market share, MNL (percent)"] = (
        100 * sample_average_logit_probabilities.values
    )

    solution_data.loc["Price (Euro)"] = priceEuro

    solution_data.loc["Contribution margin (Euro)"] = (
        priceEuro - solution_data.loc["Variable costs (Euro)"]
    )
    solution_data.loc["Markup"] = priceEuro / solution_data.loc["Variable costs (Euro)"]

    solution_data.loc["Sample mean energy cost savings (percent)"] = (
        100 * energyCostSavings
    )
    solution_data.loc["Sample mean CO2 savings (percent)"] = (
        100 * carbonDioxideEmissionReductions
    )

    solution_data = solution_data.reindex(
        [
            *solution_data.index[-9:-4],
            solution_data.index[0],
            solution_data.index[-4],
            solution_data.index[-3],
            solution_data.index[-2],
            solution_data.index[-1],
            *list(solution_data.index[1:-9]),
        ]
    )

    return solution_data
