"""phdtools.optimization.preprocessing.py

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


References
----------

Gebäudeenergiegesetz (2020) BGBl. I 2020, S. 1728. Available at:
    https://www.gesetze-im-internet.de/geg/.

Rommel, K. and Sagebiel, J. (2017) 'Preferences for micro-cogeneration in
    Germany: Policy implications for grid expansion from a discrete choice
    experiment', Applied Energy, 206, pp. 612–622. Available at:
    https://doi.org/10.1016/j.apenergy.2017.08.216.

Statistical Offices Of The Federation And The Länder (2025) "Dwellings:
    Year of construction (microcensus categories)/Rooms - Floor area of the
    dwelling (10m² increments) - Dwellings in the building."
    https://ergebnisse.zensus2022.de: Zensus Datenbank. Available at:
    https://ergebnisse.zensus2022.de/datenbank/online/url/48ed0498
    (Accessed: January 12, 2026).

"""

from enum import Enum
import os
from typing import Dict

import json
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar, minimize_scalar

from phdtools.data import (
    Compound,
    ISO_STD_REF_PRESSURE_SI,
    ISO_STD_REF_TEMPERATURE_SI,
    ISO_STD_REF_REL_HUMIDITY,
)
from phdtools.data.thermochemical import stoichiometricNumber
from phdtools.data.thermophysical import vapourPressureModel

from phdtools.models.rommel_sagebiel_2017 import (
    MEAN_FLATSIZE,
    compute_sample_avg_mxl_prob,
    compute_sample_avg_logit_prob,
)

from phdtools.models.meck_2025 import (
    ModelParameters as FuelCellParameters,
    fuelCellVoltageModel,
)

from phdtools.optimization import (
    EPSILON,
    FEED_IN_TARIFF_EUR_PER_KWH,
    INVESTMENT_TYPE,
    CONTRACT_DURATION_YEARS,
    GAS_PRICE_EUR_PER_KWH,
    ELECTRICITY_PRICE_EUR_PER_KWH,
    THERMAL_EFFICIENCY_STATUS_QUO,
    GROSS_CALORIFIC_VALUE_METHANE_SI,
    CO2_EMISSION_FACTOR_MOL_PER_KWH,
    ISO_STD_REF_TEMPERATURE_SI,
    FUEL_CELL_POWER_SI_LB,
    FUEL_CELL_POWER_SI_UB,
    REFORMER_BULK_DENSITY_CATALYST_SI,
    REFORMER_VOID_FRACTION_CATALYST_BED,
    REFORMER_TUBE_DIAMETER_SI,
    REFORMER_TUBE_WALL_THICKNESS_SI,
    SHIFT_MASS_OF_SOLIDS_GRAM_UB,
    SHIFT_BULK_DENSITY_CATALYST_SI,
    SHIFT_VOID_FRACTION_CATALYST_BED,
    SHIFT_TUBE_DIAMETER_SI,
    SHIFT_TUBE_WALL_THICKNESS_SI,
)

from phdtools.optimization import (
    EPSILON,
    NOMINAL_THERMAL_POWER_SI,
    GROSS_CALORIFIC_VALUE_METHANE_SI,
    THERMAL_EFFICIENCY_LB,
    THERMAL_EFFICIENCY_UB,
    REFORMER_STEAM_TO_CARBON_LB,
    REFORMER_STEAM_TO_CARBON_UB,
    OXYGEN_TO_CARBON_RATIO_LB,
    OXYGEN_TO_CARBON_RATIO_UB,
    REFORMER_MASS_OF_SOLIDS_GRAM_LB,
    REFORMER_MASS_OF_SOLIDS_GRAM_UB,
    REFORMER_BULK_DENSITY_CATALYST_SI,
    REFORMER_VOID_FRACTION_CATALYST_BED,
    REFORMER_TUBE_DIAMETER_SI,
    REFORMER_TUBE_WALL_THICKNESS_SI,
    SHIFT_TEMPERATURE_SI_LB,
    SHIFT_TEMPERATURE_SI_UB,
    SHIFT_BULK_DENSITY_CATALYST_SI,
    SHIFT_VOID_FRACTION_CATALYST_BED,
    SHIFT_TUBE_DIAMETER_SI,
    SHIFT_TUBE_WALL_THICKNESS_SI,
    FUEL_CELL_PRICE_EURO_PER_SQUARE_METER,
    FUEL_CELL_TOTAL_ACTIVE_AREA_SI_LB,
    FUEL_CELL_TOTAL_ACTIVE_AREA_SI_UB,
    INDIRECT_COSTS_EURO,
    MATCHING_FACTOR,
)


SPECIFIC_SPACE_HEATING_DEMAND_SI_PER_YEAR = (
    130.0 * 1e3 * 3600
)  # (see, Statistical Offices Of The Federation And The Länder, 2025)

SPECIFIC_WATER_HEATING_DEMAND_SI_PER_YEAR = (
    12.5 * 1e3 * 3600
)  # (see, Gebäudeenergiegesetz, 2020)

ELECTRICITY_PRICE_SURVEY_YEAR_EURO_PER_KWH = 0.3  # estimate

Streams = Enum(
    "Streams",
    [
        "methane_in",
        "water_in",
        "steam_out",
        "steam_methane_mix",
        "reformer_in",
        "reformer_out",
        "shift_in",
        "shift_out",
        "anode_in",
        "anode_out",
        "burner_in",
        "burner_out",
        "air_in",
        "air_to_burner",
        "cathode_in",
        "cathode_out",
        "condenser_in",
        "condenser_out",
        "water_out",
        "gas_out",
    ],
    start=0,
)


def annualHeatDemandModel(fname_s):
    """annualHeatDemandModel

    Inputs:
    -------
        fname_s: file name of the socio-demographic sample
    """

    s = pd.read_csv(
        fname_s,
        comment="#",
        index_col=0,
    )

    mask = s["HEATSYS"] == 1
    socio_demographic_attributes = s[mask]

    annualHeatDemandSI = (
        (
            SPECIFIC_SPACE_HEATING_DEMAND_SI_PER_YEAR
            + SPECIFIC_WATER_HEATING_DEMAND_SI_PER_YEAR
        )
        * (socio_demographic_attributes["FLATSIZE"] + MEAN_FLATSIZE)
    ).rename("annualHeatDemandSI")

    return annualHeatDemandSI


def annualElectricityDemandModel(fname_s):

    s = pd.read_csv(
        fname_s,
        comment="#",
        index_col=0,
    )

    mask = s["HEATSYS"] == 1
    socio_demographic_attributes = s[mask]

    electricityCostStatusQuoSurveyYear = (
        12 * 15 * socio_demographic_attributes["ENECOST"]
    )

    annualElectricityDemandSI = (
        3600
        * 1e3
        * electricityCostStatusQuoSurveyYear
        / ELECTRICITY_PRICE_SURVEY_YEAR_EURO_PER_KWH
    ).rename("annualElectricityDemandSI")

    return annualElectricityDemandSI


def icostEuroModel(priceEuro):
    return 1.3 * priceEuro


def heatingCostSavingsModel(thermalEfficiency, annualHeatDemandSI):
    heatingCostSavings = (
        (GAS_PRICE_EUR_PER_KWH * 1e-3 / 3600)
        * annualHeatDemandSI
        * (THERMAL_EFFICIENCY_STATUS_QUO ** (-1) - thermalEfficiency ** (-1))
    )
    return heatingCostSavings


def energyCostSavingsModel(
    thermalEfficiency: float,
    powerIndex: float,
    matchingFactor: float,
    fname_heat: str | os.PathLike,
    fname_electricity: str | os.PathLike,
):
    """
    Docstring for energyCostSavingsModel

    :param thermalEfficiency: thermal efficiency of the cogeneration unit
    :type thermalEfficiency: float
    :param powerIndex: power index of the cogeneration unit power index := electrical efficiency / thermal efficiency
    :type powerIndex: float
    :param fname_heat: Description
    :type fname_heat: str | os.PathLike
    :param fname_electricity: Description
    :type fname_electricity: str | os.PathLike

    """
    if np.isscalar(thermalEfficiency):
        thermalEfficiency = pd.Series(thermalEfficiency)
        thermalEfficiency.index.name = "ALTERNATIVE"

    if np.isscalar(powerIndex):
        powerIndex = pd.Series(powerIndex)
        powerIndex.index.name = "ALTERNATIVE"

    if np.isscalar(matchingFactor):
        matchingFactor = pd.Series(matchingFactor)
        matchingFactor.index.name = "ALTERNATIVE"

    annualHeatDemandSI = pd.read_csv(
        fname_heat,
        comment="#",
        index_col=0,
    )["annualHeatDemandSI"]

    annualElectricityDemandSI = pd.read_csv(
        fname_electricity,
        comment="#",
        index_col=0,
    )["annualElectricityDemandSI"]

    index = pd.MultiIndex.from_product(
        [annualHeatDemandSI.index, powerIndex.index], names=["AGENT", "ALTERNATIVE"]
    )

    upperBoundNetConsumer = pd.Series(
        np.outer((annualHeatDemandSI / annualElectricityDemandSI), powerIndex).ravel(),
        index=index,
    )

    upperBoundNetProducer = pd.Series(np.ones(len(upperBoundNetConsumer)), index=index)

    upperBound = pd.concat([upperBoundNetConsumer, upperBoundNetProducer], axis=1).min(
        axis=1
    )

    selfSufficiency = upperBound.mul(matchingFactor, level="ALTERNATIVE")

    heatingCostStatusQuo = (
        (GAS_PRICE_EUR_PER_KWH * 1e-3 / 3600)
        * annualHeatDemandSI
        / THERMAL_EFFICIENCY_STATUS_QUO
    )
    electricityCostsStatusQuo = (
        ELECTRICITY_PRICE_EUR_PER_KWH * 1e-3 / 3600
    ) * annualElectricityDemandSI

    heatingCostSavings = pd.Series(
        np.outer(
            (GAS_PRICE_EUR_PER_KWH * 1e-3 / 3600) * annualHeatDemandSI,
            THERMAL_EFFICIENCY_STATUS_QUO ** (-1) - thermalEfficiency ** (-1),
        ).ravel(),
        index=index,
    )

    electricityProduction = pd.Series(
        np.outer(annualHeatDemandSI, powerIndex).ravel(), index=index
    )

    electricityCostSavings = (
        ELECTRICITY_PRICE_EUR_PER_KWH * 1e-3 / 3600
    ) * selfSufficiency.mul(annualElectricityDemandSI, level="AGENT") + (
        FEED_IN_TARIFF_EUR_PER_KWH * 1e-3 / 3600
    ) * (
        electricityProduction
        - selfSufficiency.mul(annualElectricityDemandSI, level="AGENT")
    )

    ret_val = (heatingCostSavings + electricityCostSavings).div(
        heatingCostStatusQuo + electricityCostsStatusQuo, level="AGENT"
    )

    return ret_val


def carbonDioxideEmissionReductionsModel(
    thermalEfficiency: float,
    powerIndex: float,
    fname_heat: str | os.PathLike,
    fname_electricity: str | os.PathLike,
):
    """
    Docstring for carbonDioxideEmissionReductionsModel

    :param thermalEfficiency: thermal efficiency of the cogeneration unit
    :type thermalEfficiency: float
    :param powerIndex: power index of the cogeneration unit power index := electrical efficiency / thermal efficiency
    :type powerIndex: float
    :param fname_heat: Description
    :type fname_heat: str | os.PathLike
    :param fname_electricity: Description
    :type fname_electricity: str | os.PathLike

    """
    if np.isscalar(thermalEfficiency):
        thermalEfficiency = pd.Series(thermalEfficiency)
        thermalEfficiency.index.name = "ALTERNATIVE"

    if np.isscalar(powerIndex):
        powerIndex = pd.Series(powerIndex)
        powerIndex.index.name = "ALTERNATIVE"

    annualHeatDemandSI = pd.read_csv(
        fname_heat,
        comment="#",
        index_col=0,
    )["annualHeatDemandSI"]

    annualElectricityDemandSI = pd.read_csv(
        fname_electricity,
        comment="#",
        index_col=0,
    )["annualElectricityDemandSI"]

    index = pd.MultiIndex.from_product(
        [annualHeatDemandSI.index, powerIndex.index], names=["AGENT", "ALTERNATIVE"]
    )

    co2EmissionsStatusQuo = (
        -1
        * annualHeatDemandSI
        / (THERMAL_EFFICIENCY_STATUS_QUO * GROSS_CALORIFIC_VALUE_METHANE_SI)
        + (CO2_EMISSION_FACTOR_MOL_PER_KWH * 1e-3 / 3600) * annualElectricityDemandSI
    )

    electricityProduction = pd.Series(
        np.outer(annualHeatDemandSI, powerIndex).ravel(), index=index
    )

    co2EmissionsReductionsThermalEfficiency = pd.Series(
        np.outer(
            -1 * annualHeatDemandSI / GROSS_CALORIFIC_VALUE_METHANE_SI,
            THERMAL_EFFICIENCY_STATUS_QUO ** (-1) - thermalEfficiency ** (-1),
        ).ravel(),
        index=index,
    )

    co2EmissionsReductions = (
        co2EmissionsReductionsThermalEfficiency
        + (CO2_EMISSION_FACTOR_MOL_PER_KWH * 1e-3 / 3600) * electricityProduction
    )

    ret_val = co2EmissionsReductions / co2EmissionsStatusQuo

    return ret_val


def get_attributes_alternative(
    designs,
    data_id_socio_demographic_attributes,
    data_id_heating_demands,
    data_id_electricity_demands,
    file_date,
    file_date_agents,
    sample_size,
    # nominalThermalPowerSI=NOMINAL_THERMAL_POWER_SI,
    fit=100 * FEED_IN_TARIFF_EUR_PER_KWH,
    itype=INVESTMENT_TYPE,
    dur=CONTRACT_DURATION_YEARS,
):

    socio_demographic_attributes = pd.read_csv(
        data_id_socio_demographic_attributes.get_path(fail_exists=False)
        / f"{file_date_agents}_socio_demographic_attributes_{sample_size}.csv",
        comment="#",
        index_col=0,
    )

    mask = socio_demographic_attributes["HEATSYS"] == 1
    socio_demographic_attributes = socio_demographic_attributes[mask]

    index = pd.MultiIndex.from_product(
        [socio_demographic_attributes.index, designs.index],
        names=["AGENT", "ALTERNATIVE"],
    )

    frame = pd.DataFrame(
        index=index,
        # columns=["ICOST", "CSAV", "CO2SAV", "FIT", "ITYPE", "DUR"]
    )

    icost = 1e-3 * icostEuroModel(designs["PRICE_EURO"]).rename("ICOST")

    csav = 100 * energyCostSavingsModel(
        thermalEfficiency=designs["THERMAL_EFFICIENCY"],
        powerIndex=designs["POWER_INDEX"],
        matchingFactor=designs["MATCHING_FACTOR"],
        fname_heat=(
            data_id_heating_demands.get_path(fail_exists=False)
            / f"{file_date}_annual_heating_demands_{sample_size}.csv"
        ),
        fname_electricity=(
            data_id_electricity_demands.get_path(fail_exists=False)
            / f"{file_date}_annual_electricity_demands_{sample_size}.csv"
        ),
    ).rename("CSAV")

    co2sav = 10 * carbonDioxideEmissionReductionsModel(
        thermalEfficiency=designs["THERMAL_EFFICIENCY"],
        powerIndex=designs["POWER_INDEX"],
        fname_heat=(
            data_id_heating_demands.get_path(fail_exists=False)
            / f"{file_date}_annual_heating_demands_{sample_size}.csv"
        ),
        fname_electricity=(
            data_id_electricity_demands.get_path(fail_exists=False)
            / f"{file_date}_annual_electricity_demands_{sample_size}.csv"
        ),
    ).rename("CO2SAV")

    frame = pd.merge(
        frame.reset_index(), icost.reset_index(), on="ALTERNATIVE", how="right"
    ).set_index(["AGENT", "ALTERNATIVE"])
    frame["CSAV"] = csav
    frame["CO2SAV"] = co2sav

    frame["FIT"] = fit
    frame["ITYPE"] = itype
    frame["DUR"] = dur

    return frame


def get_plugFlowReactorCostCoefficients(
    bulkDensityCatalysSI,
    voidFractionCatalystBed,
    tubeDiameterSI,
    tubeWallThicknessSI,
    fname_coefs,
    _type="Fixed-tube-sheet heat exchanger",
    _subtype="Carbon-steel tubes",
):

    with open(fname_coefs) as f:
        costCoefficients = json.load(f)

    coefs = costCoefficients[_type][_subtype]["Single variable model with constant"]

    coefs["Ansatz"] = "$c/c_0 = a_1 + a_2 \\, (x/x0)^k$"

    a1 = coefs["a1"]
    a2 = coefs["a2"]
    k = coefs["k"]

    x0 = coefs["x0"]

    coefs["a2"] = (
        a2
        * (
            1
            / x0
            * (4 * tubeDiameterSI / (tubeDiameterSI - tubeWallThicknessSI) ** 2)
            * bulkDensityCatalysSI ** (-1)
            / (1 - voidFractionCatalystBed)
        )
        ** k
    )

    return coefs


def get_currentDensityUpperBound(params: FuelCellParameters):

    # E - jRs = 0 --> j = E / Rs
    tmp = np.min(
        [
            params.reversibleCellPotentialSI / params.areaSpecificResistanceSI,
            params.limitingCurrentDensitySI,
        ]
    )

    r = root_scalar(fuelCellVoltageModel, bracket=[EPSILON, tmp], args=(params))
    if r.converged:
        val = r.root
    else:
        val = params.limitingCurrentDensitySI

    return val


def get_powerDensityUpperBound(
    params: FuelCellParameters, currentDensityUpperBoundSI=None
):

    if currentDensityUpperBoundSI is None:
        tmp = np.min(
            [
                params.reversibleCellPotentialSI / params.areaSpecificResistanceSI,
                params.limitingCurrentDensitySI,
            ]
        )
    else:
        tmp = currentDensityUpperBoundSI

    negativePowerDensity = lambda j: -1 * (j * fuelCellVoltageModel(j, params))

    sol = minimize_scalar(
        negativePowerDensity,
        bounds=(EPSILON, tmp),
    )

    if sol.success:
        val = -1 * sol.fun
    else:
        val = None

    return val


def get_moleFractionsAir(
    temperatureKelvin=ISO_STD_REF_TEMPERATURE_SI,
    pressureSI=ISO_STD_REF_PRESSURE_SI,
    relHumidity=ISO_STD_REF_REL_HUMIDITY,
    Compound=Compound,
):

    dryMoleFractionAir = 0 * np.ones(len(Compound))
    dryMoleFractionAir[Compound["O2(ref)"].value] = 0.209476  # see ISO 2533:1975
    dryMoleFractionAir[Compound["N2(ref)"].value] = (
        1 - dryMoleFractionAir[Compound["O2(ref)"].value]
    )

    moleFractionAir = 0 * np.ones(len(Compound))

    vapourPressureSI = vapourPressureModel(temperatureKelvin)[0]

    moleFractionAir[Compound["H2O1(g)"].value] = (
        relHumidity * vapourPressureSI / pressureSI
    )
    for c in Compound:
        if not c.name == "H2O1(g)":
            moleFractionAir[c.value] = dryMoleFractionAir[c.value] * (
                1 - moleFractionAir[Compound["H2O1(g)"].value]
            )

    return moleFractionAir


def get_molarFlowRateBoundsSI():

    molarFlowRateLowerBound = np.full((len(Compound)), np.nan)
    molarFlowRateUpperBound = np.full((len(Compound)), np.nan)

    moleFractionsAir = get_moleFractionsAir()

    # Methane feed
    molarFlowRateLowerBound[Compound["C1H4(g)"].value] = (
        THERMAL_EFFICIENCY_UB ** (-1)
        * NOMINAL_THERMAL_POWER_SI
        / np.abs(GROSS_CALORIFIC_VALUE_METHANE_SI)
    )

    molarFlowRateUpperBound[Compound["C1H4(g)"].value] = (
        THERMAL_EFFICIENCY_LB ** (-1)
        * NOMINAL_THERMAL_POWER_SI
        / np.abs(GROSS_CALORIFIC_VALUE_METHANE_SI)
    )

    # Steam feed
    molarFlowRateLowerBound[Compound["H2O1(l)"].value] = (
        REFORMER_STEAM_TO_CARBON_LB * molarFlowRateUpperBound[Compound["C1H4(g)"].value]
    )

    molarFlowRateUpperBound[Compound["H2O1(l)"].value] = (
        REFORMER_STEAM_TO_CARBON_UB * molarFlowRateUpperBound[Compound["C1H4(g)"].value]
    )

    # Oxygen feed
    molarFlowRateLowerBound[Compound["O2(ref)"].value] = (
        OXYGEN_TO_CARBON_RATIO_LB
        / 2
        * (
            molarFlowRateLowerBound[Compound["C1H4(g)"].value]
            * stoichiometricNumber.loc["O2(ref)", "MCR1"]
            / stoichiometricNumber.loc["C1H4(g)", "MCR1"]
        )
    )

    molarFlowRateUpperBound[Compound["O2(ref)"].value] = (
        OXYGEN_TO_CARBON_RATIO_UB
        / 2
        * (
            molarFlowRateUpperBound[Compound["C1H4(g)"].value]
            * stoichiometricNumber.loc["O2(ref)", "MCR1"]
            / stoichiometricNumber.loc["C1H4(g)", "MCR1"]
        )
    )

    for k in {"N2(ref)", "H2O1(g)"}:
        molarFlowRateLowerBound[Compound[k].value] = (
            moleFractionsAir[Compound[k].value]
            / moleFractionsAir[Compound["O2(ref)"].value]
            * molarFlowRateLowerBound[Compound["O2(ref)"].value]
        )
        molarFlowRateUpperBound[Compound[k].value] = (
            moleFractionsAir[Compound[k].value]
            / moleFractionsAir[Compound["O2(ref)"].value]
            * molarFlowRateUpperBound[Compound["O2(ref)"].value]
        )

    return molarFlowRateLowerBound, molarFlowRateUpperBound


def get_variableCostsBounds(fname_coefs: str | os.PathLike):

    costCoefficientsReformer = get_plugFlowReactorCostCoefficients(
        bulkDensityCatalysSI=REFORMER_BULK_DENSITY_CATALYST_SI,
        voidFractionCatalystBed=REFORMER_VOID_FRACTION_CATALYST_BED,
        tubeDiameterSI=REFORMER_TUBE_DIAMETER_SI,
        tubeWallThicknessSI=REFORMER_TUBE_WALL_THICKNESS_SI,
        fname_coefs=fname_coefs,
    )

    costCoefficientsShift = get_plugFlowReactorCostCoefficients(
        bulkDensityCatalysSI=SHIFT_BULK_DENSITY_CATALYST_SI,
        voidFractionCatalystBed=SHIFT_VOID_FRACTION_CATALYST_BED,
        tubeDiameterSI=SHIFT_TUBE_DIAMETER_SI,
        tubeWallThicknessSI=SHIFT_TUBE_WALL_THICKNESS_SI,
        fname_coefs=fname_coefs,
    )

    singleVariableCostModel = lambda W, coef: coef["a1"] + coef["a2"] * W ** coef["k"]

    variableCostLowerBoundEuro = (
        singleVariableCostModel(
            REFORMER_MASS_OF_SOLIDS_GRAM_LB * 1e-3, costCoefficientsReformer
        )
        + singleVariableCostModel(SHIFT_TEMPERATURE_SI_LB * 1e-3, costCoefficientsShift)
        + FUEL_CELL_PRICE_EURO_PER_SQUARE_METER * FUEL_CELL_TOTAL_ACTIVE_AREA_SI_LB
        + INDIRECT_COSTS_EURO
    )

    variableCostUpperBoundEuro = (
        singleVariableCostModel(
            REFORMER_MASS_OF_SOLIDS_GRAM_UB * 1e-3, costCoefficientsReformer
        )
        + singleVariableCostModel(SHIFT_TEMPERATURE_SI_UB * 1e-3, costCoefficientsShift)
        + FUEL_CELL_PRICE_EURO_PER_SQUARE_METER * FUEL_CELL_TOTAL_ACTIVE_AREA_SI_UB
        + INDIRECT_COSTS_EURO
    )

    return variableCostLowerBoundEuro, variableCostUpperBoundEuro


def get_energyCostSavingsLowerBound(
    fname_heat: str | os.PathLike, fname_electricity: str | os.PathLike
) -> Dict[int | str, float]:

    lb = energyCostSavingsModel(
        thermalEfficiency=THERMAL_EFFICIENCY_LB,
        powerIndex=FUEL_CELL_POWER_SI_LB / NOMINAL_THERMAL_POWER_SI,
        matchingFactor=MATCHING_FACTOR,
        fname_heat=fname_heat,
        fname_electricity=fname_electricity,
    ).droplevel("ALTERNATIVE")

    return lb.to_dict()


def get_energyCostSavingsUpperBound(
    fname_heat: str | os.PathLike, fname_electricity: str | os.PathLike
) -> Dict[int | str, float]:

    ub = energyCostSavingsModel(
        thermalEfficiency=THERMAL_EFFICIENCY_UB,
        powerIndex=FUEL_CELL_POWER_SI_UB / NOMINAL_THERMAL_POWER_SI,
        matchingFactor=MATCHING_FACTOR,
        fname_heat=fname_heat,
        fname_electricity=fname_electricity,
    ).droplevel("ALTERNATIVE")

    return ub.to_dict()


def get_carbonDioxideEmissionReductionsLowerBound(
    fname_heat: str | os.PathLike, fname_electricity: str | os.PathLike
) -> Dict[int | str, float]:

    lb = carbonDioxideEmissionReductionsModel(
        thermalEfficiency=THERMAL_EFFICIENCY_LB,
        powerIndex=FUEL_CELL_POWER_SI_LB / NOMINAL_THERMAL_POWER_SI,
        fname_heat=fname_heat,
        fname_electricity=fname_electricity,
    ).droplevel("ALTERNATIVE")

    return lb.to_dict()


def get_carbonDioxideEmissionReductionsUpperBound(
    fname_heat: str | os.PathLike, fname_electricity: str | os.PathLike
) -> Dict[int | str, float]:

    ub = carbonDioxideEmissionReductionsModel(
        thermalEfficiency=THERMAL_EFFICIENCY_UB,
        powerIndex=FUEL_CELL_POWER_SI_UB / NOMINAL_THERMAL_POWER_SI,
        fname_heat=fname_heat,
        fname_electricity=fname_electricity,
    ).droplevel("ALTERNATIVE")

    return ub.to_dict()


def get_fuelCellCostValues():
    fuelCellAreaRangeSI = np.linspace(
        FUEL_CELL_TOTAL_ACTIVE_AREA_SI_LB, FUEL_CELL_TOTAL_ACTIVE_AREA_SI_UB
    )
    fuelCellCostValuesEuro = FUEL_CELL_PRICE_EURO_PER_SQUARE_METER * fuelCellAreaRangeSI
    df = pd.DataFrame(
        {
            "fuelCellAreaRangeSI": fuelCellAreaRangeSI,
            "fuelCellCostValuesEuro": fuelCellCostValuesEuro,
        }
    )
    return df


def get_reactorCostValues(fname_cost_coefs):

    def singleVariableCostModel(x, a, b, c):
        return a + b * x**c

    catalystMassRangeSI = 1e-3 * np.geomspace(
        EPSILON, SHIFT_MASS_OF_SOLIDS_GRAM_UB, 250
    )

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

    costValuesReformer = singleVariableCostModel(
        catalystMassRangeSI,
        costCoefficientsReformer["a1"],
        costCoefficientsReformer["a2"],
        costCoefficientsReformer["k"],
    )

    costValuesShift = singleVariableCostModel(
        catalystMassRangeSI,
        costCoefficientsShift["a1"],
        costCoefficientsShift["a2"],
        costCoefficientsShift["k"],
    )

    df = pd.DataFrame(
        {
            "catalystMassSI": catalystMassRangeSI,
            "Reformer": costValuesReformer,
            "Shift": costValuesShift,
        }
    )

    return df
