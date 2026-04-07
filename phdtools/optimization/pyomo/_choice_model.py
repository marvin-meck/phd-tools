import pyomo.environ as pyo


def _add_expressions_exlanatory_variables(model):

    model.GAS_PRICE_EUR_PER_KWH = pyo.Param()
    model.ELECTRICITY_PRICE_EUR_PER_KWH = pyo.Param()
    model.THERMAL_EFFICIENCY_STATUS_QUO = pyo.Param()
    model.CO2_EMISSION_FACTOR_MOL_PER_KWH = pyo.Param()

    model.annualHeatDemandSI = pyo.Param(model.setAgents)
    model.annualElectricityDemandSI = pyo.Param(model.setAgents)

    model.priceLowerBoundEuro = pyo.Param()
    model.priceUpperBoundEuro = pyo.Param()

    model.MATCHING_FACTOR = pyo.Param()

    model.selfSufficiency = pyo.Var(
        model.setAgents, bounds=(0, 1.0), within=pyo.NonNegativeReals
    )

    @model.Constraint(model.setAgents)
    def self_sufficiency_net_consumer_bound_constr(block, n):
        return (
            block.selfSufficiency[n]
            <= block.MATCHING_FACTOR
            * block.powerIndex
            * block.annualHeatDemandSI[n]
            / block.annualElectricityDemandSI[n]
        )

    @model.Constraint(model.setAgents)
    def self_sufficiency_net_producer_bound_constr(block, n):
        return block.selfSufficiency[n] <= block.MATCHING_FACTOR * 1.0

    model.priceScaled = pyo.Var(
        bounds=(model.priceLowerBoundEuro / model.priceUpperBoundEuro, 1.0),
        within=pyo.NonNegativeReals,
    )

    @model.Expression()
    def priceEuro(block):
        return block.priceUpperBoundEuro * block.priceScaled

    @model.Constraint()
    def icost_price_constr(block):
        return block.investmentCostsEuro == 1.3 * block.priceEuro

    @model.Constraint(model.setAgents)
    def csav_constr(block, n):
        den = (
            1e-3
            / 3600
            * block.GAS_PRICE_EUR_PER_KWH
            * block.annualHeatDemandSI[n]
            / block.THERMAL_EFFICIENCY_STATUS_QUO
            + 1e-3
            / 3600
            * block.ELECTRICITY_PRICE_EUR_PER_KWH
            * block.annualElectricityDemandSI[n]
        )
        return block.energyCostSavings[n] == (
            1
            / den
            * (
                1e-3
                / 3600
                * block.GAS_PRICE_EUR_PER_KWH
                * block.annualHeatDemandSI[n]
                * (
                    1 / block.THERMAL_EFFICIENCY_STATUS_QUO
                    - block.inverseThermalEfficiency
                )
                + 1e-3
                / 3600
                * block.ELECTRICITY_PRICE_EUR_PER_KWH
                * block.selfSufficiency[n]
                * block.annualElectricityDemandSI[n]
                + 1e-3
                / 3600
                * block.FEED_IN_TARIFF_EUR_PER_KWH
                * (
                    block.powerIndex * block.annualHeatDemandSI[n]
                    - block.selfSufficiency[n] * block.annualElectricityDemandSI[n]
                )
            )
        )

    @model.Constraint(model.setAgents)
    def co2sav_constr(block, n):
        den = (
            -1
            * block.annualHeatDemandSI[n]
            / (block.THERMAL_EFFICIENCY_STATUS_QUO * block.grossCalorificValueMethaneSI)
            + 1e-3
            / 3600
            * block.CO2_EMISSION_FACTOR_MOL_PER_KWH
            * block.annualElectricityDemandSI[n]
        )
        return block.carbonDioxideEmissionReductions[n] == (
            1
            / den
            * (
                -1
                * block.annualHeatDemandSI[n]
                / block.grossCalorificValueMethaneSI
                * (
                    1 / block.THERMAL_EFFICIENCY_STATUS_QUO
                    - block.inverseThermalEfficiency
                )
                + 1e-3
                / 3600
                * block.CO2_EMISSION_FACTOR_MOL_PER_KWH
                * block.powerIndex
                * block.annualHeatDemandSI[n]
            )
        )


def _add_logit_model(model):
    model.setAgents = pyo.Set()

    model.logitVariablesIndex = pyo.Set()
    model.socioDemographicAttributesIndex = pyo.Set()

    model.logitCoefs = pyo.Param(model.logitVariablesIndex)
    model.socioDemographicAttributes = pyo.Param(
        model.setAgents * model.socioDemographicAttributesIndex
    )

    model.FEED_IN_TARIFF_EUR_PER_KWH = pyo.Param()
    model.INVESTMENT_TYPE = pyo.Param()
    model.CONTRACT_DURATION_YEARS = pyo.Param()

    model.energyCostSavingsLowerBound = pyo.Param(model.setAgents)
    model.energyCostSavingsUpperBound = pyo.Param(model.setAgents)

    model.carbonDioxideEmissionReductionsLowerBound = pyo.Param(model.setAgents)
    model.carbonDioxideEmissionReductionsUpperBound = pyo.Param(model.setAgents)

    def investment_costs_bounds_rule(block):
        lb = 1.3 * block.priceLowerBoundEuro
        ub = 1.3 * block.priceUpperBoundEuro
        return lb / ub, 1.0

    model.investmentCostsScaled = pyo.Var(
        bounds=investment_costs_bounds_rule,
        within=pyo.NonNegativeReals,
    )

    def energy_cost_savings_bounds_rule(block, n):
        return (
            block.energyCostSavingsLowerBound[n] / block.energyCostSavingsUpperBound[n],
            1.0,
        )

    model.energyCostSavingsScaled = pyo.Var(
        model.setAgents, bounds=energy_cost_savings_bounds_rule, within=pyo.Reals
    )

    def carbon_dioxide_emissions_reductions_bounds_rule(block, n):
        return (
            block.carbonDioxideEmissionReductionsLowerBound[n]
            / block.carbonDioxideEmissionReductionsUpperBound[n],
            1.0,
        )

    model.carbonDioxideEmissionReductionsScaled = pyo.Var(
        model.setAgents,
        bounds=carbon_dioxide_emissions_reductions_bounds_rule,
        within=pyo.Reals,
    )

    model.marketShare = pyo.Var(bounds=(0, 1), within=pyo.NonNegativeReals)

    @model.Expression()
    def investmentCostsEuro(block):
        return 1.3 * block.priceUpperBoundEuro * block.investmentCostsScaled

    @model.Expression(model.setAgents)
    def energyCostSavings(block, n):
        return block.energyCostSavingsUpperBound[n] * block.energyCostSavingsScaled[n]

    @model.Expression(model.setAgents)
    def carbonDioxideEmissionReductions(block, n):
        return (
            block.carbonDioxideEmissionReductionsUpperBound[n]
            * block.carbonDioxideEmissionReductionsScaled[n]
        )

    @model.Expression()
    def icost(block):
        return 1e-3 * block.investmentCostsEuro

    @model.Expression(model.setAgents)
    def csav(block, n):
        return 100 * block.energyCostSavings[n]

    @model.Expression(model.setAgents)
    def co2sav(block, n):
        return 10 * block.carbonDioxideEmissionReductions[n]

    model.FIT = pyo.Expression(expr=100 * model.FEED_IN_TARIFF_EUR_PER_KWH)
    model.ITYPE = pyo.Expression(expr=model.INVESTMENT_TYPE)
    model.DUR = pyo.Expression(expr=model.CONTRACT_DURATION_YEARS)

    # model.observedUtility = pyo.Var(model.setAgents)

    @model.Expression(model.setAgents, model.logitVariablesIndex)
    def logitVariables(block, n, k):
        expr = None
        if k == "ICOST":
            expr = block.icost
        elif k == "CSAV":
            expr = block.csav[n]
        elif k == "CO2SAV":
            expr = block.co2sav[n]
        elif k == "FIT":
            expr = block.FIT
        elif k == "ITYPE":
            expr = block.ITYPE
        elif k == "DUR":
            expr = block.DUR
        elif k == "ICOST x HEATSYS":
            expr = block.icost * block.socioDemographicAttributes[n, "HEATSYS"]
        elif k == "CO2SAV x AGE":
            expr = block.co2sav[n] * block.socioDemographicAttributes[n, "AGE"]
        elif k == "CSAV x AGE":
            expr = block.csav[n] * block.socioDemographicAttributes[n, "AGE"]
        elif k == "ITYPE x AGE":
            expr = block.ITYPE * block.socioDemographicAttributes[n, "AGE"]
        elif k == "DUR x SEX":
            expr = block.DUR * block.socioDemographicAttributes[n, "SEX"]
        elif k == "FIT x SEX":
            expr = block.FIT * block.socioDemographicAttributes[n, "SEX"]
        elif k == "FIT x FLATSIZE":
            expr = block.FIT * block.socioDemographicAttributes[n, "FLATSIZE"]
        elif k == "ASC":
            expr = -1
        else:
            raise ValueError(f"Undefined variable with index {k}.")
        return expr

    # @model.Constraint(model.setAgents)
    # def observed_utiltiy_constr(block, n):
    #     return block.observedUtility[n] == sum(
    #         block.logitCoefs[j] * block.logitVariables[n, j]
    #         for j in block.logitVariablesIndex
    #     )

    @model.Expression(model.setAgents)
    def observedUtility(block, n):
        return sum(
            block.logitCoefs[j] * block.logitVariables[n, j]
            for j in block.logitVariablesIndex
        )

    @model.Expression(model.setAgents)
    def logitProbability(block, n):
        return 1 / (1 + pyo.exp(-1 * block.observedUtility[n]))

    @model.Expression()
    def sampleAverageChoiceProbabilityExpression(block):
        return (
            1
            / len(block.setAgents)
            * sum(block.logitProbability[n] for n in block.setAgents)
        )

    @model.Constraint()
    def market_share_constr(block):
        return block.marketShare <= block.sampleAverageChoiceProbabilityExpression


def add_consumer_preference_model(model):
    _add_logit_model(model)
    _add_expressions_exlanatory_variables(model)
