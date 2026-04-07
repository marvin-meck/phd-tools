import pyomo.environ as pyo
from phdtools.optimization import EPSILON

from phdtools.optimization.pyomo._reformer_block import reformer_block_rule
from phdtools.optimization.pyomo._shift_block import shift_block_rule
from phdtools.optimization.pyomo._fuel_cell_block import fuel_cell_block_rule


def add_sets(model):
    model.setCompounds = pyo.Set()
    model.setCompoundsIn = pyo.Set(within=model.setCompounds)
    model.setCompoundsOut = pyo.Set(within=model.setCompounds)
    model.setCompoundsAir = pyo.Set(within=model.setCompounds)
    model.costCoefIndex = pyo.Set()


def add_parameters(model):
    # Paramter declarations
    model.costCoef = pyo.Param(model.costCoefIndex)

    model.variableCostLowerBoundEuro = pyo.Param()
    model.variableCostUpperBoundEuro = pyo.Param()

    model.INDIRECT_COSTS_EURO = pyo.Param()

    model.methaneConversionLowerBound = pyo.Param()

    model.NOMINAL_THERMAL_POWER_SI = pyo.Param()

    model.FUEL_CELL_POWER_SI_LB = pyo.Param()
    model.FUEL_CELL_POWER_SI_UB = pyo.Param()

    model.OXYGEN_TO_CARBON_RATIO_LB = pyo.Param()
    model.OXYGEN_TO_CARBON_RATIO_UB = pyo.Param()

    model.REFORMER_STEAM_TO_CARBON_LB = pyo.Param()
    model.REFORMER_STEAM_TO_CARBON_UB = pyo.Param()

    model.FUEL_CELL_CARBON_MONOXIDE_TOLERANCE = pyo.Param()

    model.pressureSI = pyo.Param()

    model.stoichiometricNumber = pyo.Param(
        model.setCompoundsIn | model.setCompoundsOut,
        doc="Stoichiometric number for the combustion of methane",
    )

    model.moleFractionsAir = pyo.Param(model.setCompoundsAir)
    model.vapourPressureGasOutSI = pyo.Param()

    model.stdEnthalpyInSI = pyo.Param(model.setCompoundsIn)
    model.stdEnthalpyOutSI = pyo.Param(model.setCompoundsOut)

    # model.netCalorificValueMethaneSI = pyo.Param()
    model.grossCalorificValueMethaneSI = pyo.Param()

    model.molarFlowRateInLowerBoundSI = pyo.Param(model.setCompoundsIn)
    model.molarFlowRateInUpperBoundSI = pyo.Param(model.setCompoundsIn)


def add_variables(model):
    # Variable declarations
    def molar_flow_rate_in_bounds_rule(block, k):
        return (
            block.molarFlowRateInLowerBoundSI[k] / block.molarFlowRateInUpperBoundSI[k],
            1.0,
        )

    model.molarFlowRateInScaled = pyo.Var(
        model.setCompoundsIn,
        domain=pyo.NonNegativeReals,
        bounds=molar_flow_rate_in_bounds_rule,
        doc="molarFlowRateScaled[i,j]: molar flow rate of compound i normalized",
    )

    model.electricalPowerScaled = pyo.Var(
        bounds=(model.FUEL_CELL_POWER_SI_LB / model.FUEL_CELL_POWER_SI_UB, 1.0),
        within=pyo.PositiveReals,
    )

    def variable_costs_bound_rule(block):
        return block.variableCostLowerBoundEuro / block.variableCostUpperBoundEuro, 1.0

    model.variableCostsScaled = pyo.Var(
        bounds=variable_costs_bound_rule, domain=pyo.NonNegativeReals
    )


def add_expressions(model):

    @model.Expression(model.setCompoundsIn)
    def molarFlowRateInSI(block, k):
        return block.molarFlowRateInUpperBoundSI[k] * block.molarFlowRateInScaled[k]

    @model.Expression(model.setCompoundsOut)
    def molarFlowRateOutSI(block, k):
        if k == "C1O2(g)":
            expr = (
                -1
                * block.stoichiometricNumber["C1O2(g)"]
                / block.stoichiometricNumber["C1H4(g)"]
                * block.molarFlowRateInSI["C1H4(g)"]
            )
        elif k == "H2O1(g)":
            expr = (
                block.vapourPressureGasOutSI
                / block.pressureSI
                * sum(block.molarFlowRateInSI[j] for j in block.setCompoundsIn)
            )
        elif k == "H2O1(l)":
            expr = (
                -1
                * block.stoichiometricNumber["H2O1(g)"]
                / block.stoichiometricNumber["C1H4(g)"]
                * block.molarFlowRateInSI["C1H4(g)"]  # product water
                + block.molarFlowRateInSI["H2O1(l)"]  # steam reforming
                + block.molarFlowRateInSI["H2O1(g)"]  # feed air
            ) - block.vapourPressureGasOutSI / block.pressureSI * sum(
                block.molarFlowRateInSI[j] for j in block.setCompoundsIn
            )
        elif k == "N2(ref)":
            expr = block.molarFlowRateInSI["N2(ref)"]
        elif k == "O2(ref)":
            expr = (
                block.molarFlowRateInSI["O2(ref)"]
                - block.stoichiometricNumber["O2(ref)"]
                / block.stoichiometricNumber["C1H4(g)"]
                * block.molarFlowRateInSI["C1H4(g)"]
            )
        else:
            raise ValueError(f"Invalid out stream {k}")

        return expr

    @model.Expression()
    def molarFlowRateProductWaterOutSI(block):
        return (
            -1
            * block.stoichiometricNumber["H2O1(g)"]
            / block.stoichiometricNumber["C1H4(g)"]
            * block.molarFlowRateInSI["C1H4(g)"]
        )

    @model.Expression()
    def electricalPowerSI(block):
        return block.FUEL_CELL_POWER_SI_UB * block.electricalPowerScaled

    @model.Expression()
    def thermalEfficiency(block):
        return (
            -1
            * block.NOMINAL_THERMAL_POWER_SI
            / (block.molarFlowRateInSI["C1H4(g)"] * block.grossCalorificValueMethaneSI)
        )

    @model.Expression()
    def inverseThermalEfficiency(block):
        return (
            -1 * block.grossCalorificValueMethaneSI / block.NOMINAL_THERMAL_POWER_SI
        ) * block.molarFlowRateInSI["C1H4(g)"]

    @model.Expression()
    def electricalEfficiency(block):
        return (
            -1
            * block.electricalPowerSI
            / (block.molarFlowRateInSI["C1H4(g)"] * block.grossCalorificValueMethaneSI)
        )

    @model.Expression()
    def powerIndex(block):
        return block.electricalPowerSI / block.NOMINAL_THERMAL_POWER_SI

    @model.Expression()
    def variableCostsExpression(block):
        return (
            block.costCoef["R1", "a1"]
            + block.costCoef["R1", "a2"]
            * block.reformer.massCatalystSI ** block.costCoef["R1", "k"]
            + block.costCoef["R2", "a1"]
            + block.costCoef["R2", "a2"]
            * block.shift.massCatalystSI ** block.costCoef["R2", "k"]
            + block.costCoef["FC", "a1"] * block.fuel_cell.totalActiveAreaSI
            + block.INDIRECT_COSTS_EURO
        )

    @model.Expression()
    def variableCostsEuro(block):
        return block.variableCostUpperBoundEuro * block.variableCostsScaled


def add_constraints(model):

    @model.Constraint()
    def variable_costs_constraint(block):
        return block.variableCostsExpression <= block.variableCostsEuro

    @model.Constraint(doc="Heat balance, see eq. (3.22)")
    def heat_balance(block):
        return (
            sum(
                block.molarFlowRateInSI[j] * block.stdEnthalpyInSI[j]
                for j in block.setCompoundsIn
            )
            - sum(
                block.molarFlowRateOutSI[j] * block.stdEnthalpyOutSI[j]
                for j in block.setCompoundsOut
            )
            == block.NOMINAL_THERMAL_POWER_SI + block.electricalPowerSI
        )

    @model.Constraint(doc="Lower bound on the steam-to-carbon ratio, see eq. (3.31)")
    def reformer_steam_to_carbon_ratio_lower_bound(block):
        return (
            block.REFORMER_STEAM_TO_CARBON_LB * block.molarFlowRateInSI["C1H4(g)"]
            <= block.molarFlowRateInSI["H2O1(l)"]
        )

    @model.Constraint(doc="Upper bound on the steam-to-carbon ratio, see eq. (3.31)")
    def reformer_steam_to_carbon_ratio_upper_bound(block):
        return (
            block.molarFlowRateInSI["H2O1(l)"]
            <= block.REFORMER_STEAM_TO_CARBON_UB * block.molarFlowRateInSI["C1H4(g)"]
        )

    @model.Constraint(
        doc="Upper bound on the carbon monoxide concentration, see eq. (3.34)"
    )
    def fuel_cell_carbon_monoxide_upper_bound(block):
        return block.shift.molarFlowRateSI[
            "C1O1(g)", block.shift.setTimeSteps.last()
        ] <= block.FUEL_CELL_CARBON_MONOXIDE_TOLERANCE * sum(
            block.shift.molarFlowRateSI[j, block.shift.setTimeSteps.last()]
            for j in block.shift.setReactingCompounds
        )

    @model.Constraint(
        model.setCompoundsAir,
        doc="Air composition, see eqs. (3.39)",
    )
    def air_composition(block, k):
        return block.molarFlowRateInSI[k] == block.moleFractionsAir[k] * sum(
            block.molarFlowRateInSI[j] for j in block.setCompoundsAir
        )

    @model.Constraint(doc="Lower bound on the oxygen-to-carbon ratio")
    def oxygen_to_carbon_ratio_lowre_bound(block):
        return (
            block.OXYGEN_TO_CARBON_RATIO_LB * block.molarFlowRateInSI["C1H4(g)"]
            <= block.molarFlowRateInSI["O2(ref)"]
        )

    @model.Constraint(doc="Upper bound on the oxygen-to-carbon ratio")
    def oxygen_to_carbon_ratio_upper_bound(block):
        return (
            block.molarFlowRateInSI["O2(ref)"]
            <= block.OXYGEN_TO_CARBON_RATIO_UB * block.molarFlowRateInSI["C1H4(g)"]
        )

    @model.Constraint(model.setCompounds, doc="Inlet conditions reformer.")
    def inlet_conditions_reformer(block, k):
        if k == "C1H4(g)":
            expr = (
                block.reformer.molarFlowRateSI[
                    "C1H4(g)", block.reformer.setTimeSteps.first()
                ]
                == block.molarFlowRateInSI["C1H4(g)"]
            )
        elif k == "H2O1(g)":
            expr = (
                block.reformer.molarFlowRateSI[
                    "H2O1(g)", block.reformer.setTimeSteps.first()
                ]
                == block.molarFlowRateInSI["H2O1(l)"]
            )
        elif k in {"C1H4(g)", "C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"}:
            # expr = block.reformer.molarFlowRateSI[k, block.reformer.setTimeSteps.first()] <= block.reformer.molarFlowRateUpperBoundSI[k, block.reformer.setTimeSteps.first()]*EPSILON
            expr = (
                block.reformer.molarFlowRateScaled[
                    k, block.reformer.setTimeSteps.first()
                ]
                <= EPSILON
            )
        else:
            expr = pyo.Constraint.Skip
        return expr

    @model.Constraint()
    def minimum_conversion_reformer(block):
        return (
            block.reformer.molarFlowRateSI[
                "C1H4(g)", block.reformer.setTimeSteps.last()
            ]
            <= (1 - block.methaneConversionLowerBound)
            * block.reformer.molarFlowRateSI[
                "C1H4(g)", block.reformer.setTimeSteps.first()
            ]
        )

    @model.Constraint(model.setCompounds, doc="Inlet conditions shift.")
    def inlet_conditions_shift(block, k):
        if k in {"C1H4(g)", "C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"}:
            expr = (
                block.shift.molarFlowRateSI[k, block.shift.setTimeSteps.first()]
                == block.reformer.molarFlowRateSI[k, block.reformer.setTimeSteps.last()]
            )
        else:
            expr = pyo.Constraint.Skip
        return expr

    @model.Constraint()
    def fuel_cell_mass_balance(block):
        expr = block.shift.molarFlowRateSI[
            "H2(ref)", block.shift.setTimeSteps.last()
        ] >= (
            block.fuel_cell.totalChargeTransferRateSI
            / (2 * block.fuel_cell.FARADAY_CONST_SI)
        )
        return expr

    @model.Constraint()
    def fuel_cell_power(block):
        return (
            block.electricalPowerSI
            == block.fuel_cell.powerDensitySI * block.fuel_cell.totalActiveAreaSI
        )

    @model.Constraint()
    def postive_liquid_water_balance(block):
        return block.molarFlowRateInSI["H2O1(l)"] <= block.molarFlowRateOutSI["H2O1(l)"]

    # Valid inequalities
    @model.Constraint(model.setCompoundsOut)
    def non_negativity_molar_flow_rate_out(block, k):
        return block.molarFlowRateOutSI[k] >= 0


def BaseModel(name):
    model = pyo.AbstractModel(name)

    add_sets(model)
    add_parameters(model)
    add_variables(model)

    model.reformer = pyo.Block(rule=reformer_block_rule)
    model.shift = pyo.Block(rule=shift_block_rule)
    model.fuel_cell = pyo.Block(rule=fuel_cell_block_rule)

    add_expressions(model)
    add_constraints(model)

    return model
