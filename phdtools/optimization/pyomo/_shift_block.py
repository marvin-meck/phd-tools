from enum import Enum

import pyomo.environ as pyo
import numpy as np

from phdtools.data.constants import GAS_CONST_SI

from phdtools.optimization import (
    EPSILON,
    NOMINAL_THERMAL_POWER_SI,
    GROSS_CALORIFIC_VALUE_METHANE_SI,
    REFORMER_STEAM_TO_CARBON_UB,
    THERMAL_EFFICIENCY_LB,
    SHIFT_TEMPERATURE_SI_LB,
    SHIFT_TEMPERATURE_SI_UB,
    SHIFT_MASS_OF_SOLIDS_GRAM_LB,
    SHIFT_MASS_OF_SOLIDS_GRAM_UB,
)

steamReformingCompounds = Enum(
    "Compound", ["C1H4(g)", "C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"], start=0
)

from phdtools.models.mendes_2010 import (
    STST_PRESSURE_BAR,
    stoichiometricNumber,
    rateConstModel,
    equilibriumConstModel,
    adsorptionCoefModel,
    ModelParameters as WaterGasShiftParameters,
)

NUM_FINITE_ELEMENTS = 10

TEMPERATURE_SCALE = 1000
MASS_CATALYST_SCALE = 1e-3
MOLAR_FLOW_RATE_SCALE = 1e-2

PARAMS = WaterGasShiftParameters.init(model="LH1")


def shift_block_rule(block):

    block.NUM_FINITE_ELEMENTS = pyo.Param()

    block.setTimeSteps = pyo.RangeSet(0, block.NUM_FINITE_ELEMENTS, 1)

    block.setReactingCompounds = pyo.Set(
        doc="The set of reactive chemical compounds in the water-gas shift reactions."
    )

    # block.setInertCompounds = pyo.Set(
    #     initialize=[],
    #     doc="The set of inert chemical compounds."
    # )

    block.setCompoundsAdsorption = pyo.Set(
        within=block.setReactingCompounds,
        doc="The set of compounds with associated adsorption constants.",
    )

    # Parmater declarations
    block.STST_PRESSURE_BAR = pyo.Param(doc="Standard state pressure in bar.")

    block.GAS_CONST_SI = pyo.Param(doc="The molar gas constant in J/(K*mol).")

    block.stoichiometricNumber = pyo.Param(
        block.setReactingCompounds,
    )

    block.pressureBar = pyo.Param(
        mutable=True,
        doc="The absolute pressure in bar.",
    )

    block.rateConstantFactorSI = pyo.Param(
        doc="Pre-exponential factor of the rate constant k0 in mol/(s kg(cat)) taken from Mendes (2010a)",
    )

    block.activationEnergySI = pyo.Param(
        doc="Activation energies E[i] in J/mol taken from Mendes (2010a)",
    )

    block.refTemperatureEquilibriumSI = pyo.Param(
        doc="Reference temperature Tref[i] of the equilibrium constants Keq[i] in Kelvin taken from NIST JANAF Thermochemical tables"
    )

    block.equilibriumConstRef = pyo.Param(
        doc="Equilibrium constants Keq[i] taken from NIST JANAF Thermochemical tables",
    )

    block.enthalpyReactionSI = pyo.Param(
        doc="Enthalpy changes of reaction dH[i] in J/mol taken from NIST JANAF Thermochemical tables",
    )

    block.adsorptionCoefFactor = pyo.Param(
        block.setCompoundsAdsorption,
        doc="Pre-exponential factor of the adsorption constants K[i] taken from Mendes (2010a)",
    )

    block.enthalpyAdsorptionSI = pyo.Param(
        block.setCompoundsAdsorption,
        doc="Enthalpy changes of adsorption dH[i] in J/mol taken from Mendes (2010a)",
    )

    # block.molarFlowRateInertCompoundsSI = pyo.Param(
    #     block.setInertCompounds,
    #     initialize={}
    # )

    # Variables
    block.temperatureLowerBoundSI = pyo.Param()
    block.temperatureUpperBoundSI = pyo.Param()

    block.temperatureScaled = pyo.Var(
        bounds=(
            pyo.value(block.temperatureLowerBoundSI / block.temperatureUpperBoundSI),
            1.0,
        ),
        domain=pyo.PositiveReals,
        doc="Reaction temperature",
    )

    @block.Expression(doc="Reaction temperature in K")
    def temperatureKelvin(block):
        return block.temperatureScaled * block.temperatureUpperBoundSI

    # Temperature denpendent variables
    block.rateConstLowerBound = pyo.Param()
    block.rateConstUpperBound = pyo.Param()

    block.rateConstantScaled = pyo.Var(
        bounds=(block.rateConstLowerBound / block.rateConstUpperBound, 1.0),
        domain=pyo.Reals,
    )

    @block.Expression()
    def rateConstant(block):
        return block.rateConstUpperBound * block.rateConstantScaled

    block.equilibriumConstLowerBound = pyo.Param()
    block.equilibriumConstUpperBound = pyo.Param()

    block.equilibriumConstScaled = pyo.Var(
        bounds=(
            block.equilibriumConstLowerBound / block.equilibriumConstUpperBound,
            1.0,
        ),
        domain=pyo.PositiveReals,
    )

    @block.Expression()
    def equilibriumConst(block):
        return block.equilibriumConstUpperBound * block.equilibriumConstScaled

    block.adsorptionCoefLowerBound = pyo.Param(block.setCompoundsAdsorption)

    block.adsorptionCoefUpperBound = pyo.Param(block.setCompoundsAdsorption)

    def adsorption_coef_bounds_rule(block, k):
        return (
            block.adsorptionCoefLowerBound[k] / block.adsorptionCoefUpperBound[k],
            1.0,
        )

    block.adsorptionCoefScaled = pyo.Var(
        block.setCompoundsAdsorption,
        bounds=adsorption_coef_bounds_rule,
        domain=pyo.Reals,
    )

    @block.Expression(block.setCompoundsAdsorption)
    def adsorptionCoef(block, k):
        return block.adsorptionCoefUpperBound[k] * block.adsorptionCoefScaled[k]

    # Design variable
    block.massCatalystLowerBoundSI = pyo.Param()
    block.massCatalystUpperBoundSI = pyo.Param()

    block.massCatalystScaled = pyo.Var(
        bounds=(block.massCatalystLowerBoundSI / block.massCatalystUpperBoundSI, 1.0),
        domain=pyo.PositiveReals,
        doc="The mass of solids",
    )

    @block.Expression(doc="The mass of solids in kg")
    def massCatalystSI(block):
        return block.massCatalystScaled * block.massCatalystUpperBoundSI

    # Discretized variables
    block.molarFlowRateUpperBoundSI = pyo.Param(
        block.setReactingCompounds * block.setTimeSteps
    )

    block.molarFlowRateScaled = pyo.Var(
        block.setReactingCompounds * block.setTimeSteps,
        bounds=(EPSILON, 1.0),
        domain=pyo.PositiveReals,
        doc="molarFlowRate[k,t]: Molar flow rate of species k at discretization point t",
    )

    @block.Expression(
        block.setReactingCompounds,
        block.setTimeSteps,
        doc="molarFlowRate[k,t]: Molar flow rate species k at discretization point t in mol/s of ",
    )
    def molarFlowRateSI(block, k, t):
        return block.molarFlowRateUpperBoundSI[k, t] * block.molarFlowRateScaled[k, t]

    block.activity = pyo.Var(
        block.setReactingCompounds * (block.setTimeSteps - {0}),
        bounds=(EPSILON, 1.0),
        domain=pyo.PositiveReals,
        doc="activity[k,t]: Activity species k at discretization point t",
    )

    @block.Expression(block.setTimeSteps - {0})
    def derivMolarFlowRateLowerBoundSI(block, t):
        h = (1 - 0) / block.NUM_FINITE_ELEMENTS
        return (
            block.molarFlowRateUpperBoundSI["C1O1(g)", t]
            * (
                block.molarFlowRateScaled["C1O1(g)", t].lb
                - block.molarFlowRateScaled["C1O1(g)", t - 1].ub
            )
            / h
        )

    @block.Expression(block.setTimeSteps - {0})
    def derivMolarFlowRateUpperBoundSI(block, t):
        h = (1 - 0) / block.NUM_FINITE_ELEMENTS
        return (
            block.molarFlowRateUpperBoundSI["C1O1(g)", t]
            * (
                block.molarFlowRateScaled["C1O1(g)", t].ub
                - block.molarFlowRateScaled["C1O1(g)", t - 1].lb
            )
            / h
        )

    def deriv_molar_flow_rate_bounds_rule(block, t):
        lb = block.derivMolarFlowRateLowerBoundSI[t]
        ub = block.derivMolarFlowRateUpperBoundSI[t]
        return pyo.value(lb / ub), 1.0

    block.derivMolarFlowRateScaled = pyo.Var(
        block.setTimeSteps - {0},
        bounds=deriv_molar_flow_rate_bounds_rule,
        doc="derivMolarFlowRate[k,t]: Derivative of the molar flow rate (normalized) of species k at discretization point t with respect to the dimensionless mass of solids.",
    )

    @block.Expression(
        block.setTimeSteps - {0},
        doc="derivMolarFlowRate[k,t]: Derivative of the molar flow rate in mol/s of species k at discretization point t with respect to the dimensionless mass of solids.",
    )
    def derivMolarFlowRateSI(block, t):
        return (
            block.derivMolarFlowRateUpperBoundSI[t] * block.derivMolarFlowRateScaled[t]
        )

    @block.Expression(
        block.setTimeSteps - {0},
    )
    def formationRateLowerBoundSI(block, t):
        a = pyo.value(
            block.derivMolarFlowRateLowerBoundSI[t] / block.massCatalystLowerBoundSI
        )
        b = pyo.value(
            block.derivMolarFlowRateLowerBoundSI[t] / block.massCatalystUpperBoundSI
        )

        if a <= b:
            val = a
        else:
            val = b

        return val

    @block.Expression(
        block.setTimeSteps - {0},
    )
    def formationRateUpperBoundSI(block, t):
        a = pyo.value(
            block.derivMolarFlowRateUpperBoundSI[t] / block.massCatalystLowerBoundSI
        )
        b = pyo.value(
            block.derivMolarFlowRateUpperBoundSI[t] / block.massCatalystUpperBoundSI
        )

        if a <= b:
            val = b
        else:
            val = a

        return val

    def formation_rate_bounds_rule(block, t):
        lb = block.formationRateLowerBoundSI[t]
        ub = block.formationRateUpperBoundSI[t]
        return (pyo.value(lb / ub), 1.0)

    block.formationRateScaled = pyo.Var(
        block.setTimeSteps - {0},
        bounds=formation_rate_bounds_rule,
        domain=pyo.Reals,
        doc="formationRateSI[k,t]: Formation rate in mol/(kg cat s) of species k at discretization point t",
    )

    @block.Expression(
        block.setTimeSteps - {0},
    )
    def formationRateSI(block, t):
        return block.formationRateUpperBoundSI[t] * block.formationRateScaled[t]

    # Constraint declarations

    @block.Constraint(block.setReactingCompounds, block.setTimeSteps)
    def activityConstr(block, k, t):
        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        totalMolarFlowRateSI = sum(
            block.molarFlowRateSI[j, t] for j in block.setReactingCompounds
        )  # + \
        # sum( block.setInertCompounds[j] for j in block.setInertCompounds )

        return (
            block.activity[k, t] * totalMolarFlowRateSI
            == block.pressureBar / block.STST_PRESSURE_BAR * block.molarFlowRateSI[k, t]
        )

    @block.Constraint(block.setTimeSteps)
    def reactionRateConstr(block, t):
        """
        TODO
        """
        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        DEN = (
            1
            + block.adsorptionCoef["C1O1(g)"] * block.activity["C1O1(g)", t]
            + block.adsorptionCoef["H2O1(g)"] * block.activity["H2O1(g)", t]
            + block.adsorptionCoef["C1O2(g)"] * block.activity["C1O2(g)", t]
            + block.adsorptionCoef["H2(ref)"] * block.activity["H2(ref)", t]
        ) ** 2

        r = (
            block.rateConstant
            * (
                block.activity["C1O1(g)", t] * block.activity["H2O1(g)", t]
                - block.activity["C1O2(g)", t]
                * block.activity["H2(ref)", t]
                / block.equilibriumConst
            )
            / DEN
        )

        return block.formationRateSI[t] == -r

    @block.Constraint()
    def equilibriumConstConstr(block):
        return block.equilibriumConst == block.equilibriumConstRef * pyo.exp(
            -block.enthalpyReactionSI
            / block.GAS_CONST_SI
            * (
                block.temperatureKelvin ** (-1)
                - block.refTemperatureEquilibriumSI ** (-1)
            )
        )

    @block.Constraint()
    def rateConstantConstr(block):
        return block.rateConstant == block.rateConstantFactorSI * pyo.exp(
            -block.activationEnergySI / (block.GAS_CONST_SI * block.temperatureKelvin)
        )

    @block.Constraint(block.setCompoundsAdsorption)
    def adsorptionCoefConstr(block, k):
        if k not in {"C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"}:
            return pyo.Constraint.Skip

        return block.adsorptionCoef[k] == block.adsorptionCoefFactor[k] * pyo.exp(
            -block.enthalpyAdsorptionSI[k]
            / (block.GAS_CONST_SI * block.temperatureKelvin)
        )

    @block.Constraint(block.setReactingCompounds, block.setTimeSteps)
    def stoichiometry(block, k, t):
        """dF_i / \nu_i = dF_{CO} / \nu_{CO}"""

        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        if k == "C1O1(g)":
            return pyo.Constraint.Skip

        return block.molarFlowRateSI[k, t] - block.molarFlowRateSI[
            k, t - 1
        ] == block.stoichiometricNumber[k] / block.stoichiometricNumber["C1O1(g)"] * (
            block.molarFlowRateSI["C1O1(g)", t]
            - block.molarFlowRateSI["C1O1(g)", t - 1]
        )
        # return (
        #     block.derivMolarFlowRateSI[k, t]
        #     == block.stoichiometricNumber[k]/block.stoichiometricNumber["C1O1(g)"] * block.derivMolarFlowRateSI["C1O1(g)", t]
        # )

    @block.Constraint(block.setTimeSteps)
    def backward_euler_discretization(block, t):

        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        expr = None
        h = (1 - 0) / NUM_FINITE_ELEMENTS

        scale = 1 / MOLAR_FLOW_RATE_SCALE  # * 1/h

        # dF[t+1] / dW = F[t+1] - F[t] / h
        expr = scale * h * block.derivMolarFlowRateSI[t] == scale * (
            block.molarFlowRateSI["C1O1(g)", t]
            - block.molarFlowRateSI["C1O1(g)", t - 1]
        )

        return expr

    @block.Constraint(block.setTimeSteps)
    def pfrPerformanceEquation(block, t):

        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        return (
            block.derivMolarFlowRateSI[t]
            == block.massCatalystSI * block.formationRateSI[t]
        )

    return block


def pyomo_create_model(**kwargs):

    model = pyo.AbstractModel("Feasibility problem: shift reactor")
    model.shift = pyo.Block(rule=shift_block_rule)

    model.setCompounds = pyo.Set(
        # initialize=[c.name for c in steamReformingCompounds],
        doc="The set of reactive chemical compounds in the steam methane reforming and water-gas shift reactions."
    )

    model.molarFlowRateInSI = pyo.Param(model.setCompounds)

    @model.Constraint(model.setCompounds)
    def inlet_conditions(model, k):
        return (
            model.shift.molarFlowRateScaled[k, model.shift.setTimeSteps.first()]
            == model.molarFlowRateInSI[k]
        )

    @model.Objective(sense=pyo.minimize)
    def obj(model):
        return 0

    return model
