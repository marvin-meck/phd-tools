import pyomo.environ as pyo

from phdtools.optimization import (
    EPSILON,
)


def reformer_block_rule(block):

    block.NUM_FINITE_ELEMENTS = pyo.Param()

    block.setTimeSteps = pyo.RangeSet(0, block.NUM_FINITE_ELEMENTS, 1)

    block.setReactingCompounds = pyo.Set(
        doc="The set of reactive chemical compounds in the steam methane reforming and water-gas shift reactions."
    )

    block.setSteamReformingReactions = pyo.Set(doc="The set of reactions considered.")

    block.setIndependentComponents = pyo.Set(
        within=block.setReactingCompounds,
        doc="The set of independent compounds or *degrees of freedom*. Follows from the stoichiometry.",
    )

    block.setCompoundsAdsorption = pyo.Set(
        within=block.setReactingCompounds,
        doc="The set of compounds with associated adsorption constants.",
    )

    # Parmater declarations
    block.STST_PRESSURE_BAR = pyo.Param(doc="Standard state pressure in bar.")

    block.GAS_CONST_SI = pyo.Param(doc="The molar gas constant in J/(K*mol).")

    block.stoichiometricNumber = pyo.Param(
        block.setSteamReformingReactions * block.setReactingCompounds,
    )

    block.pressureBar = pyo.Param(
        mutable=True,
        doc="The absolute pressure in bar.",
    )

    block.refTemperatureEquilibriumSI = pyo.Param(
        block.setSteamReformingReactions,
        doc="Reference temperature Tref[i] of the equilibrium constants Keq[i] in Kelvin taken from NIST JANAF Thermochemical tables",
    )

    block.equilibriumConstRef = pyo.Param(
        block.setSteamReformingReactions,
        doc="Equilibrium constants Keq[i] taken from NIST JANAF Thermochemical tables",
    )

    block.enthalpyReactionSI = pyo.Param(
        block.setSteamReformingReactions,
        doc="Enthalpy changes of reaction dH[i] in J/mol taken from NIST JANAF Thermochemical tables",
    )

    block.refTemperatureRateSI = pyo.Param(
        block.setSteamReformingReactions,
        doc="Reference temperature Tref[i] of the rate constants k[i] in Kelvin taken from Xu and Froment (1989) Tab. 5",
    )

    block.rateConstantRefSI = pyo.Param(
        block.setSteamReformingReactions,
        doc="Rate constants k[i] in mol / (kg(cat) s) taken from Xu and Froment (1989) Tab. 5",
    )

    block.activationEnergySI = pyo.Param(
        block.setSteamReformingReactions,
        doc="Activation energies E[i] in J/mol taken from Xu and Froment (1989) Tab. 5",
    )

    block.refTemperatureAdsorptionSI = pyo.Param(
        block.setCompoundsAdsorption,
        doc="Reference temperature Tref[i] of the adsorption Constants K[i] in Kelvin taken from Xu and Froment (1989) Tab. 5",
    )

    block.adsorptionCoefRef = pyo.Param(
        block.setCompoundsAdsorption,
        doc="Adsorption constants K[i] taken from Xu and Froment (1989) Tab. 5",
    )

    block.enthalpyAdsorptionSI = pyo.Param(
        block.setCompoundsAdsorption,
        doc="Enthalpy changes of adsorption dH[i] in J/mol taken from Xu and Froment (1989) Tab. 5",
    )

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
    block.rateConstLowerBound = pyo.Param(block.setSteamReformingReactions)
    block.rateConstUpperBound = pyo.Param(block.setSteamReformingReactions)

    def rate_const_bounds_rule(block, r):
        return (block.rateConstLowerBound[r] / block.rateConstUpperBound[r], 1.0)

    block.rateConstantScaled = pyo.Var(
        block.setSteamReformingReactions,
        bounds=rate_const_bounds_rule,
        domain=pyo.Reals,
    )

    @block.Expression(block.setSteamReformingReactions)
    def rateConstant(block, r):
        return block.rateConstUpperBound[r] * block.rateConstantScaled[r]

    block.equilibriumConstLowerBound = pyo.Param(block.setSteamReformingReactions)
    block.equilibriumConstUpperBound = pyo.Param(block.setSteamReformingReactions)

    def equilibrium_const_bounds_rule(block, r):
        return (
            block.equilibriumConstLowerBound[r] / block.equilibriumConstUpperBound[r],
            1.0,
        )

    block.equilibriumConstScaled = pyo.Var(
        block.setSteamReformingReactions,
        bounds=equilibrium_const_bounds_rule,
        domain=pyo.Reals,
    )

    @block.Expression(block.setSteamReformingReactions)
    def equilibriumConst(block, r):
        return block.equilibriumConstUpperBound[r] * block.equilibriumConstScaled[r]

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

    @block.Expression(block.setIndependentComponents * (block.setTimeSteps - {0}))
    def derivMolarFlowRateLowerBoundSI(block, k, t):
        h = (1 - 0) / block.NUM_FINITE_ELEMENTS
        return (
            block.molarFlowRateUpperBoundSI[k, t]
            * (
                block.molarFlowRateScaled[k, t].lb
                - block.molarFlowRateScaled[k, t - 1].ub
            )
            / h
        )

    @block.Expression(block.setIndependentComponents * (block.setTimeSteps - {0}))
    def derivMolarFlowRateUpperBoundSI(block, k, t):
        h = (1 - 0) / block.NUM_FINITE_ELEMENTS
        return (
            block.molarFlowRateUpperBoundSI[k, t]
            * (
                block.molarFlowRateScaled[k, t].ub
                - block.molarFlowRateScaled[k, t - 1].lb
            )
            / h
        )

    def deriv_molar_flow_rate_bounds_rule(block, k, t):
        lb = block.derivMolarFlowRateLowerBoundSI[k, t]
        ub = block.derivMolarFlowRateUpperBoundSI[k, t]
        return pyo.value(lb / ub), 1.0

    block.derivMolarFlowRateScaled = pyo.Var(
        block.setIndependentComponents * (block.setTimeSteps - {0}),
        bounds=deriv_molar_flow_rate_bounds_rule,
        doc="derivMolarFlowRate[k,t]: Derivative of the molar flow rate (normalized) of species k at discretization point t with respect to the dimensionless mass of solids.",
    )

    @block.Expression(
        block.setIndependentComponents * (block.setTimeSteps - {0}),
        doc="derivMolarFlowRate[k,t]: Derivative of the molar flow rate in mol/s of species k at discretization point t with respect to the dimensionless mass of solids.",
    )
    def derivMolarFlowRateSI(block, k, t):
        return (
            block.derivMolarFlowRateUpperBoundSI[k, t]
            * block.derivMolarFlowRateScaled[k, t]
        )

    @block.Expression(
        block.setIndependentComponents * (block.setTimeSteps - {0}),
    )
    def formationRateLowerBoundSI(block, k, t):
        # block.derivMolarFlowRateSI[k, t]
        #   = block.massCatalystSI * block.formationRateSI[k, t]
        #   >= block.derivMolarFlowRateLowerBoundSI[k,t]
        # --> block.formationRateSI[k, t] >= inf(block.derivMolarFlowRateLowerBoundSI[k,t] / block.massCatalystSI)
        a = pyo.value(
            block.derivMolarFlowRateLowerBoundSI[k, t] / block.massCatalystLowerBoundSI
        )
        b = pyo.value(
            block.derivMolarFlowRateLowerBoundSI[k, t] / block.massCatalystUpperBoundSI
        )

        if a <= b:
            val = a
        else:
            val = b

        return val

    @block.Expression(
        block.setIndependentComponents * (block.setTimeSteps - {0}),
    )
    def formationRateUpperBoundSI(block, k, t):
        # block.derivMolarFlowRateSI[k, t]
        #   = block.massCatalystSI * block.formationRateSI[k, t]
        #   <= block.derivMolarFlowRateUpperBoundSI[k,t]
        # --> block.formationRateSI[k, t] <= sup(block.derivMolarFlowRateLowerBoundSI[k,t] / block.massCatalystSI)

        a = pyo.value(
            block.derivMolarFlowRateUpperBoundSI[k, t] / block.massCatalystLowerBoundSI
        )
        b = pyo.value(
            block.derivMolarFlowRateUpperBoundSI[k, t] / block.massCatalystUpperBoundSI
        )

        if a <= b:
            val = b
        else:
            val = a

        return val

    def formation_rate_bounds_rule(block, k, t):
        lb = block.formationRateLowerBoundSI[k, t]
        ub = block.formationRateUpperBoundSI[k, t]
        return (pyo.value(lb / ub), 1.0)

    block.formationRateScaled = pyo.Var(
        block.setIndependentComponents * (block.setTimeSteps - {0}),
        bounds=formation_rate_bounds_rule,
        domain=pyo.Reals,
        doc="formationRateSI[k,t]: Formation rate in mol/(kg cat s) of species k at discretization point t",
    )

    @block.Expression(
        block.setIndependentComponents * (block.setTimeSteps - {0}),
    )
    def formationRateSI(block, k, t):
        return block.formationRateUpperBoundSI[k, t] * block.formationRateScaled[k, t]

    # Expressions
    @block.Expression(
        block.setIndependentComponents,
        block.setReactingCompounds - block.setIndependentComponents,
    )
    def stoichiometyCoefficient(block, i, k):
        den = (
            block.stoichiometricNumber["SMR", "C1H4(g)"]
            * block.stoichiometricNumber["WGS", "C1O2(g)"]
            - block.stoichiometricNumber["SMR", "C1O2(g)"]
            * block.stoichiometricNumber["WGS", "C1H4(g)"]
        )
        if i == "C1H4(g)":
            coef = (
                block.stoichiometricNumber["SMR", k]
                * block.stoichiometricNumber["WGS", "C1O2(g)"]
                - block.stoichiometricNumber["SMR", "C1O2(g)"]
                * block.stoichiometricNumber["WGS", k]
            ) / den
        elif i == "C1O2(g)":
            coef = (
                block.stoichiometricNumber["SMR", "C1H4(g)"]
                * block.stoichiometricNumber["WGS", k]
                - block.stoichiometricNumber["SMR", k]
                * block.stoichiometricNumber["WGS", "C1H4(g)"]
            ) / den

        return coef

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

    @block.Constraint(block.setIndependentComponents, block.setTimeSteps)
    def reactionRateConstr(block, k, t):
        """
        Rate equations ri,  i=1,2,3 according to Eq. (3), Xu and Froment (1989).
        Indices i refer to the following reactions:
            1. CH4 + H2O <=> CO + 3H2
            2. CO + H2O <=> CO2 + H2
            3. CH4 + 2 H2O <=> CO2 + 4H2

        References:
        -----------
        Xu, Jianguo; Froment, Gilbert F. (1989): Methane steam reforming, methanation and water-gas shift: I. Intrinsic kinetics. In AIChE J. 35 (1), pp. 88–96. DOI: 10.1002/aic.690350109.
        """
        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        DEN = (
            1
            + block.adsorptionCoef["C1O1(g)"] * block.activity["C1O1(g)", t]
            + block.adsorptionCoef["H2(ref)"] * block.activity["H2(ref)", t]
            + block.adsorptionCoef["C1H4(g)"] * block.activity["C1H4(g)", t]
            + block.adsorptionCoef["H2O1(g)"]
            * block.activity["H2O1(g)", t]
            / block.activity["H2(ref)", t]
        )

        r1 = (
            block.rateConstant["SMR"]
            / (block.activity["H2(ref)", t] ** 2.5)
            * (
                block.activity["C1H4(g)", t] * block.activity["H2O1(g)", t]
                - block.activity["H2(ref)", t] ** 3
                * block.activity["C1O1(g)", t]
                / block.equilibriumConst["SMR"]
            )
            / (DEN**2)
        )

        r2 = (
            block.rateConstant["WGS"]
            / block.activity["H2(ref)", t]
            * (
                block.activity["C1O1(g)", t] * block.activity["H2O1(g)", t]
                - block.activity["H2(ref)", t]
                * block.activity["C1O2(g)", t]
                / block.equilibriumConst["WGS"]
            )
            / (DEN**2)
        )

        r3 = (
            block.rateConstant["DSR"]
            / (block.activity["H2(ref)", t] ** 3.5)
            * (
                block.activity["C1H4(g)", t] * block.activity["H2O1(g)", t] ** 2
                - block.activity["H2(ref)", t] ** 4
                * block.activity["C1O2(g)", t]
                / block.equilibriumConst["DSR"]
            )
            / (DEN**2)
        )

        return (
            block.formationRateSI[k, t]
            == block.stoichiometricNumber["SMR", k] * r1
            + block.stoichiometricNumber["WGS", k] * r2
            + block.stoichiometricNumber["DSR", k] * r3
        )

    @block.Constraint(block.setSteamReformingReactions)
    def equilibriumConstConstr(block, r):
        return block.equilibriumConst[r] == block.equilibriumConstRef[r] * pyo.exp(
            -block.enthalpyReactionSI[r]
            / block.GAS_CONST_SI
            * (
                block.temperatureKelvin ** (-1)
                - block.refTemperatureEquilibriumSI[r] ** (-1)
            )
        )

    @block.Constraint(block.setSteamReformingReactions)
    def rateConstantConstr(block, r):
        return block.rateConstant[r] == block.rateConstantRefSI[r] * pyo.exp(
            -block.activationEnergySI[r]
            / block.GAS_CONST_SI
            * (block.temperatureKelvin ** (-1) - block.refTemperatureRateSI[r] ** (-1))
        )

    @block.Constraint(block.setCompoundsAdsorption)
    def adsorptionCoefConstr(block, k):
        if k not in {"C1H4(g)", "C1O1(g)", "H2(ref)", "H2O1(g)"}:
            return pyo.Constraint.Skip

        return block.adsorptionCoef[k] == block.adsorptionCoefRef[k] * pyo.exp(
            -block.enthalpyAdsorptionSI[k]
            / block.GAS_CONST_SI
            * (
                block.temperatureKelvin ** (-1)
                - block.refTemperatureAdsorptionSI[k] ** (-1)
            )
        )

    @block.Constraint(
        block.setReactingCompounds - block.setIndependentComponents, block.setTimeSteps
    )
    def stoichiometry(block, k, t):

        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        return block.molarFlowRateSI[k, t] - block.molarFlowRateSI[
            k, t - 1
        ] == block.stoichiometyCoefficient["C1H4(g)", k] * (
            block.molarFlowRateSI["C1H4(g)", t]
            - block.molarFlowRateSI["C1H4(g)", t - 1]
        ) + block.stoichiometyCoefficient[
            "C1O2(g)", k
        ] * (
            block.molarFlowRateSI["C1O2(g)", t] - block.molarFlowRateSI["C1O2(g)", t]
        )

    @block.Constraint(block.setIndependentComponents, block.setTimeSteps)
    def backward_euler_discretization(block, k, t):

        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        expr = None
        h = (1 - 0) / block.NUM_FINITE_ELEMENTS

        # dF[t+1] / dW = F[t+1] - F[t] / h
        expr = h * block.derivMolarFlowRateSI[k, t] == (
            block.molarFlowRateSI[k, t] - block.molarFlowRateSI[k, t - 1]
        )

        return expr

    @block.Constraint(block.setIndependentComponents, block.setTimeSteps)
    def performanceEquation(block, k, t):
        # dFi = Ri dW
        # dW = W dW+
        # --> dFi = W Ri dW+
        # --> dFi / dW+ = W Ri
        if t == block.setTimeSteps.first():
            return pyo.Constraint.Skip

        return (
            block.derivMolarFlowRateSI[k, t]
            == block.massCatalystSI * block.formationRateSI[k, t]  # mol / (kg h)
        )

    return block


def pyomo_create_model(**kwargs):

    model = pyo.AbstractModel("Feasibility problem: reforming reactor")
    model.reformer = pyo.Block(rule=reformer_block_rule)

    model.setCompounds = pyo.Set(
        # initialize=[c.name for c in SteamReformingCompounds],
        doc="The set of reactive chemical compounds in the steam methane reforming and water-gas shift reactions."
    )

    model.molarFlowRateInSI = pyo.Param(model.setCompounds)

    @model.Constraint(model.setCompounds)
    def inlet_conditions(model, k):
        return (
            model.reformer.molarFlowRateSI[k, model.reformer.setTimeSteps.first()]
            == model.molarFlowRateInSI[k]
        )

    @model.Objective(sense=pyo.minimize)
    def obj(model):
        return 0

    return model
