import pyomo.environ as pyo
from phdtools.data.constants import GAS_CONST_SI


def pyomo_create_model(options=None, model_options=None):
    """pyomo callback, see documentation"""

    model = pyo.ConcreteModel(
        name="Gibbs reactor model for steam-methane reformer",
        doc="...",
    )

    # 1. Sets
    model.SetCompounds = pyo.Set(
        initialize=["C1H4(g)", "C1O1(g)", "C1O2(g)", "H2(ref)", "H2O1(g)"],
        doc="chemical species involved",
    )
    model.SetElements = pyo.Set(
        initialize=["C", "H", "O"], doc="chemical elements involved"
    )

    # 2. Parameters
    model.gasConst = pyo.Param(
        within=pyo.NonNegativeReals,
        initialize=GAS_CONST_SI * 1e-3,
        doc="universal gas constant (intialized as 8.314e-3 kJ/(mol K))",
    )

    model.stdPressureBar = pyo.Param(
        within=pyo.NonNegativeReals,
        initialize=1,
        doc="standard pressure (intialized as 1 bar)",
    )

    model.numberAtoms = pyo.Param(
        model.SetElements,
        model.SetCompounds,
        within=pyo.NonNegativeReals,
        mutable=False,
        default=0,
        initialize={
            ("C", "C1H4(g)"): 1,
            ("C", "C1O2(g)"): 1,
            ("C", "C1O1(g)"): 1,
            ("H", "C1H4(g)"): 4,
            ("H", "H2O1(g)"): 2,
            ("H", "H2(ref)"): 2,
            ("O", "C1O2(g)"): 2,
            ("O", "C1O1(g)"): 1,
            ("O", "H2O1(g)"): 1,
        },
        doc="numberAtoms[i,j]: number of elements i present in species j",
    )

    model.stdGibbsEnergySlope = pyo.Param(
        model.SetCompounds,
        mutable=False,
        initialize={
            "C1H4(g)": -0.2536598,
            "C1O1(g)": -0.23700023636363635,
            "C1O2(g)": -0.27340541818181824,
            "H2(ref)": -0.16842472727272723,
            "H2O1(g)": -0.2359823818181818,
        },
        doc="stdGibbsEnergySlope[i]: ...",
    )

    model.stdGibbsEnergyIntercept = pyo.Param(
        model.SetCompounds,
        mutable=False,
        initialize={
            "C1H4(g)": -33.77134727272721,
            "C1O1(g)": -87.85134,
            "C1O2(g)": -358.43029454545433,
            "H2(ref)": 21.538472727272676,
            "H2O1(g)": -214.42487090909088,
        },
        doc="stdGibbsEnergySlope[i]: ...",
    )

    # 3. Variables
    model.inFlowRateMolePerSecond = pyo.Var(
        model.SetCompounds,
        domain=pyo.NonNegativeReals,
        doc="inFlowRateMolePerSecond[j]: rate of flow of moles j into the reactor in mol/s",
    )

    model.outFlowRateMolePerSecond = pyo.Var(
        model.SetCompounds,
        domain=pyo.NonNegativeReals,
        doc="outFlowRateMolePerSecond[j]: rate of flow of moles j out of the reactor in mol/s",
    )

    model.temperatureKelvin = pyo.Var(
        within=pyo.NonNegativeReals,
        bounds=(600, 1600),
        doc="absolute temperature in K",
    )

    model.pressureBar = pyo.Var(
        within=pyo.NonNegativeReals,
        bounds=(5, 20),
        doc="pressure in bar",
    )

    model.stdGibbsEnergy = pyo.Var(
        model.SetCompounds,
        within=pyo.Reals,
        # bounds=, #TODO ! compute
        doc="gibbs free energy of formation in kJ/mol",
    )

    model.lagrangeMultiplier = pyo.Var(
        model.SetElements,
        within=pyo.Reals,
        doc="Lagrange multiplier for each element conservation constraint",
    )

    # 4. Constraints
    @model.Constraint(model.SetElements, doc="material balance")
    def element_conservation(self, k):
        return (
            sum(
                self.numberAtoms[k, j]
                * (self.inFlowRateMolePerSecond[j] - self.outFlowRateMolePerSecond[j])
                for j in self.SetCompounds
            )
            == 0
        )

    @model.Constraint(
        model.SetCompounds,
        doc="KKT-(1) condition of Gibbs free energy minimization (perfect gas)",
    )
    def kkt1_gibbs_minimization(self, k):
        return (
            self.stdGibbsEnergy[k] / (self.gasConst * self.temperatureKelvin)
            + pyo.log(self.pressureBar / self.stdPressureBar)
            + pyo.log(
                self.outFlowRateMolePerSecond[k]
                / pyo.summation(self.outFlowRateMolePerSecond)
            )
            + sum(
                self.numberAtoms[j, k] * self.lagrangeMultiplier[j]
                for j in self.SetElements
            )
            == 0
        )

    @model.Constraint(
        model.SetCompounds,
        doc="temperature dependence of std. Gibbs free energy",
    )
    def std_gibbs_energy_model(self, k):
        return (
            self.stdGibbsEnergy[k]
            == self.stdGibbsEnergySlope[k] * model.temperatureKelvin
            + self.stdGibbsEnergyIntercept[k]
        )

    # 5. Objective(s)
    @model.Expression(model.SetCompounds)
    def conversion(self, k):
        return 1 - self.outFlowRateMolePerSecond[k] / self.inFlowRateMolePerSecond[k]

    @model.Objective(sense=pyo.maximize)
    def max_methane_conversion(self):
        return self.conversion["C1H4(g)"]

    return model
