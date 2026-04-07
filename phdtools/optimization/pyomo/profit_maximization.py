from io import StringIO
import pyomo.environ as pyo

from phdtools.optimization.pyomo._base_model import BaseModel
from phdtools.optimization.pyomo._choice_model import add_consumer_preference_model


def create_abstract_model():

    model = BaseModel("Profit maximization")

    add_consumer_preference_model(model)

    model.contributionMarginLowerBoundEuro = pyo.Param()
    model.contributionMarginUpperBoundEuro = pyo.Param()

    def contribution_margin_bounds_rule(block):
        return (
            block.contributionMarginLowerBoundEuro
            / block.contributionMarginUpperBoundEuro,
            1.0,
        )

    model.contributionMarginScaled = pyo.Var(
        bounds=contribution_margin_bounds_rule, within=pyo.NonNegativeReals
    )

    @model.Expression()
    def contributionMarginEuro(block):
        return block.contributionMarginUpperBoundEuro * block.contributionMarginScaled

    model.normalizedTotalContributionLowerBoundEuro = pyo.Param()
    model.normalizedTotalContributionUpperBoundEuro = pyo.Param()

    def normalized_total_contribution_bounds_rule(block):
        return (
            block.normalizedTotalContributionLowerBoundEuro
            / block.normalizedTotalContributionUpperBoundEuro,
            1.0,
        )

    model.normalizedTotalContributionScaled = pyo.Var(
        bounds=normalized_total_contribution_bounds_rule, within=pyo.NonNegativeReals
    )

    @model.Expression()
    def normalizedTotalContributionEuro(block):
        return (
            block.normalizedTotalContributionUpperBoundEuro
            * block.normalizedTotalContributionScaled
        )

    @model.Constraint()
    def contribution_margin_constr(block):
        return block.variableCostsEuro + block.contributionMarginEuro <= block.priceEuro

    @model.Constraint()
    def normalized_total_contribution_constr(block):
        return (
            block.normalizedTotalContributionEuro
            <= block.contributionMarginEuro * block.marketShare
        )

    @model.Objective(sense=pyo.maximize)
    def obj(block):
        return block.normalizedTotalContributionScaled

    return model


def pyomo_create_model(options=None, model_options=None):

    files = options["data"]["files"]

    dp = pyo.DataPortal(filename=files[0])

    model = create_abstract_model()
    model.reformer.construct(dp.data(namespace="reformer"))
    model.shift.construct(dp.data(namespace="shift"))
    model.fuel_cell.construct(dp.data(namespace="fuel_cell"))
    instance = model.create_instance(dp, namespace=None)

    return instance


def pyomo_print_results(options=None, instance=None, results=None):
    with StringIO() as ostream:
        ostream.write(f"\n")
        ostream.write(
            f"----------------------------------------------------------------\n"
        )
        ostream.write(
            f"Market share:\t\t\t{100*pyo.value(instance.marketShare):.2f} %\n"
        )
        ostream.write(
            f"Cost per unit:\t\t\t{pyo.value(instance.variableCostsEuro):.2f} Euro\n"
        )
        ostream.write(
            f"Thermal efficiency (HHV):\t{pyo.value(instance.thermalEfficiency):.2f}\n"
        )
        ostream.write(
            f"Electrical efficiency (HHV):\t{pyo.value(instance.electricalEfficiency):.2f}\n"
        )
        ostream.write(f"\n")
        ostream.write(
            f"Thermal power:\t\t\t{1e-3*pyo.value(instance.NOMINAL_THERMAL_POWER_SI):.2f} kW(th)\n"
        )
        ostream.write(
            f"Electrical power:\t\t{1e-3*pyo.value(instance.electricalPowerSI):.2f} kW(el)\n"
        )
        ostream.write(f"\n")
        ostream.write(
            f"Cell voltage:\t\t\t{pyo.value(instance.fuel_cell.cellPotentialSI):.2f} V\n"
        )
        ostream.write(
            f"Current density:\t\t{1e-4*pyo.value(instance.fuel_cell.currentDensitySI):.2f} A/cm2\n"
        )
        ostream.write(
            f"Power density:\t\t\t{1e-4*pyo.value(instance.fuel_cell.powerDensitySI):.2f} W/cm2\n"
        )
        ostream.write(f"\n")
        ostream.write(
            f"Methane (CH4) feed:\t\t{pyo.value(instance.molarFlowRateInSI["C1H4(g)"]*3600):.2f} mol/h\n"
        )
        ostream.write(
            f"STC ratio (H2O/CH4):\t\t{pyo.value(instance.molarFlowRateInSI["H2O1(l)"]/instance.molarFlowRateInSI["C1H4(g)"]):.2f}\n"
        )
        ostream.write(
            f"ATF ratio (Air/CH4):\t\t{pyo.value(sum(instance.molarFlowRateInSI[j] for j in instance.setCompoundsAir)/instance.molarFlowRateInSI["C1H4(g)"]):.2f}\n"
        )
        ostream.write(f"\n")
        ostream.write(
            f"Total water (H2O) in:\t\t{pyo.value((instance.molarFlowRateInSI["H2O1(g)"] + instance.molarFlowRateInSI["H2O1(l)"])*3600):.2f} mol/h\n"
        )
        ostream.write(
            f"Total water (H2O) out:\t\t{pyo.value((instance.molarFlowRateOutSI["H2O1(g)"] + instance.molarFlowRateOutSI["H2O1(l)"])*3600):.2f} mol/h\n"
        )
        ostream.write(
            f"Total water (H2O) produced:\t{pyo.value((instance.molarFlowRateProductWaterOutSI)*3600):.2f} mol/h\n"
        )
        ostream.write(f"\n")
        ostream.write(
            f"Liquid water (H2O(l)) in:\t{pyo.value(instance.molarFlowRateInSI["H2O1(l)"]*3600):.2f} mol/h\n"
        )
        ostream.write(
            f"Liquid water (H2O(l)) out:\t{pyo.value(instance.molarFlowRateOutSI["H2O1(l)"]*3600):.2f} mol/h\n"
        )
        ostream.write(
            f"Water vapour (H2O(g)) out:\t{pyo.value(instance.molarFlowRateOutSI["H2O1(g)"]*3600):.2f} mol/h\n"
        )
        ostream.write(f"\n")
        ostream.write(
            f"Fuel cell active area:\t\t{pyo.value(instance.fuel_cell.totalActiveAreaSI)*1e4:.2f} cm2\n"
        )
        ostream.write(
            f"Catalyst mass reformer:\t\t{pyo.value(instance.reformer.massCatalystSI)*1e3:.2f} g\n"
        )
        ostream.write(
            f"Catalyst mass shift reactor:\t{pyo.value(instance.shift.massCatalystSI)*1e3:.2f} g\n"
        )

        print(ostream.getvalue())
