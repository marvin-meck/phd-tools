from io import StringIO

import pyomo.environ as pyo

from phdtools.optimization.pyomo._base_model import BaseModel

# def pyomo_preprocess(options=None):
#     print("Here are the options that were provided:")
#     if options is not None:
#         options.display()


def create_abstract_model():

    model = BaseModel("Cost minimization")

    # model.epsThermalEfficiency = pyo.Param()
    model.epsElectricalPowerSI = pyo.Param()

    # Epsilon constraints
    # @model.Constraint()
    # def epsilon_constraint_thermal_efficiency(block):
    #     return block.molarFlowRateInSI[
    #         "C1H4(g)"
    #     ] <= -1 * block.NOMINAL_THERMAL_POWER_SI / (
    #         block.grossCalorificValueMethaneSI * block.epsThermalEfficiency
    #     )

    @model.Constraint()
    def epsilon_constraint_power(block):
        return block.electricalPowerSI >= block.epsElectricalPowerSI

    @model.Objective(sense=pyo.minimize)
    def obj(block):
        return block.variableCostsEuro

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


# def pyomo_load_solution(fname, model: pyo.AbstractModel | pyo.ConcreteModel, modeldata: pyo.DataPortal | None=None):

#     if isinstance(model, pyo.AbstractModel):
#         if modeldata is None:
#             raise ValueError("'model' is of type 'AbstractModel', 'modeldata' cannot be 'None'!")

#         model.reformer.construct(modeldata.data(namespace="reformer"))
#         model.shift.construct(modeldata.data(namespace="shift"))
#         model.fuel_cell.construct(modeldata.data(namespace="fuel_cell"))
#         instance = model.create_instance(modeldata, namespace=None)

#         instance = model.create_instance(data=modeldata)

#     elif isinstance(model, pyo.ConcreteModel):
#         instance = model
#     else:
#         raise TypeError("Model must be 'AbstactModel' or 'ConcreteModel'")

#     results = SolverResults()
#     results.read(filename=fname)

#     # fix the solution object, otherwise results.solutions.load_from(...) won't work
#     results.solution(0)._cuid = False
#     results.solution.Constraint = {}

#     instance.solutions.load_from(results)

#     # default_variable_value=0 doesn't work because smap_id = None,
#     # so we set them manually
#     for var in instance.component_data_objects(pyo.Var):
#         if var.value is None:
#             var.value = 0

#     return instance
