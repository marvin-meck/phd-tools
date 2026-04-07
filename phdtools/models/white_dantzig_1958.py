"""phdtools.models.white_dantzig_1958.py

Copyright 2023 Technical University Darmstadt

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Description
-----------

Computes the chemical equilibrium by direct minimization of Gibbs free energy, c.f. [1,2].
Implementation in Pyomo modelling language, see [3]

Author: Marvin Meck
E-Mail: marvin.meck@tu-darmstadt.de

Corrensponding: Prof. Dr.-Ing. Peter F. Pelz
E-Mail: peter.pelz@tu-darmstadt.de

References:
-----------

[1] White, W. B.; Johnson, S. M.; Dantzig, G. B. (1958): Chemical Equilibrium in Complex Mixtures.
    In The Journal of Chemical Physics 28 (5), pp. 751–755. DOI: 10.1063/1.1744264.

[2] McDonald, C. M.; Floudas, C. A. (1995): Global optimization for the phase and chemical equilibrium
    problem: Application to the NRTL equation. In Computers & Chemical Engineering 19 (11), pp. 1111–1139.
    DOI: 10.1016/0098-1354(94)00106-5.

[3] Hart, William E., William E. Hart; Watson, Jean-Paul; Laird, Carl D.; Nicholson, Bethany L.;
    Siirola, John D. (2017): Pyomo. Optimization modeling in Python. Second edition /
    William E Hart [and six others]. Cham: Springer (Springer Optimization and Its Applications, 67).

"""

import hashlib
import io
import pyomo.environ as pyo

from phdtools.data.constants import GAS_CONST_SI


def pyomo_create_model(options=None, model_options=None):
    m = pyo.AbstractModel(
        name="Chemical Equilibrium Problem (perfect gas)",
        doc="Minimize Gibbs free energy of a mixture of perfect gases",
    )

    # 1. Sets
    m.SetCompounds = pyo.Set(doc="chemical species involved")
    m.SetElements = pyo.Set(doc="chemical elements involved")

    # 2. Parameters
    m.std_gibbs_free_energy = pyo.Param(
        m.SetCompounds,
        within=pyo.Reals,
        mutable=True,
        doc="gibbs free energy of formation in kJ/mol",
    )

    m.gas_const = pyo.Param(
        within=pyo.NonNegativeReals,
        initialize=GAS_CONST_SI * 1e-3,
        doc="universal gas constant in kJ/(mol K)",
    )

    m.temperature = pyo.Param(
        within=pyo.NonNegativeReals,
        mutable=True,
        doc="absolute temperature in K",
    )

    m.pressure = pyo.Param(
        within=pyo.NonNegativeReals,
        mutable=True,
        doc="pressure",
    )

    m.standard_pressure = pyo.Param(
        within=pyo.NonNegativeReals,
        initialize=1,
        doc="standard pressure in atm",
    )

    @m.Expression(m.SetCompounds)
    def constant(self, i):
        return self.std_gibbs_free_energy[i] / (
            self.gas_const * self.temperature
        ) + pyo.log(self.pressure / self.standard_pressure)

    # TODO where does that term come from exactly, see Bearns S.40. What is the correct reference pressure and WHY?
    #   why do people write ln(p) all the time...

    m.number_of_atoms = pyo.Param(
        m.SetElements * m.SetCompounds,
        within=pyo.NonNegativeReals,
        mutable=False,
        default=0,
        doc="atom-weight of elements present in species",
    )

    m.amount_element = pyo.Param(
        m.SetElements,
        within=pyo.NonNegativeReals,
        mutable=True,
        default=0,
        doc="amount of elements",
    )

    # 3. Variables
    m.amount_substance = pyo.Var(
        m.SetCompounds,
        within=pyo.NonNegativeReals,
        doc="amount of substance of species in equilibrium",
    )

    m.total_amount_substance = pyo.Var(
        within=pyo.NonNegativeReals, doc="total amount of substance in equilibrium"
    )

    @m.Constraint(m.SetElements, doc="material balance")
    def element_conservation(self, k):
        lhs = sum(
            self.number_of_atoms[k, i] * self.amount_substance[i]
            for i in self.SetCompounds
        )
        rhs = self.amount_element[k]
        return lhs == rhs

    @m.Constraint(doc="total amount is the sum of amount of species")
    def total_amount_constr(self):
        return pyo.summation(self.amount_substance) == self.total_amount_substance

    @m.Objective(sense=pyo.minimize, doc="minimize Gibbs free energy")
    def obj(self):
        return pyo.summation(self.constant, self.amount_substance) + sum(
            self.amount_substance[i]
            * pyo.log(self.amount_substance[i] / self.total_amount_substance)
            for i in self.SetCompounds
        )

    return m


# def pyomo_modify_instance(options=None, model=None, instance=None):
#     """ "callback function modifies the model instance after it ahs been constructed" [3]. Used here to initialize a feasible equilibrium."""
# m, n = len(instance.amount_element), len(instance.SetCompounds)
# feastol = 1e-6

# A = np.zeros(shape=(m, n), dtype=np.int32)

# for i, e in enumerate(instance.SetElements):
#     for j, c in enumerate(instance.SetCompounds):
#         A[i, j] = pyo.value(instance.number_of_atoms[(e, c)])

# b = np.array([pyo.value(instance.amount_element[e]) for e in instance.SetElements])

# res = minimize(
#     lambda x: np.sum((np.matmul(A, x) - b.T) ** 2),
#     x0=np.zeros(len(instance.SetCompounds)),
#     bounds=[(feastol, 1)] * len(instance.SetCompounds),
# )

# for j, c in enumerate(instance.SetCompounds):
#     instance.amount_substance[c].value = res.x[j]
# pass


def pyomo_postprocess(options=None, instance=None, results=None):

    datafile = options.data.files[0]
    with open(datafile, "r") as f:
        datacmds = f.read()

    checksum = hashlib.md5(datacmds.encode("utf-8"))

    # default_variable_value=0 doesn't work because smap_id = None,
    # so we set them manually
    for var in instance.component_data_objects(pyo.Var):
        if var.value is None:
            var.value = 0

    mole_fractions = {
        c: pyo.value(instance.amount_substance[c])
        / pyo.value(instance.total_amount_substance)
        for c in instance.SetCompounds
    }

    sc = (pyo.value(instance.amount_element["H"]) - 4) / (
        2 * pyo.value(instance.amount_element["C"])
    )

    with io.StringIO() as f:
        f.write(f"\nResults for {datafile}; MD5 checksum: {checksum.hexdigest()}\n\n")
        f.write("\t**Parameters**\n")
        f.write(f"\t\tTemperature (K):\t{pyo.value(instance.temperature)}\n")
        f.write(f"\t\tPressure (atm):\t\t{pyo.value(instance.pressure)}\n")
        f.write(f"\t\tSteam to carbon ratio:\t{sc}\n")
        f.write("\n")
        f.write("\t**Mole fractions**\n")
        for c in instance.SetCompounds.sorted_data():
            f.write(f"\t\t{c}:\t{mole_fractions[c]}\n")

        out = f.getvalue()
    print(out)


# def warm_start(model, sol, verbose=0):
# #     with open(fname, 'r') as f:
# #         sol = json.loads(f.read())

#     for name in sol.keys():
#         if verbose > 0:
#             print("initializing component:", name)
#         for var in sol[name]:
#             if type(var['key']) is tuple:
#                 index = tuple(var['key'])
#             else:
#                 index = var['key']

#             value = var['value']
#             obj = getattr(model, name)
#             obj[index].set_value(value)
