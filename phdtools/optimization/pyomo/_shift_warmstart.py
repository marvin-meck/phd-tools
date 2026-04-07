import numpy as np
import pyomo.environ as pyo
from scipy.integrate import solve_ivp

from phdtools.integrate import euler
from phdtools.optimization import (
    SHIFT_TEMPERATURE_SI_INIT,
    FUEL_CELL_CARBON_MONOXIDE_TOLERANCE,
)

from phdtools.optimization.pyomo._shift_block import (
    # NUM_FINITE_ELEMENTS,
    TEMPERATURE_SCALE,
    MOLAR_FLOW_RATE_SCALE,
    MASS_CATALYST_SCALE,
)


from phdtools.models.choi_stenger_2003 import equilibriumConversionWGS

from phdtools.models.mendes_2010 import (
    STST_PRESSURE_BAR,
    Compound as Compound,
    ModelParameters as WaterGasShiftParameters,
    stoichiometricNumber,
    # Compound as allCompounds,
    rateConstModel,
    equilibriumConstModel,
    adsorptionCoefModel,
    reactionRateModel,
    stoichiometryShift,
    initialValueProblemSpaceTime,
    initialValueProblemConversion,
)

PARAMS = WaterGasShiftParameters.init(model="LH1")

MAXITER = 100


def get_equilibriumSpaceTime(
    molarFlowRateIn, temperatureKelvin, pressureBar, params, Compound=Compound
):

    moleFractionIn = molarFlowRateIn / molarFlowRateIn.sum()

    eqConversion = equilibriumConversionWGS(
        moleFractionIn, temperatureKelvin, model="vantHoff"
    )[0, 0]

    conversion = np.linspace(0, eqConversion - 1e-3)

    sol = solve_ivp(
        fun=initialValueProblemConversion,
        t_span=np.array([conversion.min(), conversion.max()]),
        y0=np.array([0]),
        method="RK45",
        t_eval=conversion,
        dense_output=False,
        events=None,
        vectorized=True,
        args=(moleFractionIn, temperatureKelvin, pressureBar, params),
    )

    spaceTimeSI = sol.y[0][-1]

    return spaceTimeSI


def get_spaceTimeFromConversion(
    conversion,
    molarFlowRateIn,
    temperatureKelvin,
    pressureBar,
    params=PARAMS,
    Compound=Compound,
):

    moleFractionIn = molarFlowRateIn / molarFlowRateIn.sum()

    conversionRange = np.linspace(0, conversion)

    sol = solve_ivp(
        fun=initialValueProblemConversion,
        t_span=np.array([conversionRange.min(), conversionRange.max()]),
        y0=np.array([0]),
        method="RK45",
        t_eval=conversionRange,
        dense_output=False,
        events=None,
        vectorized=True,
        args=(moleFractionIn, temperatureKelvin, pressureBar, params),
    )

    spaceTimeSI = sol.y[0][-1]

    return spaceTimeSI


def rate_const_initialization_rule(block, temperatureKelvin):
    return rateConstModel(temperatureKelvin, PARAMS)[0][0]


def equilibrium_const_initialization_rule(block, temperatureKelvin):
    return equilibriumConstModel(temperatureKelvin, PARAMS)[0][0]


def adsorption_coef_initialization_rule(block, k, temperatureKelvin):
    return adsorptionCoefModel(temperatureKelvin, PARAMS)[0][Compound[k].value]


def mass_of_solids_initialization_rule(block, temperatureKelvin, molarFlowRateInSI):

    # kg(cat) s / mol = 1/3.6 g(cat) h / mol
    conversion = (
        1
        - FUEL_CELL_CARBON_MONOXIDE_TOLERANCE
        * molarFlowRateInSI.sum(axis=0)
        / molarFlowRateInSI[Compound["C1O1(g)"].value]
    )

    spaceTimeSI = get_spaceTimeFromConversion(
        conversion,
        molarFlowRateIn=molarFlowRateInSI,
        temperatureKelvin=temperatureKelvin,
        pressureBar=block.pressureBar.value,
        params=PARAMS,
    )

    return molarFlowRateInSI[Compound["C1O1(g)"].value] * spaceTimeSI


def molar_flow_rate_initialization_rule(block, temperatureKelvin, molarFlowRateInSI):

    scale = 1

    conversion = (
        1
        - FUEL_CELL_CARBON_MONOXIDE_TOLERANCE
        * molarFlowRateInSI.sum(axis=0)
        / molarFlowRateInSI[Compound["C1O1(g)"].value]
    )
    moleFractionIn = molarFlowRateInSI / molarFlowRateInSI.sum()

    spaceTimeSI = get_spaceTimeFromConversion(
        conversion,
        molarFlowRateIn=molarFlowRateInSI,
        temperatureKelvin=temperatureKelvin,
        pressureBar=block.pressureBar.value,
        params=PARAMS,
    )

    # print(spaceTimeSI)

    sol = euler(
        fun=initialValueProblemSpaceTime,
        t_span=np.array([0, spaceTimeSI]),
        y0=np.array([0]),
        method="backward",
        n=block.NUM_FINITE_ELEMENTS.value,
        maxiter=50,
        args=(moleFractionIn, temperatureKelvin, block.pressureBar.value, PARAMS),
    )

    moleFractionOut = np.zeros((len(Compound), len(sol.y[0])))
    for num, X in enumerate(sol.y[0]):
        moleFractionOut[:, num] = stoichiometryShift(moleFractionIn, X)[:,0] / scale

    molarFlowRate = moleFractionOut * molarFlowRateInSI.sum()

    return {
        (c.name, t): float(molarFlowRate[c.value, t])
        for c in Compound
        if c.name in block.setReactingCompounds
        for t in range(len(sol.t))
    }


def activity_initialization_rule(block, k, t):

    molarFlowRateIn = np.zeros(len(Compound))
    for c in Compound:
        if c.name in block.setReactingCompounds:
            molarFlowRateIn[c.value] = pyo.value(block.molarFlowRateSI[c.name, t])
        # elif c.name in block.setInertCompounds:
        #     molarFlowRateIn[c.value] = block.molarFlowRateInertCompoundsSI[c.name].value

    partialPressureBar = (
        block.pressureBar.value
        * molarFlowRateIn[Compound[k].value]
        / molarFlowRateIn.sum()
    )

    if t > 0:
        val = pyo.value(partialPressureBar / block.STST_PRESSURE_BAR)
    else:
        val = None

    return val


def formation_rate_initialization_rule(block, t, temperatureKelvin):

    molarFlowRateIn = np.zeros(len(Compound))
    for c in Compound:
        if c.name in block.setReactingCompounds:
            molarFlowRateIn[c.value] = pyo.value(block.molarFlowRateSI[c.name, t])

    partialPressureBar = (
        block.pressureBar.value * molarFlowRateIn / molarFlowRateIn.sum()
    )

    r = reactionRateModel(partialPressureBar, temperatureKelvin, PARAMS)

    if t > 0:
        val = r[0]
    else:
        val = None

    return val


def deriv_molar_flow_rate_initialization_rule(block, t):

    tf, t0 = 1, 0
    h = (tf - t0) / block.NUM_FINITE_ELEMENTS.value

    if t > 0:
        val = (
            pyo.value(
                block.molarFlowRateSI["C1O1(g)", t]
                - block.molarFlowRateSI["C1O1(g)", t - 1]
            )
            / h
        )
    else:
        val = None

    return val


def warmstart_shift(block, temperatureKelvin, molarFlowRateInSI):

    block.temperatureScaled = temperatureKelvin / block.temperatureUpperBoundSI

    block.rateConstantScaled = (
        rate_const_initialization_rule(block, temperatureKelvin)
        / block.rateConstUpperBound
    )
    block.equilibriumConstScaled = (
        equilibrium_const_initialization_rule(block, temperatureKelvin)
        / block.equilibriumConstUpperBound
    )

    for c in block.setCompoundsAdsorption:
        block.adsorptionCoefScaled[c] = (
            adsorption_coef_initialization_rule(block, c, temperatureKelvin)
            / block.adsorptionCoefUpperBound[c]
        )

    block.massCatalystScaled = (
        mass_of_solids_initialization_rule(block, temperatureKelvin, molarFlowRateInSI)
        / block.massCatalystUpperBoundSI
    )

    molarFlowRateDict = molar_flow_rate_initialization_rule(
        block, temperatureKelvin, molarFlowRateInSI
    )
    for c, t in block.setReactingCompounds * block.setTimeSteps:
        block.molarFlowRateScaled[c, t] = (
            molarFlowRateDict[c, t] / block.molarFlowRateUpperBoundSI[c, t]
        )

    for c, t in block.setReactingCompounds * block.setTimeSteps:
        val = activity_initialization_rule(block, c, t)
        if val is not None:
            block.activity[c, t] = val

    for t in block.setTimeSteps:
        val = formation_rate_initialization_rule(block, t, temperatureKelvin)
        if val is not None:
            block.formationRateScaled[t] = val / block.formationRateUpperBoundSI[t]

    for t in block.setTimeSteps:
        val = deriv_molar_flow_rate_initialization_rule(block, t)
        if val is not None:
            block.derivMolarFlowRateScaled[t] = (
                val / block.derivMolarFlowRateUpperBoundSI[t]
            )
