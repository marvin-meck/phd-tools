import numpy as np
import pyomo.environ as pyo
from scipy.integrate import solve_ivp

from phdtools.integrate import euler
from phdtools.optimization import REFORMER_METHANE_CONVERSION_INIT


from phdtools.models.xu_froment_1989 import (
    Compound as Compound,
    Reaction as SteamReformingReactions,
    ModelParameters as SteamReformingParameters,
    stoichiometricNumber,
    rateConstModel,
    equilibriumConstModel,
    adsorptionCoefModel,
    reactionRateModel,
    get_equilibriumConversion,
    stoichiometryReformer,
    initialValueProblemSpaceTime,
    initialValueProblemConversion,
)

PARAMS = SteamReformingParameters.init()

MAXITER = 100

# block.pressureBar.value = REFORMER_PRESSURE_SI * 1e-5


def get_equilibriumSpaceTime(
    molarFlowRateIn, temperatureKelvin, pressureBar, params=PARAMS, Compound=Compound
):

    eqConversion = get_equilibriumConversion(
        molarFlowRateIn=molarFlowRateIn,
        temperatureKelvin=temperatureKelvin,
        pressureBar=pressureBar,
        params=params,
        Compound=Compound,
        maxiter=MAXITER,
    )
    conversion = np.linspace(0, eqConversion[0] - 1e-3)

    sol = solve_ivp(
        fun=initialValueProblemConversion,
        t_span=np.array([conversion.min(), conversion.max()]),
        y0=np.array([0, 0]),
        method="RK45",
        t_eval=conversion,
        dense_output=False,
        events=None,
        vectorized=True,
        args=(molarFlowRateIn, temperatureKelvin, pressureBar, params),
    )

    spaceTimeSI = sol.y[0][-1]

    return spaceTimeSI


def get_spaceTimeFromConversion(
    conversion, molarFlowRateIn, temperatureKelvin, pressureBar, params=PARAMS
):

    conversionRange = np.linspace(0, conversion)

    sol = solve_ivp(
        fun=initialValueProblemConversion,
        t_span=np.array([conversionRange.min(), conversionRange.max()]),
        y0=np.array([0, 0]),
        method="RK45",
        t_eval=conversionRange,
        dense_output=False,
        events=None,
        vectorized=True,
        args=(molarFlowRateIn, temperatureKelvin, pressureBar, params),
    )

    spaceTimeSI = sol.y[0][-1]

    return spaceTimeSI


def rate_const_initialization_rule(block, r, temperatureKelvin):
    return rateConstModel(temperatureKelvin, PARAMS)[0][
        SteamReformingReactions[r].value
    ]


def equilibrium_const_initialization_rule(block, r, temperatureKelvin):
    return equilibriumConstModel(temperatureKelvin, PARAMS)[0][
        SteamReformingReactions[r].value
    ]


def adsorption_coef_initialization_rule(block, k, temperatureKelvin):
    return adsorptionCoefModel(temperatureKelvin, PARAMS)[0][Compound[k].value]


def mass_of_solids_initialization_rule(block, temperatureKelvin, molarFlowRateInSI):

    # kg(cat) s / mol = 1/3.6 g(cat) h / mol

    conversion = REFORMER_METHANE_CONVERSION_INIT
    spaceTimeSI = get_spaceTimeFromConversion(
        conversion,
        molarFlowRateIn=molarFlowRateInSI,
        temperatureKelvin=temperatureKelvin,
        pressureBar=block.pressureBar.value,
        params=PARAMS,
    )
    # spaceTimeSI = get_equilibriumSpaceTime(
    #     molarFlowRateIn=molarFlowRateIn,
    #     temperatureKelvin=temperatureKelvin,
    #     pressureBar=block.pressureBar.value,
    #     params=PARAMS,
    #     Compound=Compound
    # )

    return molarFlowRateInSI[Compound["C1H4(g)"].value] * spaceTimeSI


def molar_flow_rate_initialization_rule(block, temperatureKelvin, molarFlowRateInSI):

    conversion = REFORMER_METHANE_CONVERSION_INIT
    spaceTimeSI = get_spaceTimeFromConversion(
        conversion,
        molarFlowRateIn=molarFlowRateInSI,
        temperatureKelvin=temperatureKelvin,
        pressureBar=block.pressureBar.value,
        params=PARAMS,
    )
    # get_equilibriumSpaceTime(
    #     molarFlowRateIn=molarFlowRateIn,
    #     temperatureKelvin=temperatureKelvin,
    #     pressureBar=block.pressureBar.value,
    #     params=PARAMS,
    #     Compound=Compound
    # )

    sol = euler(
        fun=initialValueProblemSpaceTime,
        t_span=np.array([0, spaceTimeSI]),
        y0=np.array([0, 0]),
        method="backward",
        n=block.NUM_FINITE_ELEMENTS.value,
        maxiter=MAXITER,
        args=(molarFlowRateInSI, temperatureKelvin, block.pressureBar.value, PARAMS),
    )

    molarFlowRate = stoichiometryReformer(molarFlowRateInSI, sol.y)

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


def formation_rate_initialization_rule(block, k, t, temperatureKelvin):

    molarFlowRateIn = np.zeros(len(Compound))
    for c in Compound:
        if c.name in block.setReactingCompounds:
            molarFlowRateIn[c.value] = pyo.value(block.molarFlowRateSI[c.name, t])

    partialPressureBar = (
        block.pressureBar.value * molarFlowRateIn / molarFlowRateIn.sum()
    )

    r1, r2, r3 = reactionRateModel(partialPressureBar, temperatureKelvin, PARAMS)

    if t > 0:
        val = float(
            stoichiometricNumber["SMR"][k] * r1
            + stoichiometricNumber["WGS"][k] * r2
            + stoichiometricNumber["DSR"][k] * r3
        )
    else:
        val = None

    return val


def deriv_molar_flow_rate_initialization_rule(block, k, t):

    tf, t0 = 1, 0
    h = (tf - t0) / block.NUM_FINITE_ELEMENTS.value

    if t > 0:
        val = (
            pyo.value(block.molarFlowRateSI[k, t] - block.molarFlowRateSI[k, t - 1]) / h
        )
    else:
        val = None

    return val


def warmstart_reformer(block, temperatureKelvin, molarFlowRateInSI):

    block.temperatureScaled = temperatureKelvin / block.temperatureUpperBoundSI

    for r in block.setSteamReformingReactions:
        block.rateConstantScaled[r] = (
            rate_const_initialization_rule(block, r, temperatureKelvin)
            / block.rateConstUpperBound[r]
        )

    for r in block.setSteamReformingReactions:
        block.equilibriumConstScaled[r] = (
            equilibrium_const_initialization_rule(block, r, temperatureKelvin)
            / block.equilibriumConstUpperBound[r]
        )

    for c in block.setCompoundsAdsorption:
        block.adsorptionCoefScaled[c] = (
            adsorption_coef_initialization_rule(block, c, temperatureKelvin)
            / block.adsorptionCoefUpperBound[c]
        )

    block.massCatalystScaled = (
        mass_of_solids_initialization_rule(block, temperatureKelvin, molarFlowRateInSI)
    ) / block.massCatalystUpperBoundSI

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

    for c, t in block.setIndependentComponents * block.setTimeSteps:
        val = formation_rate_initialization_rule(block, c, t, temperatureKelvin)
        if val is not None:
            block.formationRateScaled[c, t] = (
                val / block.formationRateUpperBoundSI[c, t]
            )

    for c, t in block.setIndependentComponents * block.setTimeSteps:
        val = deriv_molar_flow_rate_initialization_rule(block, c, t)
        if val is not None:
            block.derivMolarFlowRateScaled[c, t] = (
                val / block.derivMolarFlowRateUpperBoundSI[c, t]
            )
