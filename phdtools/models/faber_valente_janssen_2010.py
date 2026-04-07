"""phdtools.models.faber_valente_janssen_2010.py

Copyright 2021 Technical University Darmstadt

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

Repast4py implementation of Faber et al. (2010)

@reference: Faber, Albert; Valente, Marco; Janssen, Peter (2010): Exploring domestic
    micro-cogeneration in the Netherlands: An agent-based demand model for technology
    diffusion. In Energy Policy 38 (6), pp. 2763–2775.
    DOI: 10.1016/j.enpol.2010.01.008.

@reference: Collier, Nicholson T.; Ozik, Jonathan; Tatara, Eric R. (2020): Experiences
    in Developing a Distributed Agent-based Modeling Toolkit with Python.
    In : 2020 IEEE/ACM 9th Workshop on Python for High-Performance and Scientific
    Computing (PyHPC). 2020 IEEE/ACM 9th Workshop on Python for High-Performance and
    Scientific Computing (PyHPC). GA, USA, 13.11.2020 - 13.11.2020: IEEE, pp. 1–12.

@author: Marvin Meck,
@email: marvin.meck@fst.tu-darmstadt.de
"""

from typing import Dict
from dataclasses import dataclass

from mpi4py import MPI
import numpy as np
import numpy.ma as ma

from repast4py import context as ctx
from repast4py import core, random, schedule, logging, parameters

MODEL = None


class ConsumerAgent(core.Agent):
    """Fabers consumer agents"""

    def __init__(
        self, nid: int, agent_type: int, rank: int, heating_unit_age, awareness, adopted
    ):
        super().__init__(nid, agent_type, rank)
        self.housing_type = agent_type
        rng = random.default_rng
        self.visibility_threshold = rng.random(2)  # rng.random(MODEL.num_tech_options)
        self.adopted = adopted
        self.heating_unit_age = heating_unit_age
        self.aware = awareness

    def visibility(self):
        """Defines the visibility function, see Eq. (1)"""
        market_size_effect = (
            MODEL.advertising_factor + MODEL.market_share**MODEL.confidence_in_market
        )
        return np.fmax(self.aware, np.fmin(market_size_effect, 1))

    def step(self):
        """Implements steps:
        1. enter the market when current heating system reaches EOL
        2. scan the market for replacement options
            2.1 compute the visibility
            2.2 visibility is compared to a threshold to optain awareness
        3. out of the options the agent is aware of, choose the one associated
            with the lowest total cost.
        """
        self.heating_unit_age += 1
        if self.heating_unit_age == MODEL.age_of_replacement[self.adopted]:
            mask = self.visibility() <= self.visibility_threshold
            self.aware = (~mask).astype(np.uint32)
            masked_cost = ma.masked_array(
                MODEL.total_cost[self.housing_type, :], mask=mask
            )
            self.adopted = masked_cost.argmin()
            self.heating_unit_age = 0

    def update(self, agent_data):
        """Implements the updat methods, used when modifying
        agents on different ranks
        """
        self.adopted = agent_data[2]
        self.aware = agent_data[1]
        self.heating_unit_age = agent_data[0]

    def save(self):
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data.

        Returns:
            The agent's state
        """
        return (self.uid, self.heating_unit_age, self.aware, self.adopted)


def restore_agent(agent_data):
    """defines a callable to reconstruct ghost agents on other ranks

    Inputs:
        agent_data: tuple, the agents state
    Returns:
        agent: ConsumerAgent, an instance of a consumer agent
    """
    uid = agent_data[0]
    return ConsumerAgent(*uid, *agent_data[1:])


@dataclass
class AdoptionCounts:
    """data class for logging"""

    consumers_aware: int
    potential_adopters: int
    chp_installed: np.uint32
    boiler_installed: np.uint32


class Model:
    """the model class"""

    def __init__(self, comm, params):
        """model initializer"""
        self.context = ctx.SharedContext(comm)
        self.rank = comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)

        # General Parameters
        self.price_of_gas = params["price_of_gas"]
        self.price_of_electricity_purchased = params["price_of_electricity_purchased"]
        self.price_of_electricity_sold = params["price_of_electricity_sold"]
        self.higher_heating_value_natural_gas = params[
            "higher_heating_value_natural_gas"
        ]
        self.market_size = params["market_size"]

        self.housing_types = params["housing_types"]
        self.tech_options = params["tech_options"]

        # Technology (j) specific parameters and variables (t-1)
        self.price_first_unit = np.array(params["price_first_unit"], dtype=np.float64)

        self.age_of_replacement = np.array(
            params["age_of_replacement"], dtype=np.uint32
        )
        self.progress_ratio = np.array(params["progress_ratio"], dtype=np.float64)
        self.growth_rate_hpr_learning = np.array(
            params["growth_rate_hpr_learning"], dtype=np.float64
        )
        self.heat_to_power_ratio_ubound = np.array(
            params["heat_to_power_ratio_ubound"], dtype=np.float64
        )
        self.heat_to_power_ratio_lbound = np.array(
            params["heat_to_power_ratio_lbound"], dtype=np.float64
        )  # ToDo??
        self.technological_lifetime = np.array(
            params["technological_lifetime"], dtype=np.uint32
        )
        self.technology_factor = np.array(params["technology_factor"], dtype=np.float64)
        self.subsidy_for_purchase = np.array(
            params["subsidy_for_purchase"], dtype=np.float64
        )
        self.subsidy_for_usage = np.array(params["subsidy_for_usage"], dtype=np.float64)
        self.subsidy_for_feedback = np.array(
            params["subsidy_for_feedback"], dtype=np.float64
        )  # missing?
        self.advertising_factor = np.array(
            params["advertising_factor"], dtype=np.float64
        )
        self.confidence_in_market = np.array(
            params["confidence_in_market"], dtype=np.float64
        )
        self.cost_of_maintenance = np.array(
            params["cost_of_maintenance"], dtype=np.float64
        )

        # Class (i) specific paramters
        self.gas_consumption_heating = np.array(
            params["gas_consumption_heating"], dtype=np.float64
        )
        self.discount_rate = np.array(params["discount_rate"], dtype=np.float64)
        self.user_horizon = np.array(params["user_horizon"], dtype=np.uint32)
        self.class_size = np.array(params["class_size"], dtype=np.float64)
        self.share_of_electricity_feedback = np.array(
            params["share_of_electricity_feedback"], dtype=np.float64
        )

        # -> derived
        self.num_tech_options = len(params["tech_options"])
        self.idx_chp = self.tech_options.index("micro-CHP")
        self.idx_boiler = self.tech_options.index("condensing boiler")

        self.num_agent_types = len(params["housing_types"])

        self.cum_sales = np.zeros(self.num_tech_options, dtype="uint32")
        self.installed = np.zeros(self.num_tech_options, dtype="uint32")
        self.market_share = np.zeros(self.num_tech_options, dtype="float64")

        self.heat_to_power_ratio = np.zeros(self.num_tech_options, dtype="float64")
        self.gas_consumption_electricity = np.zeros(
            (self.num_agent_types, self.num_tech_options), dtype="float64"
        )

        self.new_sales = np.zeros(self.num_tech_options, dtype="uint32")
        self.purchase_price = np.zeros(self.num_tech_options, dtype="float64")
        self.upfront_cost = np.zeros(self.num_tech_options, dtype="float64")
        self.usage_cost = np.zeros(
            (self.num_agent_types, self.num_tech_options), dtype="float64"
        )
        self.total_cost = np.zeros(
            (self.num_agent_types, self.num_tech_options), dtype="float64"
        )

        # inintialize
        self.cum_sales = np.array(params["initial_cum_sales"], dtype=np.uint32)
        self.installed = np.array(params["initial_cum_sales"], dtype=np.uint32)
        self.consumers_aware = self.cum_sales[self.idx_chp]
        self.market_share = self.cum_sales / self.market_size

        self.initial_price = self.price_first_unit / self.cum_sales ** (
            np.log2(self.progress_ratio)
        )
        self.heat_to_power_ratio = self.heat_to_power_ratio_ubound

        self.update_technology_level()
        self.update_cost()

        # create agent and seed
        self._add_agents(comm, params)
        self._seed_adopters(comm)

        # set up the loggers
        self.counts = AdoptionCounts(
            consumers_aware=sum(
                1
                for agent in self.context.agents()
                if (self.rank == agent.local_rank) and agent.aware[self.idx_chp]
            ),
            potential_adopters=sum(
                1
                for agent in self.context.agents()
                if (self.rank == agent.local_rank)
                and (
                    agent.heating_unit_age
                    == self.age_of_replacement[self.idx_boiler] - 1
                )
            ),
            chp_installed=sum(
                1
                for agent in self.context.agents()
                if (self.rank == agent.local_rank) and (agent.adopted == self.idx_chp)
            ),
            boiler_installed=sum(
                1
                for agent in self.context.agents()
                if (self.rank == agent.local_rank)
                and (agent.adopted == self.idx_boiler)
            ),
        )

        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.diffusion_logger = logging.ReducingDataSet(
            loggers, MPI.COMM_WORLD, params["diffusion_log_file"]
        )

        tech_header = [
            "tick",
            *[self.tech_options[j] for j in np.arange(self.num_tech_options)],
        ]
        cost_header = [
            "tick",
            *[
                (self.housing_types[i], self.tech_options[j])
                for i in np.arange(self.num_agent_types)
                for j in np.arange(self.num_tech_options)
            ],
        ]

        self.price_log = logging.TabularLogger(
            comm, params["price_log_file"], tech_header
        )
        self.heat_to_power_log = logging.TabularLogger(
            comm, params["heat_to_power_log_file"], tech_header
        )

        self.usage_cost_log = logging.TabularLogger(
            comm, params["usage_cost_log_file"], cost_header
        )
        self.total_cost_log = logging.TabularLogger(
            comm, params["total_cost_log_file"], cost_header
        )

        self.diffusion_logger.log(0)

    def _add_agents(self, comm, params):
        """create agents"""
        world_size = comm.Get_size()
        num_agents = self.market_size * self.class_size / 100
        int_num_agents = np.rint(num_agents).astype(np.uint32)

        diff_size = self.market_size - np.sum(int_num_agents)
        if diff_size:
            # re-inintialize
            self.market_size -= int(diff_size)
            self.cum_sales[self.idx_boiler] -= diff_size
            self.installed[self.idx_boiler] -= diff_size
            self.market_share = self.installed / self.market_size

        for agent_type in np.arange(self.num_agent_types):
            agents_per_rank = int_num_agents[agent_type] // world_size
            remainder = int_num_agents[agent_type] % world_size
            if self.rank < remainder:
                agents_per_rank += 1

            for nid in range(agents_per_rank):
                defaults = dict(
                    heating_unit_age=random.default_rng.integers(
                        0, self.age_of_replacement[self.idx_boiler]
                    ),
                    awareness=params["visibility_at_t0"],
                    adopted=self.tech_options.index("condensing boiler"),
                )

                consumer = ConsumerAgent(nid, agent_type, self.rank, **defaults)
                self.context.add(consumer)

    def _seed_adopters(self, comm):
        world_size = comm.Get_size()
        init_chp_sales = self.cum_sales[self.idx_chp]
        adoption_counts = np.zeros(world_size, np.int32)
        if self.rank == 0:
            for _ in range(init_chp_sales):
                idx = random.default_rng.integers(0, high=world_size)
                adoption_counts[idx] += 1

        adoption_count = np.empty(1, dtype=np.int32)
        comm.Scatter(adoption_counts, adoption_count, root=0)

        for agent in self.context.agents(count=adoption_count[0], shuffle=True):
            if self.rank == agent.local_rank:
                agent.adopted = self.idx_chp
                agent.aware = np.ones(self.num_tech_options, dtype=np.uint32)
                agent.heating_unit_age = 0

    def at_end(self):
        """optional function, close the logging file"""
        self.diffusion_logger.close()

    def update_cost(self):
        """updates cost of technologies for the time step"""
        # electricity production E_{ij} -> Eq. inferred
        self.gas_consumption_electricity = (
            np.tile(self.gas_consumption_heating, (self.num_tech_options, 1)).T
            * np.tile(self.technology_factor, (self.num_agent_types, 1))
            / self.heat_to_power_ratio
        )

        # purchase price P_j(t) -> as in Eq. (3)
        self.purchase_price = self.initial_price * self.cum_sales ** (
            np.log2(self.progress_ratio)
        )

        # upfront cost C^f_j -> as documented in text
        self.upfront_cost = self.purchase_price - self.subsidy_for_purchase

        # usage cost C^u_j -> as in Eq. (4)
        for i in np.arange(self.num_agent_types):
            for j in np.arange(self.num_tech_options):
                cost_for_gas = self.price_of_gas * (
                    self.gas_consumption_heating[i]
                    + self.gas_consumption_electricity[i, j]
                )

                if j == self.idx_chp:
                    electricity_produced = (
                        self.gas_consumption_electricity[i, j]
                        * self.higher_heating_value_natural_gas
                    )
                else:
                    electricity_produced = 0

                cost_for_electricity_saved = (
                    self.price_of_electricity_purchased
                    * electricity_produced
                    * (1 - self.share_of_electricity_feedback[i])
                    + (self.price_of_electricity_sold + self.subsidy_for_feedback[j])
                    * electricity_produced
                    * self.share_of_electricity_feedback[i]
                    + self.subsidy_for_usage[j] * electricity_produced
                )

                self.usage_cost[i, j] = sum(
                    cost_for_gas
                    - cost_for_electricity_saved
                    + self.cost_of_maintenance[j] / (1 + self.discount_rate[i]) ** t
                    for t in np.arange(1, self.user_horizon[i] + 1)
                )

        # Total cost -> as in Eq. (2)
        self.total_cost = (
            np.tile(self.upfront_cost, (self.num_agent_types, 1)) + self.usage_cost
        )

    def update_technology_level(self):
        """updates heat-to-power ratio"""
        time = self.runner.schedule.tick
        j = self.tech_options.index("micro-CHP")

        # CHP heat-to-power ratio -> as in Eq. (5)
        self.heat_to_power_ratio[j] = self.heat_to_power_ratio_lbound[j] + (
            self.heat_to_power_ratio_ubound[j] - self.heat_to_power_ratio_lbound[j]
        ) / (
            1
            + np.exp(
                -1
                * self.growth_rate_hpr_learning[j]
                * (time - self.technological_lifetime[j])
            )
        )

    def step(self):
        """mandatory function, defines simulation for each time step"""
        self.counts.potential_adopters = sum(
            1
            for agent in self.context.agents()
            if (self.rank == agent.local_rank)
            and (agent.heating_unit_age == self.age_of_replacement[self.idx_boiler] - 1)
        )

        self.market_share = self.installed / self.market_size

        self.update_technology_level()
        self.update_cost()

        if self.rank == 0:
            tick = self.runner.schedule.tick

            self.price_log.log_row(
                tick,
                *(self.purchase_price[j] for j in np.arange(self.num_tech_options))
            )

            self.heat_to_power_log.log_row(
                tick,
                *(self.heat_to_power_ratio[j] for j in np.arange(self.num_tech_options))
            )

            self.usage_cost_log.log_row(
                tick,
                *(
                    self.usage_cost[i, j]
                    for i in np.arange(self.num_agent_types)
                    for j in np.arange(self.num_tech_options)
                )
            )

            self.total_cost_log.log_row(
                tick,
                *(
                    self.total_cost[i, j]
                    for i in np.arange(self.num_agent_types)
                    for j in np.arange(self.num_tech_options)
                )
            )

        self.price_log.write()
        self.heat_to_power_log.write()
        self.usage_cost_log.write()
        self.total_cost_log.write()

        self.consumers_aware, consumers_aware = 0, 0

        self.new_sales, new_sales = np.zeros(
            self.num_tech_options, dtype="uint32"
        ), np.zeros(self.num_tech_options, dtype="uint32")

        self.installed, installed = np.zeros(
            self.num_tech_options, dtype="uint32"
        ), np.zeros(self.num_tech_options, dtype="uint32")

        for agent in self.context.agents():
            if self.rank == agent.local_rank:

                agent.step()

                if np.isclose(agent.aware[self.idx_chp], 1):
                    consumers_aware += 1

                if (self.runner.schedule.tick > 0) and (agent.heating_unit_age == 0):
                    new_sales[agent.adopted] += 1

                installed[agent.adopted] += 1

        self.context.synchronize(restore_agent)

        comm = self.context.comm
        comm.Allreduce(new_sales, self.new_sales, MPI.SUM)
        comm.Allreduce(installed, self.installed, MPI.SUM)

        self.cum_sales += self.new_sales

        self.counts.consumers_aware = consumers_aware
        self.counts.chp_installed = installed[self.idx_chp]
        self.counts.boiler_installed = installed[self.idx_boiler]
        self.diffusion_logger.log(self.runner.schedule.tick)

    def start(self):
        """mandatory function, starts simulation"""
        self.runner.execute()


def run(params: Dict):
    """set up d model and run the simulation"""
    global MODEL
    MODEL = Model(MPI.COMM_WORLD, params)
    MODEL.start()


if __name__ == "__main__":

    parser = parameters.create_args_parser()
    args = parser.parse_args()
    model_params = parameters.init_params(args.parameters_file, args.parameters)
    assert model_params["market_size"] == sum(model_params["initial_cum_sales"])

    run(model_params)
