"""phdtools.abm.model.py

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

Repast4py implementation of an ABM for the adoption of fuel-cell CHP technology. 

Example:
    To run a simulation : 
        $ mpirun -n=2 ptyhon3 model.py params.yml

Notice:
    Copyright 2023 Technical University Darmstadt

    Please, refer to the LICENSE file in the root directory.

    Author: Marvin Meck
    E-mail: marvin_maximilian.meck@tu-darmstadt.de

    Corresponding: Peter Pelz
    E-mail: peter.pelz@tu-darmstadt.de
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict
import logging as pylogging
import warnings

from mpi4py import MPI
import numpy as np
from repast4py import context, random, parameters, schedule, logging

from phdtools.abm.agents import RandomConsumer as ConsumerAgent, TechnologyType

warnings.warn(
    "phdtools.abm.model is unused and may be removed in a future release.",
    FutureWarning,
    stacklevel=2
)

pylogger = pylogging.getLogger("model")
console_handler = pylogging.StreamHandler()

pylogger.addHandler(console_handler)
pylogger.setLevel(pylogging.DEBUG)


OUT_DIR = Path("./out/")


agent_cache = {}


def restore_agent(agent_data: Tuple):
    """defines a callable to reconstruct ghost agents on other ranks

    Args:
        agent_data (Tuple): the agents state

    Returns:
        ConsumerAgent, an instance of a consumer agent
    """
    uid = agent_data[0]
    if uid[1] == ConsumerAgent.type:
        if uid in agent_cache:
            c = agent_cache[uid]
        else:
            c = ConsumerAgent(
                _id=uid[0],
                _rank=uid[2],
                # heating_unit=agent_data[1],
                adopted=agent_data[2],
                choice_prob=agent_data[3]
            )
            c.heating_unit.age = agent_data[1].age
            c.heating_unit.life_expectancy = agent_data[1].life_expectancy
            c.heating_unit.technology = agent_data[1].technology
    else:
        raise NotImplementedError()

    return c


@dataclass
class AdoptionCounts:
    # sales: int
    new_adopters: np.int32
    # cumulative_sales: int
    cumulative_adopters: np.int32
    installed_base: np.int32


class Model:
    """An agent-based simulation model"""

    def __init__(self, comm: MPI.COMM_WORLD, params: Dict) -> None:
        """Model constructor.

        Args:
            comm (mpi4py.MPI.COMM_WORLD): MPI Communicator
            params (Dict): model paramters dictionary

        Todos:
            can we use mpi4py.MPI.Graphcomm?
        """
        self.rank = comm.Get_rank()

        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        # self.runner.schedule_repeating_event(1.1, 10, self.log_agents) TODO create a logger
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)

        # create the context to hold agents and manage cross process synchronization
        self.context = context.SharedContext(comm)

        # create the agents and add to context
        self.adopters = []
        self._create_agents(comm, params)

        # seed parameter values
        self._seed_agents()

        # set up log-file
        self.counts = AdoptionCounts(new_adopters=0, cumulative_adopters=0, installed_base=0)
        repast_loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)

        fname = OUT_DIR / "results_{}_agent_based_model.csv".format(
            datetime.today().strftime("%y%m%d")
        )
        self.data_set = logging.ReducingDataSet(repast_loggers, comm, fname)
        self.data_set.log(0)

    def _create_agents(self, comm, params):
        size = comm.Get_size()

        # calculate the number of agents to create per rank
        num_agents = params["num_agents"]
        agents_per_rank = num_agents // size
        remainder = num_agents % size
        if self.rank < remainder:
            agents_per_rank += 1
        pylogger.debug(f"creating {agents_per_rank} agents on rank {self.rank}")

        # add agents to context
        for i in np.arange(agents_per_rank):
            c = ConsumerAgent(_id=i, _rank=self.rank, choice_prob=0.25)
            c.heating_unit.life_expectancy = 0
            self.context.add(agent=c)

    def _seed_agents(self):
        pass
        # rng = random.default_rng
        # for agent in self.context.agents():
            # agent.heating_unit.age = rng.uniform()*agent.heating_unit.life_expectancy
            # agent.heating_unit.age = rng.normal(loc=agent.heating_unit.life_expectancy,scale=5)

    def at_end(self):
        self.data_set.close()

    def step(self):
        """performs one simulation time step"""
        rng = random.default_rng
        new_adopters = []
        has_fuelcell = []
        for agent in self.context.agents():
            if (agent.local_rank == self.rank):
                if (agent.heating_unit.age >= agent.heating_unit.life_expectancy):
                    agent.heating_unit.technology = rng.choice(list(TechnologyType), replace=False, p=[agent.choice_prob,1-agent.choice_prob])
                    # TODO implement choice_prob as array to go beyond binary choice situations
                    if (agent.heating_unit.technology == TechnologyType.FUEL_CELL) and (not agent.adopted): 
                            agent.adopted = True
                            new_adopters.append(agent)
                    else:
                        pass

                    agent.heating_unit.age = 0
                else:
                    agent.heating_unit.age += 1

                if agent.heating_unit.technology == TechnologyType.FUEL_CELL:
                    has_fuelcell.append(agent)

        # log
        self.adopters += new_adopters
        self.counts.new_adopters = len(new_adopters)
        self.counts.cumulative_adopters += self.counts.new_adopters
        self.counts.installed_base = len(has_fuelcell)

        # pylogger.debug(f"rank: {self.rank}; new adopters: {self.counts.new_adopters}")
        self.data_set.log(self.runner.schedule.tick)

        # sync processes
        self.context.synchronize(restore_agent)

    def start(self):
        """starts model execution"""
        self.runner.execute()


def run(params: Dict) -> None:
    """Initialize and run the model.

    Args:
        params (Dict): model paramters dictionary.
    """

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    pylogger.debug(f"initializing the model on rank {rank} with size {size}")
    model = Model(comm=MPI.COMM_WORLD, params=params)
    pylogger.debug(f"runnig the model on rank {rank}...")
    model.start()
    pylogger.debug(f"done! (rank: {rank})")


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
