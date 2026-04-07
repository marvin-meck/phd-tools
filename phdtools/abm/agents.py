"""phdtools.abm.agents.py

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

"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import warnings

import numpy as np
from repast4py import core


warnings.warn(
    "phdtools.abm.agents is unused and may be removed in a future release.",
    FutureWarning,
    stacklevel=2
)

class TechnologyType(Enum):
    FUEL_CELL = 1
    OTHER = 2


@dataclass
class HeatingUnit:
    technology: TechnologyType
    age: int = 0
    life_expectancy: int = 15


# class BaseConsumer():
#     def __init__(self, heating_technology: HeatingTechnology = HeatingTechnology.OTHER) -> None:
#         self.heating_technology: HeatingTechnology = heating_technology

#     def update(self, heating_technology: HeatingTechnology):
#         self.heating_technology: HeatingTechnology = heating_technology


class RandomConsumer(core.Agent):
    """A Consumer Agent with fully (quasi) random consumer behavior

    Attributes:
        heating_technology (HeatingTechnology): the current technology used
        adopted (bool): an indicator variable
        choice_prob (np.float64): choice probability

    Todo:
        extend choice probability to multinominal choice situations
    """

    type = 0

    def __init__(
        self,
        _id: int,
        _rank: int,
        # heating_unit: HeatingUnit = HeatingUnit(technology=TechnologyType.OTHER),
        adopted: bool = False,
        choice_prob: np.float64 = 0.5,
    ) -> None:
        super().__init__(id=_id, type=RandomConsumer.type, rank=_rank)
        self.heating_unit = HeatingUnit(technology=TechnologyType.OTHER)
        self.adopted = adopted
        self.choice_prob = choice_prob

    def save(self) -> Tuple:
        """Saves the state of this agent as tuple.

        A non-ghost agent will save its state using this
        method, and any ghost agents of this agent will
        be updated with that data.

        Returns:
            Tuple: The agent's state.
        """
        return (self.uid, self.heating_unit, self.adopted, self.choice_prob)

    # def update(self, data: Tuple) -> None:
    #     """Implements the update methods, used when modifying
    #         agents on different ranks

    #     Args:
    #         data (Tuple): Agent data.
    #     """
    #     self.heating_unit = data[1]
    #     self.adopted = data[2]
    #     self.choice_prob = data[3]


