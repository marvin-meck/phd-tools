"""phdtools.data.__init__.py

Copyright 2025 Marvin Meck

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

from enum import Enum

ISO_STD_REF_TEMPERATURE_SI = 288.15
ISO_STD_REF_PRESSURE_SI = 1.01325e5
ISO_STD_REF_REL_HUMIDITY = 0.6

Reaction = Enum(
    "Reaction", ["SMR", "WGS", "DSR", "MCR1", "MCR2", "HCR1", "HCR2"], start=0
)

Compound = Enum(
    "Compound",
    [
        "C1H4(g)",
        "C1O1(g)",
        "C1O2(g)",
        "H2(ref)",
        "H2O1(g)",
        "H2O1(l)",
        "N2(ref)",
        "O2(ref)",
    ],
    start=0,
)
