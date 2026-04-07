"""phdtools.plots.__init__.py

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

import matplotlib as mpl
import matplotlib.style

matplotlib.style.use("grayscale")
mpl.rcParams["figure.facecolor"] = "white"

# mpl.rcParams["axes.spines.left"] = True
# mpl.rcParams["axes.spines.right"] = False
# mpl.rcParams["axes.spines.top"] = False
# mpl.rcParams["axes.spines.bottom"] = True

mpl.rcParams["legend.frameon"] = True
mpl.rcParams["legend.edgecolor"] = "white"
mpl.rcParams["legend.facecolor"] = "white"

mpl.rcParams["grid.color"] = "gray"
mpl.rcParams["grid.linestyle"] = "dotted"
mpl.rcParams["grid.linewidth"] = 0.5
