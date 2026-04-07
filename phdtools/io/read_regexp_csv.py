"""phdtools.io.read_regexp_csv.py

Copyright 2024 Technical University Darmstadt

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

import re
import numpy as np
import pandas as pd


def read_regexp_csv(fname, regexp, header=True, skiprows={}):

    columns = []
    if header:
        skiprows = list(skiprows)
        skiprows.append(0)

    prog = re.compile(regexp)

    data = []
    with open(fname, "r") as f:
        for num, line in enumerate(f.readlines()):
            if num in skiprows:
                if num == 0 and header:
                    columns = line.strip().split(",")
                else:
                    continue
            else:
                line_data = []
                for val in line.strip().split(","):
                    m = prog.match(val)
                    if not m is None:
                        line_data.append(m.group(0))
                    else:
                        line_data.append(np.nan)

                data.append(line_data)

    return pd.DataFrame(data, dtype=float, columns=columns)
