"""phdtools.integrate.py

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

from dataclasses import dataclass
import numpy as np
from scipy.optimize import newton


@dataclass()
class Solution:
    t: np.ndarray
    y: np.ndarray


def euler(
    fun, t_span, y0, method="backward", n=20, maxiter=100, t_eval=None, args=None
):
    """Solve an initial value problem for a set of ODEs using the Euler method.

    Parameters
    ----------
    fun : callable
        f(t, y, *args) -> array_like with same shape as y
    t_span : (t0, tf)
    y0 : array_like or scalar
    method : "forward" or "backward"
    n : number of steps
    args : extra args passed to fun
    """
    if np.size(y0) == 0:
        m = 0
    else:
        m = len(y0)

    t0, tf = t_span
    t = np.linspace(t0, tf, n + 1)
    h = (tf - t0) / n

    y = np.zeros((m, n + 1))
    y[:, 0] = y0

    for k in range(n):
        if method == "forward":
            y[:, k + 1] = y[:, k] + h * fun(t, y, *args)[:, k]
        elif method == "backward":
            res = (
                lambda z: (z - y[:, k])
                - h * fun(t[k + 1], np.atleast_2d(z).T, *args)[:, 0]
            )
            r = newton(res, y[:, k], full_output=True, maxiter=maxiter)
            # print(r[0])
            y[:, k + 1] = r[0]
        else:
            raise ValueError(
                f"Unknown method {method}. Use either 'forward' or 'backward'"
            )

    return Solution(t, y)
