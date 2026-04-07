"""setup.py

Copyright 2026 Marvin Meck

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

import os
from pathlib import Path
import sqlite3
import importlib.util
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build_py import build_py


def run_pre_install():
    root = os.path.abspath(os.path.dirname(__file__))

    # Get absolute path to codata.py
    module_path = os.path.join(root, "scripts", "codata.py")
    spec = importlib.util.spec_from_file_location("codata", module_path)
    codata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codata)

    # call codata.main
    codata.main(Path(root))


def run_post_install():

    root = os.path.abspath(os.path.dirname(__file__))

    # Get absolute path to thermotables.py
    module_path = os.path.join(root, "scripts", "thermotables.py")
    spec = importlib.util.spec_from_file_location("thermotables", module_path)
    thermotables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(thermotables)

    # call thermotables.main
    thermotables.main(
        file_index=Path(os.path.join(root, "phd-data", "nist-janaf", "file_index.csv")),
        out_dir=Path(os.path.join(root, "phd-data", "nist-janaf")),
        flag_local=False,
    )

    # create views
    dbfile = os.path.join(
        root, "phd-data", "nist-janaf", "nist_janaf_thermochemical_tables.sqlite"
    )
    fname = os.path.join(root, "sql", "view_std_gibbs_free_energy.sql")

    with sqlite3.connect(dbfile) as con:
        query = "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='std_gibbs_free_energy'"
        cur = con.cursor()
        res = cur.execute(query)
        (has_view,) = res.fetchone()
        if not has_view:
            with open(fname, "r") as f:
                query = f.read()
            cur.execute(query)


class BuildPyCommand(build_py):
    def run(self):
        run_pre_install()
        build_py.run(self)


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        run_post_install()


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        run_post_install()


setup(
    name="phdtools",
    version="1.0.0",
    setup_requires=["requests>=2.32.4", "numpy>=2.2.6", "pandas>=2.3.1"],
    install_requires=[
        "ipython>=9.4.0",
        "ipykernel>=6.30.1",
        "ipywidgets>=8.1.7",
        # "jupyterlab>=4.4.5",
        # "mpi4py>=4.1.0",
        "matplotlib>=3.10.5",
        "nbconvert>=7.16.6",
        "notebook>=7.4.5",
        "numpy>=2.2.6",
        "pandas>=2.3.1",
        "pyomo>=6.9.5",
        "PyYAML>=6.0.2",
        # "pyoptdb @ file:deps/pyoptdb",
        "scikit-learn>=1.7.1",
        "scipy>=1.16.1",
        "sqids>=0.5.2",
        # "repast4py>=1.1.6"
    ],
    packages=find_packages(),
    cmdclass={
        "build_py": BuildPyCommand,
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
)
