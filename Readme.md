# `phd-tools`

This repository contains supplementary material for the study:
> Meck, Marvin (2026), "Choice-based engineering design optimization: 
a comparative analysis with conventional cost minimization applied to the design of a fuel cell micro co-generation system", 
PhD thesis (in preparation), TU Darmstadt.

It includes the code and scripts used to generate the results reported in the study.
The results can be found here: https://github.com/marvin-meck/phd-results/tree/main

To reproduce these results or generate new ones, `phd-tools` requires additional data from third-party sources. 
These data are available either as a `git submodule` or under https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/5082. 
Due to licensing restrictions, these data cannot be distributed publicly. 
Access may be obtained from the respective rights holders upon reasonable request.

If you have access to the required data, `phd-tools` expects them to be located in a directory accessible at runtime.
By default, this directory is `<project root>/phd-data`. 
This location can be changed by setting the environment variable `PHDTOOLS_DATA_DIR` or by specifying `data_dir` in `config.ini`. 

The same mechanism is used to configure the directories for results and temporary files. 
By default:
* Results: `<project root>/phd-results`
* Temporary files: `<project root>/tmp`

These can be changed via the environment variables `PHDTOOLS_RESULTS_DIR` and `PHDTOOLS_TMP_DIR`, or by specifying `results_dir` and `tmp_dir` in `config.ini`. 

## Generate results 

`phd-tools` provides multiple Jupyter Notebooks through which the results are generated. 
To execute all Jupyter Notebooks in batch, run the shell script `run.sh` from the project root:
```
source ./run.sh
```
For a description of available options:
```
source ./run.sh --help
```
To reproduce the results exactly: 
```
source ./run.sh -r -s
```

## Installation

Create a new environment
```
python3 -m venv "phd-env"
source ./phd-env/bin/activate
```

Download and install
```
git clone git@github.com:marvin-meck/phd-tools.git
cd phd-tools
git submodule init
git submodule update
python -m pip install --upgrade pip setuptools wheel
python -m pip install deps/pyoptdb
python -m pip install -e .
```

### Third party dependencies

The applications within this source tree depend on several third-party software modules (the "dependencies"). 
Use of these dependencies is subject to the terms and conditions of their respective licenses.
A list of primary dependencies, including verbatim copies of their licenses and copyright notices, is provided in: 
`<project root>/license_dependencies`. 
The primary dependencies include

| Dependency    | Version     | License                                                    |
|---------------|-------------|------------------------------------------------------------|
| ipython       | 9.4.0       | [BSD 3-Clause](license_dependencies/IPYTHON_LICENSE)       |
| ipykernel     | 6.30.1      | [BSD 3-Clause](license_dependencies/IPYKERNEL_LICENSE)     |
| ipywidgets    | 8.1.7       | [BSD 3-Clause](license_dependencies/IPYWIDGETS_LICENSE)    |
| matplotlib    | 3.10.5      | [BSD-style](license_dependencies/MATPLOTLIB_LICENSE)       |
| nbconvert     | 7.16.6      | [BSD 3-Clause](license_dependencies/NBCONVERT_LICENSE)     |
| notebook      | 7.4.5       | [BSD 3-Clause](license_dependencies/NOTEBOOK_LICENSE)      |
| numpy         | 2.2.6       | [modified BSD](license_dependencies/NUMPY_LICENSE)         |
| pandas        | 2.3.1       | [BSD 3-Clause](license_dependencies/PANDAS_LICENSE)        |
| Pyomo         | 6.9.5       | [BSD-style](license_dependencies/PYOMO_LICENSE)            |
| pyoptdb       | 0.1.1-alpha | [Apache License 2.0](license_dependencies/PYOPTDB_LICENSE) |
| PyYAML        | 6.0.2       | [MIT](license_dependencies/PYYAML_LICENSE)                 |
| scikit-learn  | 1.7.1       | [BSD 3-Clause](license_dependencies/SCIKIT-LEARN_COPYING)  |
| scipy         | 1.16.1      | [BSD 3-Clause](license_dependencies/SCIPY_LICENSE)         |
| sqids         | 0.5.2       | [MIT](license_dependencies/SQIDS_LICENSE)                  |

To run all scripts (Jupyter Notebooks) the following third-party applications are further required:
* [gnuplot 6.0.3](https://sourceforge.net/projects/gnuplot/files/gnuplot/6.0.3/) ([Gnuplot license (permissive)](license_dependencies/GNUPLOT_COPYRIGHT))
* [Ipopt 3.14.19](https://github.com/coin-or/Ipopt/releases/tag/releases%2F3.14.19) ([Eclipse Public License 2.0](license_dependencies/IPOPT_LICENSE))
* [SCIP 10.0.1](https://github.com/scipopt/scip/releases/tag/v10.0.1) ([Apache License 2.0](license_dependencies/SCIP_LICENSE))

These are not listed as strict dependencies, however, because they may be replaced with other optimization solvers and plotting software depending on the use case. 

## Data Identification and Organization

All data generated and used during this research project is systematically organized using a structured identification scheme. 
Each dataset, figure, or table is assigned a **Data ID**, a 3-tuple of integers of the form:
```
(type, chapter, counter)
```

To produce a compact and readable identifier for use in file names and references, each `(DataType, Chapter, Counter)` tuple is encoded using the [`sqids`](https://sqids.org) algorithm.
[`Sqids`](https://sqids.org) converts the three-integer tuple into a short, unique, non-sequential string identifier. 
For example:
* The tuple `(0, 3, 1)` might become `gVHrJE`, and
* the tuple `(1, 3, 1)` might become `fG9gW1`.

The components of the tuple identifying the datasets are defined as follows. 
`DataType` indicates the type of the data, and is defined to take the following values:
* `0` - Support: Intermediate or supporting data used in modelling or analysis, but not directly associated with any particular figure or table. 
* `1` - Figure: Data used to produce figures. 
* `2` — Table: Data used to generate tables. 

`Chapter` indicates the chapter in which the data is used or discussed, 
numbered as:
* `1` — Introduction
* `2` — Literature Review
* `3` — Methods
* `4` — Results
* `5` — Discussion
* `6` — Conclusion
* `7` — Appendix

Lastly, `Counter` is an integer value incremented for datasets of the same type within the same chapter. 

This system ensures that all generated data can be uniquely identified, consistently named, and easily traced back to its corresponding context in the thesis. 
It supports reproducibility, improves organization, and facilitates automated referencing and file management. 

## Usage

Licensed under the Apache License 2.0.
See: https://www.apache.org/licenses/LICENSE-2.0

If you use this material, please cite:

> Meck, Marvin (2026), "Choice-based engineering design optimization: a comparative analysis with conventional cost 
minimization applied to the design of a fuel cell micro co-generation system", PhD thesis (in preparation), TU Darmstadt.

## Contact

For questions or data access inquiries, please open an issue or contact the author.
