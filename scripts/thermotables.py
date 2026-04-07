"""scripts/thermotables.py

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

from argparse import ArgumentParser, BooleanOptionalAction

# from datetime import datetime
import os
from pathlib import Path
import requests
import sqlite3

import numpy as np
import pandas as pd

from phdtools import DATA_DIR

# TODAY = datetime.today().strftime("%y%m%d")


def main(file_index, out_dir, flag_local=False):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        Warning("{} already exists!".format(out_dir))

    df_file_index = pd.read_csv(file_index, sep=",")

    if not flag_local:
        if not os.path.exists(out_dir / "tables"):
            os.mkdir(out_dir / "tables")

        for name in df_file_index["File"]:
            fname = name + ".txt"
            if not os.path.exists(out_dir / "tables" / fname):
                r = requests.get(f"https://janaf.nist.gov/tables/{fname}")
                with open(out_dir / "tables" / fname, "w+") as f:
                    f.write(r.text)

        if not os.path.exists(out_dir / "name.txt"):
            r = requests.get("https://janaf.nist.gov/name.html")
            idx_from = r.text.find(r"<pre>")
            idx_to = r.text.find(r"</pre>")
            with open(out_dir / "name.txt", "w") as f:
                f.write(r.text[idx_from + 6 : idx_to])
    else:
        # TODO make sure all the files are present
        pass

    with sqlite3.connect(out_dir / "nist_janaf_thermochemical_tables.sqlite") as con:

        cur = con.cursor()

        # create compounds table
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='compounds';")
        if res.fetchone() is None:
            df = pd.read_fwf(
                out_dir / "name.txt",
                widths=[6, 29, 46],
                names=["JCODE", "FORMULA", "NAME"],
                skiprows=[0],
                index_col=0,
            )
            _ = df.to_sql("compounds", con, if_exists="fail", index=True)

        cur = con.cursor()

        # create thermo_tables table
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='thermo_tables';")
        if res.fetchone() is None:
            _ = cur.execute(
                'CREATE TABLE thermo_tables("jcode","T(K)","Cp","S","-[G-H(Tr)]/T","H-H(Tr)","delta-f H","delta-f G","log Kf");'
            )

        for name in df_file_index["File"]:
            # logging.debug(f"adding {name}")
            fname = out_dir / "tables" / (name + ".txt")
            with open(fname) as f:
                line = f.readline()

                # get the jcode
                _, formula = line.strip().split("\t")
                res = cur.execute(
                    r"SELECT jcode FROM compounds WHERE formula IS " + f"'{formula}'"
                )
                res = res.fetchone()

                if not res is None:
                    jcode = res[0]
                    # check if db already contains the table
                    # print(r"SELECT jcode FROM thermo_tables WHERE jcode=" + f"{jcode}")
                    res = cur.execute(
                        r"SELECT jcode FROM thermo_tables WHERE jcode=" + f"{jcode}"
                    )
                    has_entries = res.fetchone() is not None
                else:
                    has_entries = False

                if not has_entries:
                    if fname.stem == "H-063":
                        # TODO: H-063 is corrupt when obtained from the website and needs special
                        # handling of row 11 (372.780 K), row 12 (380 K) is not consistently tab-delimited
                        # and presumably misses minus signs for the formation enthalpy and Gibbs energy
                        df = pd.read_csv(
                            fname,
                            sep="\s+",
                            names=[
                                "T(K)",
                                "Cp",
                                "S",
                                "-[G-H(Tr)]/T",
                                "H-H(Tr)",
                                "delta-f H",
                                "delta-f G",
                                "log Kf",
                            ],
                            header=None,
                            skiprows=[0, 1, 2, 3, 4, 11, 12],
                        )
                    else:
                        df = pd.read_csv(fname, sep="\t", skiprows=[0])

                    df["jcode"] = jcode * np.ones(df.shape[0], dtype=np.int32)
                    _ = df.to_sql("thermo_tables", con, if_exists="append", index=False)


if __name__ == "__main__":

    parser = ArgumentParser(description="creates thermochemical tables database")

    parser.add_argument(
        "-l",
        "--local",
        dest="FLAG_LOCAL",
        action=BooleanOptionalAction,
        required=False,
        default=True,
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="OUT_DIR",
        type=Path,
        default=Path(f"{DATA_DIR}/nist-janaf"),
    )
    parser.add_argument(
        "-f",
        "--file-index",
        dest="FILE_INDEX",
        type=Path,
        default=Path(f"{DATA_DIR}/nist-janaf/file_index.csv"),
    )

    args = parser.parse_args()

    main(file_index=args.FILE_INDEX, out_dir=args.OUT_DIR, flag_local=args.FLAG_LOCAL)
