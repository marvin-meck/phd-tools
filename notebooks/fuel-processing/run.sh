#!/bin/bash

PROJECT_ROOT=$(python3 - <<EOF
from phdtools import PROJECT_ROOT
print(PROJECT_ROOT)
EOF
)

DATA_DIR=$(python3 - <<EOF
from phdtools import DATA_DIR
print(DATA_DIR)
EOF
)

TMPDIR=$(python3 - <<EOF
from phdtools import TMP_DIR
print(TMP_DIR)
EOF
)

echo "running case builder..."
../../scripts/cep_case_builder.py --out-dir $TMPDIR/chemical_equilibrium_runs --data-base $DATA_DIR/nist-janaf/nist_janaf_thermochemical_tables.sqlite ./cases.yml
echo "...done!"

no_files=$(find "$TMPDIR/chemical_equilibrium_runs" -type f | wc -l )

declare -i i=1

SAV_DIR=$(pwd)
echo $SAV_DIR

cd $TMPDIR/chemical_equilibrium_runs

echo "solving..."
for filename in *.dat; do
    echo "$i/$no_files"
    pyomo solve --solver=ipopt --save-results=${filename%.dat}.yml $PROJECT_ROOT/phdtools/models/white_dantzig_1958.py $filename > /dev/null
    i+=1
done
cd $SAV_DIR

echo "...done!"