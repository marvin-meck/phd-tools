#!/bin/bash


PROJECT_ROOT=$(python3 - <<EOF
from phdtools import PROJECT_ROOT
print(PROJECT_ROOT)
EOF
)

TMPDIR=$(python3 - <<EOF
from phdtools import TMP_DIR
print(TMP_DIR)
EOF
)

SRC_DIR=$TMPDIR/chemical_equilibrium_runs
# DST_DIR=.pyoptdb/chemical-equilibrium

# cp chemical_equilibrium_ideal.py $DST_DIR/chemical_equilibrium_ideal.py

no_files=$(ls $SRC_DIR/*.dat | wc -l )

declare -i i=1

echo "adding to database..."
for dat_file in $SRC_DIR/*dat; do
     echo "$i/$no_files"
    _id=$(basename $dat_file .dat)
    # mkdir $DST_DIR/$_id 
    sol_file=$SRC_DIR/$_id.yml
    # cp --no-preserve=mode,ownership,timestamp {$dat_file,$sol_file} $DST_DIR/$_id 
    pyoptdb insert --model-file $PROJECT_ROOT/phdtools/models/white_dantzig_1958.py --dat-file $dat_file $sol_file
    i+=1
done
echo "...done!"

# rm -rf $SRC_DIR
