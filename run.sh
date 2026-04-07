#!/bin/bash

RESULTS_DIR=$(python3 - <<EOF
from phdtools import RESULTS_DIR
print(RESULTS_DIR)
EOF
)

show_help() {
echo "Usage: $(basename "$0") [OPTIONS]"
echo 
echo "Options:"
echo "  -c, --clear        Clear all"
echo "  -r, --reproduce    Run with fixed random seeds to generate reproducible results"
echo "  -s, --sort-index   Sort the index file after running all scripts"
echo "  -h, --help         Show this help message and exit"
echo 
echo "Example:"
echo "  ./run.sh"
echo "  ./run.sh --reproduce"
echo 
}

sortidx=0
reproduce=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            return 0
            ;;
        -c|--clear)
            clear=1
            rm -rf $RESULTS_DIR
            return 0
            ;;
        -r|--reproduce)
            reproduce=1
            shift
            ;;
        -s|--sort-index)
            sortidx=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Invalid option: $1" >&2
            show_help
            return 1
            ;;
        *)
            break
            ;;
    esac
done

if [ "$reproduce" = 1 ]; then
    echo "NOTE: Running with fixed random seeds to ensure reproducible results!"
    export TIME_LIMIT_SECONDS=30
    export SEED_AGENTS=1000
    export SEED_COEFFICIENTS=1000
    export SEED_REGRESSION=1000
else
    echo "Running in random mode"
fi



export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/thermo-data/thermo-data.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/consumer-preferences/innovation-diffusion.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/fuel-processing/chemical-equilibrium.ipynb
export FILE_DATE=$(date +"%y%m%d") && export FILE_DATE_AGENTS=$(date +"%y%m%d") && export FILE_DATE_SAMPLE=$(date +"%y%m%d") && export FILE_DATE_REGRESSION=$(date +"%y%m%d") && jupyter execute notebooks/consumer-preferences/discrete-choice.ipynb 
export FILE_DATE=$(date +"%y%m%d") && export FILE_DATE_DEMANDS=$(date +"%y%m%d") && jupyter execute notebooks/design-optimization/consumer-preference-expressions.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/fuel-processing/steam-reforming.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/fuel-processing/water-gas-shift.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/fuel-cell-modelling/fuel-cell.ipynb
export FILE_DATE=$(date +"%y%m%d") && export FILE_DATE_COST_COEFS=$(date +"%y%m%d") && jupyter execute notebooks/design-optimization/cost-modelling.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/design-optimization/fesibility-problems.ipynb
export FILE_DATE=$(date +"%y%m%d") && export FILE_DATE_COST_MINIMIZATION=$(date +"%y%m%d") && jupyter execute notebooks/design-optimization/cost-minimization.ipynb
export FILE_DATE=$(date +"%y%m%d") && jupyter execute notebooks/design-optimization/choice-based-optimization.ipynb

if [ "$sortidx" = 1 ]; then
    echo "NOTE: Sorting the index!"
    python scripts/sort_index.py 
else
    
fi
echo "Done!"