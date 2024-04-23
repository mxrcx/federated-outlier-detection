#!/bin/bash -eux
#SBATCH --job-name=job
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.schaarschmidt@student.hpi.de
#SBATCH --partition=cpu # -p
#SBATCH --cpus-per-task=20 # -c
#SBATCH --mem=80gb
#SBATCH --time=01:00:00 # 48 hours
#SBATCH --output=logs/job_%j.log # %j is job id

# Get the command-line argument
arg="$1"

conda run -n fedout-det python3 analysis.py "$arg"