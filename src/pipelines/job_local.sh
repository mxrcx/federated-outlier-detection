#!/bin/bash -eux
#SBATCH --job-name=job
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.schaarschmidt@student.hpi.de
#SBATCH --partition=cpu # -p
#SBATCH --cpus-per-task=10 # -c
#SBATCH --mem=32gb
#SBATCH --time=01:00:00 # 48 hours
#SBATCH --output=logs/job_%j.log # %j is job id

conda run -n fedout-det python3 local.py