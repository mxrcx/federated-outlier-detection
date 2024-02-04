#!/bin/bash -eux
#SBATCH --job-name=job
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.schaarschmidt@student.hpi.de
#SBATCH --partition=cpu # -p
#SBATCH --cpus-per-task=58 # -c
#SBATCH --mem=300gb
#SBATCH --time=48:00:00 # 48 hours
#SBATCH --output=logs/job_%j.log # %j is job id

conda run -n fedout-det python3 cml_loho_split.py