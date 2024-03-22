#!/bin/bash -eux
#SBATCH --job-name=job
#SBATCH --partition=cpu # -p
#SBATCH --cpus-per-task=22 # -c
#SBATCH --mem=80gb
#SBATCH --time=48:00:00 # 48 hours
#SBATCH --output=logs/job_%j.log # %j is job id

conda run -n fedout-det python3 test.py