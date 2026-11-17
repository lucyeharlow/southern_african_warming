#!/bin/bash --login
#SBATCH --job-name=temperature_levels_monthly
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=125gb
#SBATCH --time=0-06:00:00
#SBATCH --partition=Short
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/southern_african_warming/src/logs/temperature_levels_monthly_%j.out
#SBATCH --error=/soge-home/users/jesu4127/southern_african_warming/src/logs/temperature_levels_monthly_%j.err

set -euo pipefail

echo "============================================"
echo "CHIRTS overlay job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================"

WORKDIR="/soge-home/users/jesu4127/southern_african_warming/src"
/hn01-home/jesu4127/micromamba/envs/myenv/bin/python "$WORKDIR/temperature_levels_monthly.py"

echo
echo "Python version:"
python --version

echo
echo "Starting analysis..."
echo "============================================"

python "$WORKDIR/temperature_levels_monthly.py"

echo
echo "============================================"
echo "Finished: $(date)"
echo "============================================"