#!/bin/bash --login
#SBATCH --job-name=omega_cf_sensitivity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=Long
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=katcutain@hotmail.com
#SBATCH --output=/soge-home/users/jesu4127/southern_african_warming/src/logs/omega_cf_sensitivity_%j.out
#SBATCH --error=/soge-home/users/jesu4127/southern_african_warming/src/logs/omega_cf_sensitivity_%j.err

set -euo pipefail

echo "============================================"
echo "OMEGA_CF_SENSITIVITY job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "============================================"

WORKDIR="/soge-home/users/jesu4127/southern_african_warming/src"
/hn01-home/jesu4127/micromamba/envs/myenv/bin/python "$WORKDIR/omega_cf_sensitivity.py" 

echo
echo "Python version:"
python --version

echo
echo "Starting analysis..."
echo "============================================"

python "$WORKDIR/omega_cf_sensitivity.py"

echo
echo "============================================"
echo "Finished: $(date)"
echo "============================================"