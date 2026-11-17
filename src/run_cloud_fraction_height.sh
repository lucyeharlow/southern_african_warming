#!/bin/bash
# run_cloud_monthly.sh: global monthly cloud fraction trends (cloud_fraction_trends_monthly.py),
# one SLURM array task per calendar month.
#
# First fill the shared caches with one month on its own, then the rest as an array:
#   sbatch --array=10 run_cloud_monthly.sh
#   sbatch --array=1-9,11-12 run_cloud_monthly.sh
# Plot all months with one colour scale once every month is done:
#   python cloud_fraction_trends_monthly.py plot

#SBATCH --job-name=cloud_monthly
#SBATCH --partition=Long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/soge-home/users/jesu4127/southern_african_warming/src/logs/cloud_monthly_%A_%a.log

set -uo pipefail
mkdir -p logs
cd "$(dirname "$0")" 2>/dev/null || true

PY=${PY:-$HOME/micromamba/envs/myenv/bin/python}
[ -x "$PY" ] || { echo "no Python at $PY: set PY=/path/to/python"; exit 1; }
export SAW_WORKERS=${SAW_WORKERS:-4}

echo "Started $(date) on $(hostname), month ${SLURM_ARRAY_TASK_ID:-all}, $PY"
"$PY" /soge-home/users/jesu4127/southern_african_warming/src/cloud_fraction_height_trends.py compute ${SLURM_ARRAY_TASK_ID:-}
echo "Finished $(date)"