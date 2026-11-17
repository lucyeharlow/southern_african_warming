#!/bin/bash
# run_hourly.sh: ERA5 hourly October 12-hour composites (hourly_composites.py).
# The reading is done in parallel over years, one worker per CPU (readers.workers()).
# A variable-year is cached once read, so a job that runs out of time can simply be
# resubmitted and carries on.
#
#   sbatch run_hourly.sh                         read, compute and plot
#   sbatch run_hourly.sh read                    only read the hourly files into the cache
#   sbatch run_hourly.sh all --vars t2m,w500     a subset of the variables
#

#SBATCH --job-name=hourly12
#SBATCH --partition=Long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --output=logs/hourly12_%j.log

set -uo pipefail
mkdir -p logs
cd "$(dirname "$0")" 2>/dev/null || true

# Python: the myenv one (it has cfgrib for the GRIB files); the system python3 does not.
PY=${PY:-$HOME/micromamba/envs/myenv/bin/python}
[ -x "$PY" ] || { echo "no Python at $PY: set PY=/path/to/python"; exit 1; }
"$PY" -c "import cfgrib" 2>/dev/null || echo "[WARN] cfgrib missing in $PY: GRIB files will be skipped"

# Fewer parallel readers than CPUs: the big pressure-level files are limited by the disk,
# not the CPU, and too many readers at once slow each other down.
export SAW_WORKERS=${SAW_WORKERS:-6}

echo "Started $(date) on $(hostname) with ${SAW_WORKERS} readers, $PY"
"$PY" /soge-home/users/jesu4127/southern_african_warming/src/hourly_composites.py "$@"
echo "Finished $(date)"