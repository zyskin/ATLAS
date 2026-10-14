#!/bin/bash
#SBATCH -J atlas-isfeo-cpu
#SBATCH -A PHY25025
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o isfeo-cpu.%j.out
#SBATCH -e isfeo-cpu.%j.err

set -euo pipefail
module reset
export QWEN_BACKEND=cpu
export QWEN_THREADS=120

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATLAS_PROJECT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export ATLAS_PROJECT
exec "$ATLAS_PROJECT/run_is_feo_qwen_tacc_v1.sh"

