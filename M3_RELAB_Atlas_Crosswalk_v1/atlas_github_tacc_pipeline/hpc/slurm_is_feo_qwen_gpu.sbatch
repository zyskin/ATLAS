#!/bin/bash
#SBATCH -J atlas-isfeo-gpu
#SBATCH -A PHY25025
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o isfeo-gpu.%j.out
#SBATCH -e isfeo-gpu.%j.err

set -euo pipefail
module reset
module load gcc/11.2.0
module load cuda/12.2
export QWEN_BACKEND=gpu
export QWEN_THREADS=12

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATLAS_PROJECT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export ATLAS_PROJECT
exec "$ATLAS_PROJECT/run_is_feo_qwen_tacc_v1.sh"
