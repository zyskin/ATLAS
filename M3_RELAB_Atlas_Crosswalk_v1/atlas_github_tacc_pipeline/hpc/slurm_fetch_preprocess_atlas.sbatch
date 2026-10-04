#!/bin/bash
#SBATCH -J atlas_pdf
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH -t 00:15:00

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: sbatch ... slurm_fetch_preprocess_atlas.sbatch URL_TASKS_TSV OUTPUT_DIR PIPELINE_DIR VENV_DIR" >&2
  exit 2
fi

TASKS_TSV=$1
OUTPUT_DIR=$2
PIPELINE_DIR=$3
VENV_DIR=$4

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_TSV")
if [ -z "$line" ]; then
  echo "No task at line ${SLURM_ARRAY_TASK_ID} in $TASKS_TSV" >&2
  exit 2
fi

IFS=$'\t' read -r document_id sample_ids source_url <<< "$line"

source "$VENV_DIR/bin/activate"
python "$PIPELINE_DIR/code/fetch_preprocess_atlas_pdf.py" \
  --url "$source_url" \
  --document-id "$document_id" \
  --sample-ids "$sample_ids" \
  --output-dir "$OUTPUT_DIR"
