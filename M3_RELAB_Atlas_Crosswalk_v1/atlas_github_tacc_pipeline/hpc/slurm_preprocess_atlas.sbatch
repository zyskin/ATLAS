#!/bin/bash
#SBATCH -J atlas_pdf
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH -t 00:15:00
#SBATCH -o slurm-atlas-%A_%a.out
#SBATCH -e slurm-atlas-%A_%a.err

set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: sbatch ... slurm_preprocess_atlas.sbatch TASKS_TSV OUTPUT_DIR REPO_DIR VENV_DIR" >&2
  exit 2
fi

TASKS_TSV=$1
OUTPUT_DIR=$2
REPO_DIR=$3
VENV_DIR=$4

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_TSV")
if [ -z "$line" ]; then
  echo "No task at line ${SLURM_ARRAY_TASK_ID} in $TASKS_TSV" >&2
  exit 2
fi

IFS=$'\t' read -r document_id sample_ids pdf_path source_url <<< "$line"

source "$VENV_DIR/bin/activate"
python "$REPO_DIR/code/preprocess_atlas_pdf.py" \
  --pdf "$pdf_path" \
  --document-id "$document_id" \
  --sample-ids "$sample_ids" \
  --source-url "$source_url" \
  --output-dir "$OUTPUT_DIR"
