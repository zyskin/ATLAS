#!/usr/bin/env bash
set -euo pipefail

# Deterministic laptop pass.  It does not call Qwen and does not confirm any
# candidate value.  Pass the RELAB root as argument 1, or set RELAB_ROOT.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ATLAS_PROJECT=${ATLAS_PROJECT:-$SCRIPT_DIR}
RELAB_ROOT=${RELAB_ROOT:-${1:-$(dirname "$ATLAS_PROJECT")/cartApollo2026}}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ ! -d "$RELAB_ROOT" ]]; then
  echo "ERROR: RELAB root not found: $RELAB_ROOT" >&2
  exit 2
fi

overwrite=()
if [[ ${OVERWRITE:-0} == 1 ]]; then
  overwrite=(--overwrite)
fi

cd "$ATLAS_PROJECT"
"$PYTHON_BIN" -m unittest -v test_is_feo_workflow_v1.py
"$PYTHON_BIN" collect_is_feo_candidates_v1.py \
  --project-root "$ATLAS_PROJECT" \
  --relab-root "$RELAB_ROOT" \
  "${overwrite[@]}"
"$PYTHON_BIN" summarize_feo_is_feo_status_v1.py \
  --project-root "$ATLAS_PROJECT"

echo
echo "Review before committing:"
git status --short -- data/is_feo_v1

