I_s/FeO CANDIDATE RECOVERY, VERSION 1
=====================================

What was already recorded
-------------------------
Earlier FeO adjudication incidentally retained some I_s/FeO values in:

  data/ai_resolution_v1/incidental_al2o3_age_maturity.csv
  data/ai_resolution_v1/missing_case_incidental_al2o3_age_maturity.csv

Those files are useful but incomplete.  The RELAB spectrum and chemistry CSVs
have no dedicated numerical I_s/FeO column.  Some spectrum descriptions contain
only qualitative terms such as Mature, Submature, or Immature.

The new workflow
----------------
1. collect_is_feo_candidates_v1.py

   * inventories the previously recorded values;
   * checks the extracted RELAB spectrum and chemistry rows;
   * checks the original RELAB PDS4 XML metadata when --relab-root is supplied;
   * scans all retained Atlas parsed-PDF JSON.GZ files;
   * chops likely numerical statements/tables into short review chunks;
   * retains source URL, PDF SHA-256, document ID, and PDF page.

2. adjudicate_atlas_is_feo_local_ai_v1.py

   * sends each short chunk to the user's LOCAL llama.cpp/Qwen server;
   * checkpoints after every successful chunk and resumes automatically;
   * reuses a checkpoint only when the chunk-text SHA-256 is unchanged;
   * maps ordered "respectively" lists only when explicit;
   * rejects graph reading and parent/child value transfer;
   * marks every model-produced row needs_human_review.

Bhatt-derived material is excluded by default in both the deterministic rules
and the local-model prompt.

Run the deterministic scan locally
----------------------------------
  ATLAS=/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1
  RELAB=/media/maxim/Elements/grants/2024/INSRU/pgdal/cartApollo2026

  python "$ATLAS/collect_is_feo_candidates_v1.py" \
    --project-root "$ATLAS" \
    --relab-root "$RELAB"

The default output directory is:

  $ATLAS/data/is_feo_v1

If it already exists and the run is intentionally being replaced:

  python "$ATLAS/collect_is_feo_candidates_v1.py" \
    --project-root "$ATLAS" \
    --relab-root "$RELAB" \
    --overwrite

Important outputs
-----------------
  existing_is_feo_records.csv
      Previously recorded I_s/FeO rows only.  Their original scope and review
      flags are preserved.

  relab_is_feo_hits.csv
      RELAB metadata hits.  match_kind distinguishes explicit_is_feo from
      qualitative_maturity.  A qualitative hit is not a numerical value.

  atlas_is_feo_candidates.csv
      All Atlas pages containing an I_s/FeO spelling.  candidate_kind separates
      likely numerical prose, table/layout candidates, graph/ambiguous context,
      citations, and text-only occurrences.

  atlas_is_feo_review_chunks.jsonl
      Only likely numerical prose/table chunks.  Citations and graph-only axes
      are not sent to the model.

  is_feo_scan_manifest.json
      Input locations, rules, counts, and SHA-256 hashes of every output.

Run local Qwen adjudication
---------------------------
Start the same local llama.cpp server used for the earlier FeO extraction.  It
must listen only on 127.0.0.1.  Then run:

  python "$ATLAS/adjudicate_atlas_is_feo_local_ai_v1.py" \
    --project-root "$ATLAS" \
    --no-stream

For a short test first:

  python "$ATLAS/adjudicate_atlas_is_feo_local_ai_v1.py" \
    --project-root "$ATLAS" \
    --limit 5 \
    --no-stream

Rerunning without --restart resumes from the checkpoint.  Use --restart only
when intentionally discarding the entire prior local-AI checkpoint.

Local-AI outputs
----------------
  data/is_feo_v1/local_ai/atlas_is_feo_ai_checkpoint.jsonl
  data/is_feo_v1/local_ai/atlas_is_feo_ai_measurements.csv
  data/is_feo_v1/local_ai/atlas_is_feo_ai_errors.jsonl
  data/is_feo_v1/local_ai/atlas_is_feo_ai_manifest.json

The measurements CSV is not a confirmed master table.  Every row remains
needs_human_review.  Confirmation must compare the exact page evidence and
scope; in particular, never assign a parent-soil value to a RELAB child or
grain-size fraction unless that exact relationship is explicitly documented.

Tests
-----
  cd "$ATLAS"
  python -m unittest -v test_is_feo_workflow_v1.py
