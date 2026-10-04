LUNAR SAMPLE ATLAS: STAGE 2
===========================

Purpose
-------
Turn the 706 Stage-1 JSON.GZ files into small, traceable tables.  This stage
does not call an AI model and does not download the PDFs again.

It reads
--------
  data/parsed/json_gz/*.json.gz
  atlas_samples.csv                 (optional but recommended)

It writes
---------
  atlas_documents.csv
      One row per parsed PDF and its source/hash/page counts.

  atlas_feo_measurements.csv
      One row per numeric FeO table cell.  It records the Atlas sample,
      value, printed reference, method code/method, PDF page, exact FeO row,
      source URL/hash, coordinates, confidence, and review flag.

  atlas_review_queue.csv
      Ambiguous FeO/header assignments that need checking.

  atlas_narrative_pages.jsonl
      Non-reference pages relevant to material, mineralogy, age, or treatment.
      This is the input for a later local-Qwen pass; it is not yet interpreted.

  atlas_samples_enriched.csv
      The Atlas HTML-index table plus parsed-document and bulk-FeO counts/range.

  stage2_manifest.json
      Counts, errors, program version, notes, and SHA-256 hashes of every output.

Important safeguards
--------------------
* Blank/missing FeO is never changed to zero or -99.
* The parser uses the PDF word coordinates, not flattened table text.
* A complex or unresolved table header is marked needs_review=yes.
* Mineral-phase FeO is distinguished from whole-sample/bulk FeO when the
  table title makes that distinction possible.
* The original PDF URL, PDF SHA-256, page, evidence row, and x/y position are
  retained for audit.

Run locally
-----------
  ATLAS=/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1

  python "$ATLAS/atlas_github_tacc_pipeline/code/extract_atlas_stage2.py" \
    --parsed-dir "$ATLAS/data/parsed" \
    --atlas-csv "$ATLAS/atlas_samples.csv" \
    --output-dir "$ATLAS/data/structured_v1"

If that output directory already has a manifest and you intentionally want to
replace the Stage-2 files, add --overwrite.

Run tests
---------
  cd "$ATLAS/atlas_github_tacc_pipeline/code"
  python -m unittest -v test_extract_atlas_stage2.py

Quick checks after a full run
-----------------------------
  cat "$ATLAS/data/structured_v1/stage2_manifest.json"

  python - "$ATLAS/data/structured_v1/atlas_feo_measurements.csv" <<'PY'
  import csv, sys
  rows = list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8')))
  print('FeO rows:', len(rows))
  print('Need review:', sum(r['needs_review'] == 'yes' for r in rows))
  print('Bulk/whole-sample:', sum(r['sample_scope'] == 'bulk_or_whole_sample' for r in rows))
  print('Mineral phase:', sum(r['sample_scope'] == 'mineral_phase' for r in rows))
  PY

After this run
--------------
Inspect atlas_review_queue.csv first.  Then a separate Qwen program can process
atlas_narrative_pages.jsonl in short page-sized requests to extract mineralogy,
material type, ages, and treatment.  Those AI-derived fields should keep the
same source/page/evidence trace and should remain reviewable.
