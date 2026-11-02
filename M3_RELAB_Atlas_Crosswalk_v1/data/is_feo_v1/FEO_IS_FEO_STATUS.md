# FeO and I_s/FeO status

Generated: 2026-10-14T10:52:47.756115+00:00

## Interpretation rules

- Bhatt-derived material: excluded.
- Parent-to-child transfer: forbidden.
- Graph/curve reading: forbidden.
- Qwen output: candidates requiring human review, never confirmed truth.
- I_s: calibrated ferromagnetic-resonance intensity near g=2.1, not a spectral-reflectance ratio.

## FeO reviewed pipeline

- Reviewed parent rows: 46
- Resolved parent rows: 36
- Unresolved or no defensible parent-bulk rows: 10
- Remaining priority cases: 5 (60215, 65315, 67729, 69003, 69004)
- Candidate adjudication rows: 402
- Resolution statuses: {"resolved_with_parent_or_documented_fraction_feo": 36, "reviewed_no_parent_bulk_feo": 10}
- Candidate decisions: {"accepted_parent_or_documented_fraction": 133, "reassigned_to_other_sample": 92, "rejected_duplicate_extraction": 3, "rejected_parser_false_positive": 106, "retained_component_or_subsample_only": 68}

## I_s/FeO recovery

- Previously recorded rows: 22
- RELAB metadata hits: 240
- Atlas candidate pages: 335
- Qwen input chunks: 139
- Qwen completed / pending: 0 / 139
- Qwen errors: 0
- Qwen candidate measurement rows: 0
- Chunks with unresolved statements: 0
- Total unresolved statements: 0
- Validation statuses: {}

Missing inputs remain marked `present: false` in the JSON report; they are not counted as completed work.

## Added independent source extracts (2026-11-02)

These source tables are stored under `data/source_extracts/`. They supplement
the reviewed pipeline above and do not silently alter its earlier counts.

- JSC 19069 lunar-soil data sheets: 169 sample rows.
- JSC 19069 FeO values: 132.
- JSC 19069 I_s/FeO values: 157.
- JSC 19069 rows with both: 126.
- LPI Regolith Breccia Workbook: 45 sample rows and 790 reported analyte rows.
- Breccia rows with FeO: 42.
- Combined source index: 214 distinct Apollo sample IDs; 174 rows with FeO and 157 with I_s/FeO.

The source index preserves separate source rows and performs no averaging,
imputation, graph reading, or parent-to-child transfer.
