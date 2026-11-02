# Additional source chemistry extracts

These files add two independently traceable sources to the existing RELAB and
Atlas workflow.  Existing RELAB spectra, Atlas PDF extraction, AI review, and
machine-learning outputs are not overwritten.

## JSC 19069: Handbook of Lunar Soils

Directory: `jsc19069/`

- `lunar_soils_feo_maturity.csv` is the simple three-column table:
  `sample_id`, `FeO_wt_percent`, and `Is_FeO`.
- `lunar_soils_feo_maturity_audit.csv` records viewer-page numbers, OCR evidence,
  missing-value status, and corrections.
- The extraction program is `../../tools/extract_lunar_soils_feo_maturity.py`.

The handbook contains 169 soil data sheets.  The extraction found 132 FeO
values, 157 Is/FeO values, and 126 samples with both.  Empty source cells remain
empty.  The printed heading `SOIL: 21003` on viewer page 79 is corrected to the
Apollo 12 sample ID `12003` and recorded in the audit table.

The source PDF is JSC 19069, *Handbook of Lunar Soils*.  It is not committed to
the repository.

## LPI Regolith Breccia Workbook

Directory: `regolith_breccia/`

- `regolith_breccia_composition.csv` is one wide row per sample.
- `regolith_breccia_reported_values.csv` is a long table preserving analyte,
  value, unit, qualifier, footnote marker, and raw reported text.
- `regolith_breccia_sources_notes.csv` records pages, table references,
  footnotes, source URL, and extraction status.

The extract covers 45 samples from PDF viewer pages 144-167 (printed pages
139-162).  It contains 790 reported analyte rows.  Qualifiers such as
`<100 micrometre`, `<100 mesh fraction`, and total-nitrogen notes are retained.
The workbook records that every source page was visually checked.

Source: https://www.lpi.usra.edu/lunar/samples/RegolithBrecciaWorkbook.pdf

## Combined index

`source_chemistry_index.csv` provides a compact sample-keyed entry point for
both sources.  Different source documents remain separate rows.  It does not
average conflicting measurements, transfer parent values to children, or fill
missing data.

Rebuild it with:

```bash
python tools/build_source_chemistry_index.py \
  --jsc-audit data/source_extracts/jsc19069/lunar_soils_feo_maturity_audit.csv \
  --breccia-composition data/source_extracts/regolith_breccia/regolith_breccia_composition.csv \
  --output data/source_extracts/source_chemistry_index.csv \
  --manifest data/source_extracts/source_chemistry_index_manifest.json
```
