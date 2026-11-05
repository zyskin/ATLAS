# Adams → Apollo/Atlas crosswalk v2 — findings

Branch `claude/adams-atlas-crosswalk-v2`. Identity-mapping only; no chemistry
values adopted, ML training data untouched. The rule from the prior review is
preserved: **accept only `full LS-JBA specimen ID + CLS measurement ID → Apollo/
Luna sample` with an explicit named source**; never infer from an LS-JBA range or
shared prefix, and never propagate a verified suffix to sibling suffixes.

## What was done
1. Built the unresolved inventory (`build_unresolved_inventory.py`).
2. Searched local evidence and built a candidate table (`build_candidates_and_join.py`).
3. Joined the validated Apollo samples to the modern Atlas-parent / ML grouping.
4. Attempted NASA PDS / RELAB / Brown metadata retrieval — **all hosts returned
   HTTP 403 through this environment's agent proxy**, so no new external evidence
   could be gathered this run. A ready-to-run fetcher (`pds_relab_fetch.py`) is
   left for a networked session.

## Answers to the high-value questions

- **Adams specimen IDs currently mapped:** 66 distinct specimens (68 exact
  specimen+CLS pairs; 60 distinct Apollo samples + 4 Luna), all from Bhatt 2012
  Appendix B / Table B.1.
- **Remaining unmapped:** 427 specimen IDs; 648 of 716 specimen+CLS pairs; 648
  CLS measurement IDs.
- **Additional exact mappings established this run:** **0 new `validated_exact_pair`.**
  External authoritative sources were unreachable (proxy 403) and local tables
  contain no independent LS-JBA→Apollo evidence beyond the 68 Bhatt rows. No weak
  web snippet met the evidence bar, so nothing was promoted to ground truth.
- **Sources that established mappings:** only Bhatt 2012 (pre-existing). The
  candidate table flags **27 non-validated CLS measurements that belong to
  already-validated specimen IDs** (same physical specimen, different corrected
  measurement file) as `probable_needs_review` — not promoted.
- **Did NASA PDS RELAB metadata materially help?** Not in this run — `pds.nasa.gov`,
  `pds-geosciences.wustl.edu` and `sites.brown.edu` were all proxy-blocked (403).
  However, the **local RELAB spectrum index confirms the mechanism**: RELAB
  specimen metadata carries `specimen_name` embedding the Apollo sample number
  (e.g. "60019,215", "15555,965 bulk <125 um") and `specimen_collection_location`
  ("Apollo 16"). Fetching the `ls-jba-*_specimenchemistry` products once a host is
  reachable is the clear path to new exact pairs.
- **LS-JBA prefixes with several separately-identifiable suffixes:** none newly
  separated this run (0 new evidence). The inventory shows the suffix structure to
  target: of 493 specimens, suffix forms are none (151), P# (90), N# (57), M# (49),
  P (41), M (37), N (37), with 225 distinct prefixes, 64 carrying ≥1 validated member.
- **Adams spectra safely assignable to an Apollo/Atlas parent:** **36** of the 64
  validated Apollo samples join to the modern Atlas-parent grouping used in the
  RELAB crosswalk / ML ground truth (`atlas_parent_confirmed`, verified by lookup,
  not numeric derivation). 28 validated Apollo samples are absent from the Atlas
  index / ML ground truth; 4 are Luna (non-Apollo). The join preserves sample ID
  and Atlas parent separately.
- **Groups to investigate next:** the high-multiplicity prefixes already carrying
  a validated member (fetch the remaining suffixes' specimen-chemistry to confirm
  per-suffix identity), the 28 validated Apollo samples not yet in the Atlas index
  (confirm whether an Atlas parent exists under a different id), and the Luna-24
  range (LS-JBA-140..145) which the prior review flagged as wrong-body.

## Update — same-specimen measurement promotion + Atlas-parent recovery

**Priority 1 (expanded measurement crosswalk):** all **27/27** extra CLS
measurements were promoted to a new, distinct class `validated_same_specimen_measurement`.
Each was promoted only because (a) its full LS-JBA specimen has exactly one
validated Apollo sample (no conflict) and (b) the Adams archive corrected-file
header explicitly names that same full specimen id. This is a same-specimen
measurement chain, not sibling-suffix inference. Evidence classes are kept
separate in `adams_measurement_crosswalk_expanded_v2.csv`:
- `validated_exact_pair`: 68 (Bhatt 2012 Table B.1, CLS explicitly listed)
- `validated_same_specimen_measurement`: 27 (Bhatt identity + Adams header)

**Totals after expansion:** **95** Adams CLS spectra safely assignable to
Apollo/Luna samples, across **66** distinct exact LS-JBA specimens and **64**
distinct samples (60 Apollo + 4 Luna). **0 conflicts** (no specimen maps to >1
sample).

**Priority 2 (missing Atlas parents):** of the 26 distinct validated Apollo
samples not previously joined, **2 (14260, 65501) were recovered** as explicit
Atlas-parent matches — the earlier 36/64 join undercounted because
`apollo_parent_sample` was compared as a float string ("14260.0") vs int
("14260"); the true local Atlas/ML join is **38/64**. The remaining 24 have no
explicit local Atlas record (17 only have a parsed Lunar Sample Compendium entry
→ `candidate_needs_review`; 7 `atlas_record_not_found`). No parent was derived by
truncating digits. See `adams_missing_atlas_parent_candidates_v2.csv`.

## Files (this branch)
- `adams_crosswalk_v2/build_unresolved_inventory.py`
- `adams_crosswalk_v2/build_candidates_and_join.py`
- `adams_crosswalk_v2/pds_relab_fetch.py` (ready-to-run, host-blocked here)
- `data/adams_id_lists/adams_mapping_unresolved_v2.csv`
- `data/adams_id_lists/adams_mapping_prefix_breakdown_v2.csv`
- `data/adams_id_lists/adams_mapping_inventory_summary_v2.json`
- `data/adams_id_lists/adams_atlas_mapping_candidates_v2.csv`
- `data/adams_id_lists/adams_validated_atlas_parent_join_v2.csv`
- `data/adams_id_lists/adams_crosswalk_join_summary_v2.json`

## Recommended next step
Run `python -m adams_crosswalk_v2.pds_relab_fetch` from a session whose proxy
allows `pds-speclib.rsl.wustl.edu` / `pds-geosciences.wustl.edu`, parse
`specimen_name` + `specimen_collection_location` from each `ls-jba-*` product,
and promote only exact specimen+CLS pairs with an explicit Apollo/Luna sample to
`validated_exact_pair`. Do not import Bhatt chemistry as ground truth.
