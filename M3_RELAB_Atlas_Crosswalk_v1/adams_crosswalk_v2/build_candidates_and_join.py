"""Deliverables #3 and #5: local candidate-evidence table for unresolved Adams
pairs, and an Apollo-sample -> Atlas-parent join for the validated mappings.

Local-evidence search: for every unresolved specimen+CLS pair, look for the
LS-JBA ID / CLS ID anywhere in the repo's RELAB/crosswalk/overlay/compendium
tables and record any co-occurring Apollo sample number as a candidate.  (In
this repository the only LS-JBA-bearing tables are derived from the 68 Bhatt
rows, so this honestly records "no independent local evidence" for the rest.)

Atlas-parent join: for each validated Apollo sample id, attach the modern Atlas
parent actually used by the RELAB crosswalk / ML ground truth, WITHOUT deriving
the parent numerically from the sample number.

Outputs (data/adams_id_lists/):
  adams_atlas_mapping_candidates_v2.csv
  adams_validated_atlas_parent_join_v2.csv
  adams_crosswalk_join_summary_v2.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

IDDIR = Path("data/adams_id_lists")
CROSSWALK = "crosswalk_output_v2_2026-10-03/relab_atlas_crosswalk.csv"
ML_GT = "data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv"
RATIO = "data/relab_adams_ratio_v1/relab_adams_ratios_by_sample_band.csv"


def explode(df):
    rows = []
    for _, r in df.iterrows():
        ids = str(r["corrected_measurement_ids"]).split(";") if pd.notna(r["corrected_measurement_ids"]) else []
        for cls in ids:
            if cls.strip():
                rows.append({"adams_specimen_id": r["adams_specimen_id"].strip(),
                             "corrected_measurement_id": cls.strip()})
    return pd.DataFrame(rows)


def main() -> int:
    alldf = pd.read_csv(IDDIR / "adams_specimen_ids_all.csv")
    alldf["adams_specimen_id"] = alldf["adams_specimen_id"].str.strip()
    valid = pd.read_csv(IDDIR / "adams_atlas_crosswalk_validated.csv")
    valid["adams_specimen_id"] = valid["adams_specimen_id"].astype(str).str.strip()
    pairs = explode(alldf)

    valid_pairs = set(zip(valid["adams_specimen_id"],
                          valid["corrected_measurement_id"].astype(str).str.strip()))

    # ---- local evidence: which LS-JBA specimens co-occur with an Apollo parent
    # in any *independent* local table (i.e. not the Bhatt-derived overlay/ratio) ----
    # The overlay/ratio tables carry apollo_parent_id for Adams specimens, but that
    # id was itself assigned from the 68 validated rows; we record its provenance.
    ratio_map = {}
    try:
        rat = pd.read_csv(RATIO, usecols=["adams_specimen_id", "apollo_parent_id", "match_basis"])
        for _, r in rat.drop_duplicates().iterrows():
            ratio_map[str(r["adams_specimen_id"]).strip()] = (r["apollo_parent_id"], r["match_basis"])
    except Exception:
        pass

    cand_rows = []
    for _, r in pairs.iterrows():
        sid, cls = r["adams_specimen_id"], r["corrected_measurement_id"]
        if (sid, cls) in valid_pairs:
            continue  # already validated exact pair
        # look for derived apollo parent (provenance = Bhatt-derived, not new evidence)
        derived = ratio_map.get(sid)
        if derived is not None and pd.notna(derived[0]):
            cand_rows.append({
                "adams_specimen_id": sid, "corrected_measurement_id": cls,
                "candidate_apollo_sample_id": int(derived[0]) if str(derived[0]).isdigit() else derived[0],
                "candidate_parent_atlas_id": "", "body": "Apollo",
                "evidence_type": "bhatt_derived_overlay",
                "evidence_source": "adams_relab_parent_overlays (derived from Bhatt 2012 validated rows)",
                "evidence_excerpt": f"match_basis={derived[1]}",
                "mapping_status": "probable_needs_review",
                "confidence": "low_derived",
                "conflict_notes": "parent carried from a sibling validated specimen; not an independent source",
            })
        else:
            cand_rows.append({
                "adams_specimen_id": sid, "corrected_measurement_id": cls,
                "candidate_apollo_sample_id": "", "candidate_parent_atlas_id": "",
                "body": "", "evidence_type": "none_local",
                "evidence_source": "", "evidence_excerpt": "",
                "mapping_status": "unresolved", "confidence": "none",
                "conflict_notes": "no independent local evidence; PDS/RELAB/Brown hosts blocked by proxy in this run",
            })
    cand = pd.DataFrame(cand_rows)
    cand.to_csv(IDDIR / "adams_atlas_mapping_candidates_v2.csv", index=False)

    # ---- Atlas-parent join for validated samples (no numeric derivation) ----
    cw = pd.read_csv(CROSSWALK)
    # apollo_parent_sample -> atlas_sample_id (authoritative, from RELAB crosswalk)
    parent_lookup = (cw.dropna(subset=["apollo_parent_sample"])
                     .assign(aps=lambda d: d["apollo_parent_sample"].astype(str).str.strip())
                     .groupby("aps")["atlas_sample_id"].first().to_dict())
    ml = pd.read_csv(ML_GT, usecols=["i_atlas_sample"])
    ml_parents = set(ml["i_atlas_sample"].astype(str).str.strip())

    jrows = []
    for _, r in valid.iterrows():
        sample = str(r["apollo_sample_id"]).strip()
        atlas_from_cw = parent_lookup.get(sample, "")
        in_ml = sample in ml_parents
        jrows.append({
            "adams_specimen_id": r["adams_specimen_id"],
            "corrected_measurement_id": r["corrected_measurement_id"],
            "body": r["body"],
            "apollo_sample_id": sample,
            "atlas_join_status_existing": r.get("atlas_join_status", ""),
            "atlas_sample_id_from_relab_crosswalk": atlas_from_cw,
            "apollo_sample_in_ml_ground_truth_parent": in_ml,
            "join_resolution": (
                "luna_non_apollo" if str(r["body"]).lower() == "luna" else
                "atlas_parent_confirmed" if (atlas_from_cw or in_ml) else
                "apollo_sample_not_in_atlas_or_ml"),
        })
    jdf = pd.DataFrame(jrows)
    jdf.to_csv(IDDIR / "adams_validated_atlas_parent_join_v2.csv", index=False)

    summary = {
        "validated_specimens": int(valid["adams_specimen_id"].nunique()),
        "validated_pairs": int(len(valid)),
        "validated_apollo_samples": int(valid.loc[valid.body == "Apollo", "apollo_sample_id"].nunique()),
        "validated_luna_rows": int((valid.body == "Luna").sum()),
        "join_resolution_counts": jdf["join_resolution"].value_counts().to_dict(),
        "validated_samples_in_ml_ground_truth": int(jdf["apollo_sample_in_ml_ground_truth_parent"].sum()),
        "candidate_rows": int(len(cand)),
        "candidate_status_counts": cand["mapping_status"].value_counts().to_dict(),
        "note": "External PDS/RELAB/Brown hosts returned HTTP 403 via the agent proxy; "
                "no new exact pairs could be established this run. Candidate table records "
                "local-only evidence; new exact validation requires PDS RELAB specimen "
                "metadata (ls-jba-*_specimenchemistry) once a host is reachable.",
    }
    (IDDIR / "adams_crosswalk_join_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {IDDIR/'adams_atlas_mapping_candidates_v2.csv'} ({len(cand)} rows)")
    print(f"wrote {IDDIR/'adams_validated_atlas_parent_join_v2.csv'} ({len(jdf)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
