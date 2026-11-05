"""Priority 2: investigate the validated Apollo samples NOT joined to the modern
Atlas-parent / ML grouping, using LOCAL repository evidence only.

For each such sample id, search: the RELAB->Atlas crosswalk (as sample, as
apollo_parent_sample, and inside relab_specimen_name), the ML ground-truth
parents, and parsed Lunar Sample Compendium filenames/text. Record explicit
Atlas ids or parent-child relationships. Never derive a parent by truncating a
sample number.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

IDDIR = Path("data/adams_id_lists")
CROSSWALK = "crosswalk_output_v2_2026-10-03/relab_atlas_crosswalk.csv"
ML_GT = "data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv"
PARSED = Path("data/parsed")


def main() -> int:
    join = pd.read_csv(IDDIR / "adams_validated_atlas_parent_join_v2.csv")
    missing = join[join["join_resolution"] == "apollo_sample_not_in_atlas_or_ml"]
    samples = sorted(set(missing["apollo_sample_id"].astype(str).str.strip()))

    cw = pd.read_csv(CROSSWALK)
    cw_s = cw.astype(str)
    ml = pd.read_csv(ML_GT, usecols=["i_atlas_sample"])
    ml_parents = set(ml["i_atlas_sample"].astype(str).str.strip())
    compendium_ids = {re.match(r"(\d+)", p.name).group(1)
                      for p in PARSED.glob("status/*compendium*") if re.match(r"\d+", p.name)}

    rows = []
    for s in samples:
        status, evidence, excerpt = "atlas_record_not_found", "", ""
        # exact Atlas id present as atlas_sample_id?
        hit_atlas = cw[cw["atlas_sample_id"].astype(str).str.strip() == s]
        hit_parent = cw[cw["apollo_parent_sample"].astype(str).str.strip() == s]
        # sample appears inside a relab_specimen_name (child of a parent)?
        name_hits = cw_s[cw_s["relab_specimen_name"].str.contains(rf"\b{re.escape(s)}\b", regex=True, na=False)]
        if not hit_atlas.empty:
            status = "exact_atlas_match"
            evidence = "relab_atlas_crosswalk.atlas_sample_id"
            excerpt = str(hit_atlas.iloc[0]["atlas_sample_id"])
        elif not hit_parent.empty:
            status = "explicit_parent_relationship"
            evidence = "relab_atlas_crosswalk.apollo_parent_sample"
            excerpt = f"atlas_sample_id={hit_parent.iloc[0]['atlas_sample_id']}"
        elif not name_hits.empty:
            r0 = name_hits.iloc[0]
            status = "candidate_needs_review"
            evidence = "relab_specimen_name mentions sample"
            excerpt = f"name={r0['relab_specimen_name']} parent={r0['apollo_parent_sample']} atlas={r0['atlas_sample_id']}"
        elif s in ml_parents:
            status = "exact_atlas_match"
            evidence = "ml_ground_truth i_atlas_sample"
            excerpt = s
        elif s in compendium_ids:
            status = "candidate_needs_review"
            evidence = "parsed compendium entry exists"
            excerpt = f"compendium id {s} parsed locally"
        rows.append({"apollo_sample_id": s, "status": status,
                     "evidence_source": evidence, "evidence_excerpt": excerpt})

    out = pd.DataFrame(rows)
    out.to_csv(IDDIR / "adams_missing_atlas_parent_candidates_v2.csv", index=False)
    summary = {
        "n_missing_investigated": len(samples),
        "status_counts": out["status"].value_counts().to_dict(),
        "recovered_joins": int(out["status"].isin(
            ["exact_atlas_match", "explicit_parent_relationship"]).sum()),
    }
    (IDDIR / "adams_missing_atlas_parent_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
