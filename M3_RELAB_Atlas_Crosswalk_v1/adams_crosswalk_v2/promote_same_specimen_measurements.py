"""Priority 1: safely promote additional CLS measurements of ALREADY-identified
exact Adams specimens.

Evidence chain (NOT sibling-suffix inference):
  Apollo sample  <- Bhatt 2012 Table B.1 ->  exact full LS-JBA specimen
  exact specimen <- Adams archive header  ->  additional CLS measurement file

A CLS is promoted to `validated_same_specimen_measurement` only if:
  * its full specimen id has exactly ONE validated Apollo/Luna sample (no conflict);
  * the Adams corrected-file header for that CLS explicitly names the SAME full
    specimen id.

Outputs data/adams_id_lists/adams_measurement_crosswalk_expanded_v2.csv with both
evidence classes kept distinct, and prints counts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

IDDIR = Path("data/adams_id_lists")
CORRECTED = Path("data/adams-20220223T225406Z-001/adams/wavelength_corrected_data")


def header_specimen(cls_id: str) -> str | None:
    """Read 'Brown University ... LS-JBA-XXX/CLSNNN' header; return the specimen id."""
    for name in (f"{cls_id.lower()}.txt", f"{cls_id.upper()}.txt", f"{cls_id}.txt"):
        f = CORRECTED / name
        if f.exists():
            first = f.read_text(errors="replace").splitlines()[0]
            m = re.search(r"(LS-JBA-[0-9A-Za-z-]+)\s*/\s*(CLS\d+)", first, re.IGNORECASE)
            if m:
                return m.group(1).upper(), m.group(2).upper()
            return ("", first.strip())
    return None


def main() -> int:
    valid = pd.read_csv(IDDIR / "adams_atlas_crosswalk_validated.csv")
    valid["adams_specimen_id"] = valid["adams_specimen_id"].astype(str).str.strip()
    valid["corrected_measurement_id"] = valid["corrected_measurement_id"].astype(str).str.strip()

    # specimen -> set of validated samples (detect conflicts)
    spec_samples: dict[str, set] = {}
    spec_body: dict[str, str] = {}
    for _, r in valid.iterrows():
        spec_samples.setdefault(r["adams_specimen_id"], set()).add(str(r["apollo_sample_id"]).strip())
        spec_body[r["adams_specimen_id"]] = r["body"]
    conflicts = {s: v for s, v in spec_samples.items() if len(v) > 1}

    cand = pd.read_csv(IDDIR / "adams_atlas_mapping_candidates_v2.csv")
    extra = cand[cand["mapping_status"] == "probable_needs_review"].copy()

    rows = []
    promoted = unresolved = conflict_blocked = 0
    for _, r in extra.iterrows():
        sid = str(r["adams_specimen_id"]).strip()
        cls = str(r["corrected_measurement_id"]).strip()
        samples = spec_samples.get(sid, set())
        if sid in conflicts:
            status, note = "conflicting", f"specimen maps to multiple samples {sorted(conflicts[sid])}"
            conflict_blocked += 1
        else:
            hdr = header_specimen(cls)
            hdr_spec = hdr[0] if hdr else ""
            if hdr and hdr_spec == sid:
                status = "promote"
                note = f"header names {hdr[0]}/{hdr[1]}"
                promoted += 1
            else:
                status = "unresolved"
                note = f"header did not confirm exact specimen (got {hdr!r})"
                unresolved += 1
        sample = next(iter(samples)) if len(samples) == 1 else ""
        rows.append({"adams_specimen_id": sid, "corrected_measurement_id": cls,
                     "apollo_sample_id": sample, "body": spec_body.get(sid, ""),
                     "promotion_status": status, "note": note})
    prom = pd.DataFrame(rows)

    # ---- build expanded crosswalk: class A (literature exact) + class B (same specimen) ----
    classA = valid[["body", "apollo_sample_id", "adams_specimen_id", "corrected_measurement_id"]].copy()
    classA["evidence_class"] = "validated_exact_pair"
    classA["identity_source"] = "Bhatt 2012 doctoral thesis, Appendix B, Table B.1"
    classA["identity_basis"] = "exact full LS-JBA specimen + CLS in literature"
    classA["measurement_source"] = "Bhatt 2012 Table B.1"
    classA["measurement_basis"] = "CLS explicitly listed in source"
    classA["inference"] = "none"

    promok = prom[prom["promotion_status"] == "promote"].copy()
    classB = promok[["body", "apollo_sample_id", "adams_specimen_id", "corrected_measurement_id"]].copy()
    classB["evidence_class"] = "validated_same_specimen_measurement"
    classB["identity_source"] = "Bhatt 2012 Table B.1"
    classB["identity_basis"] = "exact full LS-JBA specimen"
    classB["measurement_source"] = "Adams archive wavelength-corrected header"
    classB["measurement_basis"] = "CLS header explicitly names same full specimen"
    classB["inference"] = "additional spectral measurement of already identified specimen"

    expanded = pd.concat([classA, classB], ignore_index=True)
    expanded.to_csv(IDDIR / "adams_measurement_crosswalk_expanded_v2.csv", index=False)
    prom.to_csv(IDDIR / "adams_same_specimen_promotion_audit_v2.csv", index=False)

    summary = {
        "extra_cls_considered": int(len(extra)),
        "promoted_same_specimen_measurement": int(promoted),
        "remained_unresolved": int(unresolved),
        "conflict_blocked": int(conflict_blocked),
        "conflicts": {k: sorted(v) for k, v in conflicts.items()},
        "total_safe_cls_after_expansion": int(len(expanded)),
        "class_counts": expanded["evidence_class"].value_counts().to_dict(),
        "distinct_exact_specimens": int(expanded["adams_specimen_id"].nunique()),
        "distinct_samples_represented": int(expanded["apollo_sample_id"].astype(str).nunique()),
    }
    (IDDIR / "adams_expanded_crosswalk_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
