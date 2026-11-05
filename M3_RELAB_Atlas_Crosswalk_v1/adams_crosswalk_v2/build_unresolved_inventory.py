"""Deliverable #1: unresolved Adams->Apollo/Atlas mapping inventory.

Compares the full Adams specimen/measurement list against the evidence-based
validated crosswalk and reports what remains unmapped, broken down by LS-JBA
prefix and suffix form.  Writes:
  data/adams_id_lists/adams_mapping_unresolved_v2.csv   (one row per unresolved
      specimen ID, with its CLS measurement IDs and parsed prefix/suffix)
  data/adams_id_lists/adams_mapping_inventory_summary_v2.json

Run:
  python -m adams_crosswalk_v2.build_unresolved_inventory
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

IDDIR = Path("data/adams_id_lists")
ALL = IDDIR / "adams_specimen_ids_all.csv"
VALID = IDDIR / "adams_atlas_crosswalk_validated.csv"

# LS-JBA-<num>[-<suffix>]; prefix = LS-JBA-<num>, suffix = remainder (verbatim)
PREFIX_RE = re.compile(r"^(LS-JBA-\d+)(?:-(.+))?$", re.IGNORECASE)


def parse_id(sid: str) -> tuple[str, str]:
    m = PREFIX_RE.match(str(sid).strip())
    if not m:
        return (str(sid), "")
    return (m.group(1).upper(), (m.group(2) or "").upper())


def suffix_form(suffix: str) -> str:
    """Generalise a suffix to a form token, e.g. P1->P#, PX->PX, M->M, ''->none."""
    if not suffix:
        return "none"
    s = re.sub(r"\d+", "#", suffix)
    return s


def explode_measurements(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        ids = str(r["corrected_measurement_ids"]).split(";") if pd.notna(r["corrected_measurement_ids"]) else []
        for cls in ids:
            cls = cls.strip()
            if cls:
                rows.append({"adams_specimen_id": r["adams_specimen_id"], "corrected_measurement_id": cls})
    return pd.DataFrame(rows)


def main() -> int:
    alldf = pd.read_csv(ALL)
    valid = pd.read_csv(VALID)

    alldf["adams_specimen_id"] = alldf["adams_specimen_id"].str.strip()
    specimen_all = set(alldf["adams_specimen_id"])
    pairs_all = explode_measurements(alldf)
    cls_all = set(pairs_all["corrected_measurement_id"])

    mapped_specimens = set(valid["adams_specimen_id"].astype(str).str.strip())
    valid_pairs = set(zip(valid["adams_specimen_id"].astype(str).str.strip(),
                          valid["corrected_measurement_id"].astype(str).str.strip()))
    mapped_cls = set(valid["corrected_measurement_id"].astype(str).str.strip())

    unresolved_specimens = sorted(specimen_all - mapped_specimens)
    # unresolved CLS = CLS not appearing in any validated exact pair
    all_pairs_set = set(zip(pairs_all["adams_specimen_id"], pairs_all["corrected_measurement_id"]))
    unresolved_pairs = sorted(all_pairs_set - valid_pairs)
    unresolved_cls = sorted(cls_all - mapped_cls)

    # Build unresolved specimen table
    out_rows = []
    for _, r in alldf.iterrows():
        sid = r["adams_specimen_id"]
        if sid in mapped_specimens:
            continue
        prefix, suffix = parse_id(sid)
        out_rows.append({
            "adams_specimen_id": sid,
            "ls_jba_prefix": prefix,
            "suffix": suffix,
            "suffix_form": suffix_form(suffix),
            "corrected_measurement_count": r["corrected_measurement_count"],
            "corrected_measurement_ids": r["corrected_measurement_ids"],
            "prefix_has_any_validated": prefix in {parse_id(s)[0] for s in mapped_specimens},
            "mapping_status": "unresolved",
        })
    unresolved_df = pd.DataFrame(out_rows)
    unresolved_df.to_csv(IDDIR / "adams_mapping_unresolved_v2.csv", index=False)

    # Prefix / suffix breakdowns
    allp = alldf["adams_specimen_id"].map(lambda s: parse_id(s)[0])
    alls = alldf["adams_specimen_id"].map(lambda s: suffix_form(parse_id(s)[1]))
    prefix_total = allp.value_counts()
    mapped_prefixes = {parse_id(s)[0] for s in mapped_specimens}
    prefix_rows = []
    for pfx, tot in prefix_total.items():
        n_unres = int((unresolved_df["ls_jba_prefix"] == pfx).sum()) if not unresolved_df.empty else 0
        prefix_rows.append({"ls_jba_prefix": pfx, "n_specimens": int(tot),
                            "n_unresolved": n_unres,
                            "has_validated_member": pfx in mapped_prefixes})
    pd.DataFrame(prefix_rows).sort_values("ls_jba_prefix").to_csv(
        IDDIR / "adams_mapping_prefix_breakdown_v2.csv", index=False)

    summary = {
        "n_unique_adams_specimen_ids": len(specimen_all),
        "n_corrected_measurement_ids_unique": len(cls_all),
        "n_specimen_cls_pairs_total": len(all_pairs_set),
        "n_mapped_specimen_ids_exact": len(mapped_specimens & specimen_all),
        "n_mapped_specimen_cls_pairs_exact": len(valid_pairs & all_pairs_set),
        "n_validated_rows": int(valid.shape[0]),
        "n_unresolved_specimen_ids": len(unresolved_specimens),
        "n_unresolved_specimen_cls_pairs": len(unresolved_pairs),
        "n_unresolved_cls_ids": len(unresolved_cls),
        "n_ls_jba_prefixes_total": int(allp.nunique()),
        "n_ls_jba_prefixes_with_any_validated": len(mapped_prefixes),
        "suffix_form_counts_all": alls.value_counts().to_dict(),
        "body_counts_validated": valid["body"].value_counts().to_dict() if "body" in valid else {},
    }
    (IDDIR / "adams_mapping_inventory_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {IDDIR/'adams_mapping_unresolved_v2.csv'} ({len(unresolved_df)} rows)")
    print(f"wrote {IDDIR/'adams_mapping_prefix_breakdown_v2.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
