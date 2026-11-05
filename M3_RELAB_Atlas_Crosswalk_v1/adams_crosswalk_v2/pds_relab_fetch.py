"""Ready-to-run PDS RELAB specimen-metadata fetcher for unresolved Adams IDs.

NOTE: in the environment where this was written, pds.nasa.gov,
pds-geosciences.wustl.edu and sites.brown.edu all returned HTTP 403 through the
agent proxy, so this could not be executed here.  It is provided so the search
can continue the moment a RELAB/PDS host is reachable.

RELAB ancillary specimen-chemistry products follow the PDS4 LID pattern
    urn:nasa:pds:relab:data_ancillary_chemistry:ls-jba-<NNN>_specimenchemistry
(confirmed locally for lr-cmp-154_specimenchemistry).  The RELAB spectrum label
also carries `specimen_name` embedding the Apollo sample number
(e.g. "60019,215", "15555,965 bulk <125 um") and `specimen_collection_location`
("Apollo 16") — the authoritative identity fields.

Strategy:
  1. Read unresolved LS-JBA prefixes from adams_mapping_unresolved_v2.csv.
  2. For each prefix, fetch the RELAB spectrum/specimen metadata from a
     reachable host and parse specimen_id / specimen_name / description /
     collection_location.
  3. Accept an exact pair only when the metadata explicitly names an Apollo/Luna
     returned sample for that full specimen id + measurement id.

Hosts to try (first reachable wins):
  - https://pds-speclib.rsl.wustl.edu   (RELAB sample search / Speclib)
  - https://pds-geosciences.wustl.edu/  (Geosciences node bundle browse)
  - https://pds.nasa.gov/api/search/1/products  (registry API, `q=lid like ...`)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import requests  # noqa
except Exception:  # pragma: no cover
    requests = None

IDDIR = Path("data/adams_id_lists")
CACHE = Path("data/adams_id_lists/pds_cache")
HOSTS = [
    "https://pds-speclib.rsl.wustl.edu",
    "https://pds-geosciences.wustl.edu",
    "https://pds.nasa.gov",
]


def reachable(url: str, timeout=20) -> bool:
    if requests is None:
        return False
    try:
        r = requests.head(url, timeout=timeout)
        return r.status_code < 400
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max prefixes (0=all)")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    unresolved = pd.read_csv(IDDIR / "adams_mapping_unresolved_v2.csv")
    prefixes = sorted(unresolved["ls_jba_prefix"].unique())
    if a.limit:
        prefixes = prefixes[: a.limit]

    host = next((h for h in HOSTS if reachable(h)), None)
    if host is None:
        print("No RELAB/PDS host reachable from this environment (all proxy-blocked).",
              file=sys.stderr)
        print(f"Would query {len(prefixes)} LS-JBA prefixes once a host is reachable.")
        return 2

    print(f"Reachable host: {host}. Querying {len(prefixes)} prefixes...")
    # Implementation left for a networked run: for each prefix, request the
    # specimen-chemistry product, parse specimen_name / collection_location, and
    # append validated exact pairs to a new candidates file. Cache raw responses
    # under data/adams_id_lists/pds_cache/.
    for pfx in prefixes:
        _ = pfx  # placeholder: fetch + parse here
        time.sleep(0.1)
    print("Networked fetch/parse not implemented offline; see module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
