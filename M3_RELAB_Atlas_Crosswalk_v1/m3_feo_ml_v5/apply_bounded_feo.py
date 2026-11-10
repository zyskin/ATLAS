#!/usr/bin/env python3
"""Apply a saved bounded FeO model to 74-band M3-basis spectra."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from m3_feo_ml_v4.features import BAND_COLUMNS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,help="final_model.joblib")
    ap.add_argument("--input",required=True,help="CSV containing m3_b01..m3_b74")
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    art=joblib.load(a.model)
    d=pd.read_csv(a.input)
    missing=[c for c in BAND_COLUMNS if c not in d.columns]
    if missing: raise SystemExit("missing band columns: "+", ".join(missing))
    bands=d[BAND_COLUMNS].to_numpy(float)
    wt=art.predict_wt_percent(bands)
    out=d.copy()
    out["feo_fraction"]=wt/100.0
    out["feo_wt_percent"]=wt
    out["physical_fraction_0_1"]=(out["feo_fraction"]>=0)&(out["feo_fraction"]<=1)
    out.to_csv(a.output,index=False)
    print(f"wrote {a.output}; range fraction={out.feo_fraction.min():.6g}..{out.feo_fraction.max():.6g}")
if __name__=="__main__":
    main()
