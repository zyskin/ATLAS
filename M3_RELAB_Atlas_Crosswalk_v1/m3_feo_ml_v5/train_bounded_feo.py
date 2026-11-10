#!/usr/bin/env python3
"""Bounded FeO regression comparison on Apollo/RELAB laboratory spectra.

Targets are stored in wt.%.  A physical Fe mass fraction is f=y/100.
Bounded target links guarantee 0<f<1 for every future spectrum without clipping:

  logit_fraction:
      z = log(f/(1-f)),  f_hat = sigmoid(z_hat)

  tanh_softplus_fraction:
      theta = atanh(f) > 0
      z = softplus^{-1}(theta)
      f_hat = tanh(softplus(z_hat))

The second form implements the user's suggested f=tanh(theta), theta>0 while
allowing an unconstrained regressor to predict z in R.

All CV is grouped by Apollo parent (i_atlas_sample).  Final fitted artifacts
store the feature family, estimator and target link and can predict either
fraction or wt.% on any 74-band M3-basis spectrum.
"""
from __future__ import annotations

import argparse, json, math, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, ParameterGrid

from m3_feo_ml_v4.features import build_features, BAND_COLUMNS
from m3_feo_ml_v4.models import build_model as build_v4_model, SpectralAngleRegressor

from m3_feo_ml_v5.deployment import DeployableBoundedModel, forward_y, inverse_z

GROUP = "i_atlas_sample"
SCOPE = "feo_target_scope"
EXACT_SCOPES = {"exact_relab_association_only", "mixed_exact_and_parent"}

def parent_means(y,p,g):
    d=pd.DataFrame({"y":y,"p":p,"g":g}).groupby("g",sort=True).mean(numeric_only=True)
    return d["y"].to_numpy(),d["p"].to_numpy()

def metrics(y,p):
    e=np.asarray(p)-np.asarray(y)
    ss=float(np.sum((np.asarray(y)-np.mean(y))**2))
    return dict(mae=float(np.mean(np.abs(e))),
                rmse=float(np.sqrt(np.mean(e*e))),
                bias=float(np.mean(e)),
                r2=float(1-np.sum(e*e)/ss) if ss>0 else math.nan)

def v3_defs(seed, max_comp):
    comps=[v for v in [2,4,8,16] if v<=max_comp] or [1]
    pcr=[v for v in [5,10,20] if v<=max_comp] or [1]
    gpca=[v for v in [5,10,20] if v<=max_comp] or [1]
    fixed=(ConstantKernel(1.0,"fixed")*RBF(3.0,"fixed")+WhiteKernel(0.1,"fixed"))
    return {
      "v3_ridge":(Pipeline([("scale",StandardScaler()),("model",Ridge())]),{"model__alpha":[0.1,1,10]},"raw"),
      "v3_elastic_net":(Pipeline([("scale",StandardScaler()),("model",ElasticNet(max_iter=100000,random_state=seed))]),{"model__alpha":[0.01,0.1,1.0],"model__l1_ratio":[0.1,0.5,0.9]},"raw"),
      "v3_pcr_ridge":(Pipeline([("scale",StandardScaler()),("pca",PCA(random_state=seed)),("model",Ridge())]),{"pca__n_components":pcr,"model__alpha":[0.1,1,10]},"raw"),
      "v3_pls":(Pipeline([("scale",StandardScaler()),("model",PLSRegression(scale=False,max_iter=2000))]),{"model__n_components":comps},"raw"),
      "v3_svr_rbf":(Pipeline([("scale",StandardScaler()),("model",SVR(kernel="rbf"))]),{"model__C":[1,10],"model__epsilon":[0.1,0.5],"model__gamma":["scale",0.01]},"raw"),
      "v3_extra_trees":(Pipeline([("model",ExtraTreesRegressor(random_state=seed,n_estimators=250,n_jobs=-1))]),{"model__max_features":[0.5,1.0],"model__min_samples_leaf":[1,3]},"raw"),
      "v3_random_forest":(Pipeline([("model",RandomForestRegressor(random_state=seed,n_estimators=250,n_jobs=-1,bootstrap=True))]),{"model__max_features":[0.5,1.0],"model__min_samples_leaf":[1,3]},"raw"),
      "v3_gradient_boosting":(Pipeline([("model",GradientBoostingRegressor(random_state=seed,loss="huber"))]),{"model__n_estimators":[100,300],"model__learning_rate":[0.03,0.1],"model__max_depth":[1,2]},"raw"),
      "v3_gaussian_process":(Pipeline([("scale",StandardScaler()),("pca",PCA(whiten=True,random_state=seed)),("model",GaussianProcessRegressor(kernel=fixed,optimizer=None,normalize_y=True,random_state=seed))]),{"pca__n_components":gpca,"model__alpha":[1e-4,1e-2]},"raw"),
    }

def v4_defs(seed):
    specs={}
    for name in ["huber","bayesian_ridge","huber_spline","bayesian_ridge_spline","ridge",
                 "hist_gbm","kernel_ridge_rbf","gaussian_process","extra_trees"]:
        est,grid=build_v4_model(name,seed)
        # keep the original family, but cap especially large grids for this all-model sweep
        if name=="hist_gbm":
            grid={"model__max_depth":[2,3],"model__learning_rate":[0.03,0.08],
                  "model__min_samples_leaf":[10,20],"model__l2_regularization":[0.0,1.0]}
        elif name=="extra_trees":
            grid={"model__max_depth":[4,None],"model__min_samples_leaf":[2,4],"model__max_features":[0.3,0.6]}
        elif name=="huber":
            grid={"model__alpha":[0.01,0.1,1.0],"model__epsilon":[1.2,1.6]}
        elif name=="kernel_ridge_rbf":
            grid={"model__alpha":[0.01,0.1,1.0],"model__gamma":[0.001,0.01]}
        feature={"huber":"physical","bayesian_ridge":"physical","huber_spline":"physical",
                 "bayesian_ridge_spline":"physical","ridge":"raw","hist_gbm":"phys_raw",
                 "kernel_ridge_rbf":"raw","gaussian_process":"raw","extra_trees":"phys_raw"}[name]
        specs["v4_"+name]=(est,grid,feature)
    specs["v4_spectral_angle"]=(SpectralAngleRegressor(degree=1,fit_origin=False),
                                {"degree":[1,2],"fit_origin":[False,True]},"raw")
    return specs

def tune(est,grid,X,y,g,inner_folds,link):
    """Tune on parent-mean MAE in original FeO wt.% after inverse link."""
    combos=list(ParameterGrid(grid)) if grid else [{}]
    ng=len(np.unique(g)); splitter=GroupKFold(n_splits=max(2,min(inner_folds,ng)))
    best=None; best_score=np.inf
    for params in combos:
        vals=[]; ok=True
        for tr,va in splitter.split(X,y,g):
            m=clone(est); m.set_params(**params)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X[tr],forward_y(y[tr],link))
                    pred=inverse_z(m.predict(X[va]),link)
                ya,yp=parent_means(y[va],pred,g[va])
                vals.append(float(np.mean(np.abs(yp-ya))))
            except Exception:
                ok=False; break
        if ok and vals and np.mean(vals)<best_score:
            best_score=float(np.mean(vals)); best=params
    if best is None: best={}
    m=clone(est); m.set_params(**best)
    return m,best

def run_one(df,target,subset,label,est,grid,feature,link,outdir,seed=20261110):
    mask=df[target].notna()
    if subset=="exact_only": mask &= df[SCOPE].isin(EXACT_SCOPES)
    d=df.loc[mask].reset_index(drop=True)
    bands=d[BAND_COLUMNS].to_numpy(float)
    X,names=build_features(bands,feature)
    y=d[target].to_numpy(float)
    yz=forward_y(y,link)
    g=d[GROUP].to_numpy()
    outer=GroupKFold(n_splits=min(5,len(np.unique(g))))
    oof=np.full(len(d),np.nan); fold=np.full(len(d),-1)
    fold_params=[]
    for k,(tr,te) in enumerate(outer.split(X,y,g)):
        m,params=tune(est,grid,X[tr],y[tr],g[tr],3,link)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr],forward_y(y[tr],link))
        oof[te]=inverse_z(m.predict(X[te]),link); fold[te]=k
        fold_params.append({"fold":k,"params":params})
    ya,yp=parent_means(y,oof,g)
    mm=metrics(y,oof); pm=metrics(ya,yp)
    row={"target":target,"subset":subset,"model":label,"feature_family":feature,"target_link":link,
         "n_spectra":len(d),"n_parents":len(np.unique(g)),
         **{"measurement_"+k:v for k,v in mm.items()},
         **{"parent_"+k:v for k,v in pm.items()},
         "min_oof_wt_percent":float(np.nanmin(oof)),"max_oof_wt_percent":float(np.nanmax(oof)),
         "all_oof_physical_0_100":bool(np.all((oof>=0)&(oof<=100)))}
    run=outdir/f"{target}__{subset}"/link/label
    run.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":d["id"],"apollo_parent":g,"target_wt_percent":y,
                  "prediction_wt_percent":oof,"prediction_fraction":oof/100.0,"fold":fold}).to_csv(run/"oof_predictions.csv",index=False)
    # final tune + fit all
    final,params=tune(est,grid,X,y,g,3,link)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(X,yz)
    artifact=DeployableBoundedModel(final,feature,link,list(names),label,target)
    joblib.dump(artifact,run/"final_model.joblib",compress=3)
    (run/"manifest.json").write_text(json.dumps({"summary":row,"fold_params":fold_params,
        "final_params":params,"band_columns":BAND_COLUMNS},indent=2,default=str))
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="data/ml_combined_v2/m3_feo_isfeo_ground_truth_v2.csv")
    ap.add_argument("--output-dir",default="data/ml_bounded_v5")
    args=ap.parse_args()
    df=pd.read_csv(args.dataset)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    models={**v3_defs(20261110,74),**v4_defs(20261110)}
    targets=[("feo_wt_percent_mean","all"),("feo_direct_wt_percent_mean","exact_only")]
    links=["logit_fraction","tanh_softplus_fraction"]
    rows=[]
    for target,subset in targets:
        for link in links:
            for label,(est,grid,feature) in models.items():
                print(f"RUN {target} {link} {label}",flush=True)
                try:
                    rows.append(run_one(df,target,subset,label,est,grid,feature,link,out))
                except Exception as e:
                    rows.append({"target":target,"subset":subset,"model":label,"target_link":link,
                                 "status":"failed","error":f"{type(e).__name__}: {e}"})
                    print("FAILED",label,e,flush=True)
    tab=pd.DataFrame(rows)
    tab.to_csv(out/"bounded_model_comparison.csv",index=False)
    # concise ranking markdown
    md=["# Bounded FeO model comparison","",
        "All predictions use a target link that mathematically guarantees FeO fraction in (0,1); metrics are on the original wt.% scale.",""]
    for (t,l),q in tab[tab.get("status",pd.Series(["ok"]*len(tab))).fillna("ok")!="failed"].groupby(["target","target_link"]):
        q=q.sort_values("parent_mae")
        md += [f"## {t} — {l}","",q[["model","feature_family","n_parents","parent_mae","parent_rmse","parent_r2","min_oof_wt_percent","max_oof_wt_percent"]].to_markdown(index=False,floatfmt=".4f"),""]
    (out/"RESULTS.md").write_text("\n".join(md))
    (out/"RESULTS_COMPLETE").write_text("ok\n")
    print(tab.sort_values(["target","target_link","parent_mae"]).to_string(index=False))

if __name__=="__main__":
    main()
