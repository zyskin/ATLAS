"""Deployment-safe bounded FeO target links and model wrapper."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from m3_feo_ml_v4.features import build_features

EPS=1e-7

def sigmoid(z):
    z=np.asarray(z,float); out=np.empty_like(z)
    pos=z>=0
    out[pos]=1/(1+np.exp(-z[pos]))
    ez=np.exp(z[~pos]); out[~pos]=ez/(1+ez)
    return out

def softplus(z):
    z=np.asarray(z,float)
    return np.log1p(np.exp(-np.abs(z)))+np.maximum(z,0)

def softplus_inv(x):
    x=np.asarray(x,float)
    return x+np.log(-np.expm1(-x))

def forward_y(y_wt,link):
    f=np.clip(np.asarray(y_wt,float)/100.0,EPS,1-EPS)
    if link=="logit_fraction":
        return np.log(f)-np.log1p(-f)
    if link=="tanh_softplus_fraction":
        theta=np.arctanh(f)
        return softplus_inv(np.maximum(theta,EPS))
    raise ValueError(link)

def inverse_z(z,link):
    z=np.asarray(z,float)
    if link=="logit_fraction":
        f=sigmoid(z)
    elif link=="tanh_softplus_fraction":
        f=np.tanh(softplus(z))
    else:
        raise ValueError(link)
    return (100.0*f).reshape(-1)

@dataclass
class DeployableBoundedModel:
    estimator: object
    feature_family: str
    target_link: str
    feature_names: list[str]
    model_label: str
    target_column: str

    def _X(self,bands):
        X,names=build_features(np.asarray(bands,float),self.feature_family)
        if list(names)!=list(self.feature_names):
            raise ValueError("feature definition mismatch")
        return X

    def predict_wt_percent(self,bands):
        return inverse_z(self.estimator.predict(self._X(bands)),self.target_link)

    def predict_fraction(self,bands):
        return self.predict_wt_percent(bands)/100.0
