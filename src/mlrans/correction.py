"""Scaffold for the interpretable eddy-viscosity correction model.

Deliberately minimal and NOT yet trained on final data. It provides:
  * a small set of interpretable estimators (ridge -> random forest -> GBM);
  * GEOMETRY-WISE cross-validation helpers (train on some hill slopes, test on
    unseen ones), because random point-wise splits leak spatial/geometry
    information and overstate generalisation on a single flow field;
  * feature-importance extraction for interpretability.

The learning target is beta_nut (see mlrans.features). This is an a-priori
correction study: predict the correction from RANS-local features, evaluate it
against DNS-derived truth on *unseen* geometries. Coupling back into OpenFOAM
is explicitly out of scope for now.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str = "ridge", **kwargs) -> BaseEstimator:
    """Return an interpretable-first regressor by name.

    'ridge'             : linear, most interpretable baseline (scaled inputs).
    'random_forest'     : nonlinear, gives feature importances, robust.
    'gradient_boosting' : nonlinear, usually strongest of the three.
    """
    if name == "ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=kwargs.get("alpha", 1.0))),
        ])
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", None),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=kwargs.get("random_state", 0),
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 400),
            max_depth=kwargs.get("max_depth", 3),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=kwargs.get("random_state", 0),
        )
    raise ValueError(f"Unknown model '{name}'. "
                     "Choose ridge | random_forest | gradient_boosting.")


def geometry_wise_cv(X, y, groups, model_name: str = "random_forest", **kwargs):
    """Leave-one-geometry-out CV: each fold trains on all but one hill slope.

    ``groups`` labels every sample by its source case (e.g. 'case_1p0'). This is
    the honest generalisation test for this project: performance on a geometry
    the model never saw during training.

    Returns a dict with per-fold and mean (RMSE, R^2).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    logo = LeaveOneGroupOut()

    fold_rmse, fold_r2, held_out = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        model = make_model(model_name, **kwargs)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_rmse.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        fold_r2.append(float(r2_score(y[test_idx], pred)))
        held_out.append(str(np.unique(groups[test_idx])[0]))

    return {
        "held_out_case": held_out,
        "fold_rmse": fold_rmse,
        "fold_r2": fold_r2,
        "mean_rmse": float(np.mean(fold_rmse)),
        "mean_r2": float(np.mean(fold_r2)),
    }


def geometry_wise_oof(X, y, groups, model_name: str = "random_forest", **kwargs):
    """Out-of-fold predictions: predict each geometry while it is held out.

    Returns an array aligned to the input rows (each row predicted by a model
    that never saw its geometry), plus the per-fold metrics. This is what the
    a-priori correction maps and honest scatter plots should be built from.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    oof = np.full(y.shape, np.nan)
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups):
        model = make_model(model_name, **kwargs)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X[test_idx])
    return oof


def feature_importances(model, feature_names) -> dict:
    """Extract interpretable importances/coefficients from a fitted model."""
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    else:
        raise AttributeError("Model exposes neither feature_importances_ nor coef_.")
    order = np.argsort(vals)[::-1]
    return {feature_names[i]: float(vals[i]) for i in order}
