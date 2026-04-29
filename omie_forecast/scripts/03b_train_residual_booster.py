"""
03b_train_residual_booster.py — LightGBM Residual Booster (Option 1)
====================================================================
For each clock hour (H00–H23), fits a LightGBM regressor on the *residuals*
of the LASSO model:

    residual_i = y_i - lasso_oof_pred_i
    booster ≈ E[residual | X]
    final_prediction = lasso(X) + booster(X)

The residuals are obtained via out-of-fold TimeSeriesSplit predictions from a
fresh LassoCV fit on each train fold — this prevents the booster from simply
learning to undo LASSO's in-sample overfit. Tree models don't extrapolate,
so the combined forecaster inherits LASSO's strong linear signal but gains a
non-linear correction that is bounded by the residual range seen in training.

Inputs:  data/processed/hourly_dataset_h{HH}.csv  (×24)
         models/hour_{HH}.pkl                      (×24)   — existing LASSO
Outputs: models/hour_{HH}_booster.pkl              (×24)
         models/booster_summary.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "lightgbm is required. Install with:  pip install lightgbm>=4.0.0"
    ) from exc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_booster")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)
_fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# LightGBM hyperparameters — deliberately conservative.
# The booster corrects *residuals* (target has mean ~0, smaller variance than y)
# so we need a shallow, well-regularised model to avoid fitting noise.
# ---------------------------------------------------------------------------
LGB_PARAMS = dict(
    objective="regression_l1",     # MAE — residuals have heavy tails
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,                 # shallow trees
    max_depth=5,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


def _winsorize(X: np.ndarray, bounds: dict, feats: list[str]) -> np.ndarray:
    """Clip each column of X to (lo, hi) from bounds."""
    X = X.copy()
    for i, f in enumerate(feats):
        b = bounds.get(f)
        if b is None:
            continue
        lo, hi = b
        X[:, i] = np.clip(X[:, i], lo, hi)
    return X


def train_booster(hour: int) -> dict:
    """Train a LightGBM residual corrector for a single clock hour."""
    csv_path   = config.DATA_PROCESSED / f"hourly_dataset_h{hour:02d}.csv"
    lasso_path = config.MODELS_DIR / f"hour_{hour:02d}.pkl"

    if not csv_path.exists() or not lasso_path.exists():
        logger.warning("H%02d: missing data or LASSO model", hour)
        return {}

    df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
    lasso_bundle: dict = joblib.load(lasso_path)
    feats: list[str] = lasso_bundle["all_features"]
    bounds: dict = lasso_bundle.get("feature_bounds", {})
    target_mode: str = lasso_bundle.get("target_mode", "absolute")

    feats = [f for f in feats if f in df.columns]
    df_clean = df.dropna(subset=["price_es"] + feats).copy()
    if len(df_clean) < 200:
        logger.warning("H%02d: only %d clean rows — skipping", hour, len(df_clean))
        return {}

    X = df_clean[feats].values.astype(float)
    # Step 2a — match the target the LASSO was trained on. If the LASSO
    # predicts deviation-from-rolling-mean, the booster predicts residual
    # of *that* deviation, not residual of the absolute price.
    if target_mode == "deviation_from_roll_mean_24h":
        rmean = df_clean["price_roll_mean_24h"].fillna(
            df_clean["price_es"].mean()
        ).values.astype(float)
        y = df_clean["price_es"].values.astype(float) - rmean
    else:
        y = df_clean["price_es"].values.astype(float)

    # Winsorize features BEFORE computing residuals so the booster is trained
    # on the same feature space it will see at forecast time.
    X_w = _winsorize(X, bounds, feats) if bounds else X

    # ---------- Out-of-fold LASSO predictions ----------
    tscv = TimeSeriesSplit(
        n_splits=config.WALK_FORWARD_SPLITS,
        gap=config.WALK_FORWARD_GAP_DAYS,
    )
    oof_pred = np.full_like(y, np.nan, dtype=float)

    # Sample weights — same regime as 03_train_models.py
    weights = np.ones(len(y))
    if "storm_flag" in df_clean.columns:
        weights[df_clean["storm_flag"].values == 1] = config.SAMPLE_WEIGHT_ANOMALY
    if "spike_flag" in df_clean.columns:
        weights[df_clean["spike_flag"].values == 1] = config.SAMPLE_WEIGHT_SPIKE

    for fold, (tr, te) in enumerate(tscv.split(X_w)):
        lasso_fold = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso",  LassoCV(
                cv=5,
                max_iter=config.LASSO_MAX_ITER,
                n_alphas=config.LASSO_N_ALPHAS,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        try:
            lasso_fold.fit(X_w[tr], y[tr], lasso__sample_weight=weights[tr])
            oof_pred[te] = lasso_fold.predict(X_w[te])
        except Exception as exc:
            logger.debug("H%02d fold %d OOF failed: %s", hour, fold, exc)

    # Only rows with an OOF prediction
    mask = ~np.isnan(oof_pred)
    if mask.sum() < 100:
        logger.warning("H%02d: only %d OOF rows — skipping", hour, mask.sum())
        return {}

    X_oof   = X_w[mask]
    y_oof   = y[mask]
    p_oof   = oof_pred[mask]
    res_oof = y_oof - p_oof  # what the booster must predict

    # ---------- Fit LightGBM on residuals ----------
    # Use an 85/15 internal split so LightGBM can early-stop.
    split_idx = int(len(X_oof) * 0.85)
    X_tr, X_va = X_oof[:split_idx], X_oof[split_idx:]
    r_tr, r_va = res_oof[:split_idx], res_oof[split_idx:]
    w_oof = weights[mask]
    w_tr  = w_oof[:split_idx]

    booster = lgb.LGBMRegressor(**LGB_PARAMS)
    try:
        booster.fit(
            X_tr, r_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, r_va)] if len(X_va) > 20 else None,
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(0)] if len(X_va) > 20 else None,
        )
    except Exception as exc:
        logger.warning("H%02d: LightGBM fit failed (%s) — trying without eval",
                       hour, exc)
        booster = lgb.LGBMRegressor(**LGB_PARAMS)
        booster.fit(X_tr, r_tr, sample_weight=w_tr)

    # ---------- Diagnostics ----------
    mae_lasso_only = float(mean_absolute_error(y_oof, p_oof))
    combined_pred  = p_oof + booster.predict(X_oof)
    mae_combined   = float(mean_absolute_error(y_oof, combined_pred))
    improvement    = 100 * (mae_lasso_only - mae_combined) / mae_lasso_only

    # Top-5 features by LightGBM gain importance
    try:
        imp = booster.booster_.feature_importance(importance_type="gain")
        top5_idx = np.argsort(imp)[::-1][:5]
        top5 = [feats[i] for i in top5_idx if imp[i] > 0]
    except Exception:
        top5 = []

    # Residual std for bounding the correction at forecast time
    res_std = float(np.std(res_oof))

    bundle: dict = {
        "booster":        booster,
        "all_features":   feats,
        "feature_bounds": bounds,
        "hour":           hour,
        "residual_std":   res_std,
        "mae_lasso":      round(mae_lasso_only, 3),
        "mae_combined":   round(mae_combined, 3),
        "improvement_pct": round(improvement, 2),
        "top5_features":  top5,
        "train_rows":     int(mask.sum()),
        "trained_on":     str(pd.Timestamp.today().date()),
        # Step 2a — booster trained on residuals of (price - rmean).
        "target_mode":    target_mode,
    }

    out_path = config.MODELS_DIR / f"hour_{hour:02d}_booster.pkl"
    joblib.dump(bundle, out_path)

    logger.info(
        "H%02d: MAE lasso=%.2f → combined=%.2f  (Δ=%+.1f%%)  res_σ=%.2f",
        hour, mae_lasso_only, mae_combined, improvement, res_std,
    )
    return bundle


def train_all_boosters() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("TRAINING 24 HOURLY RESIDUAL BOOSTERS (LightGBM)")
    logger.info("=" * 60)

    summaries: list[dict] = []
    for h in config.HOURS:
        bundle = train_booster(h)
        if bundle:
            summaries.append({
                "hour":            bundle["hour"],
                "mae_lasso":       bundle["mae_lasso"],
                "mae_combined":    bundle["mae_combined"],
                "improvement_pct": bundle["improvement_pct"],
                "residual_std":    round(bundle["residual_std"], 3),
                "top_5_features":  ",".join(bundle["top5_features"]),
                "train_rows":      bundle["train_rows"],
            })

    summary_df = pd.DataFrame(summaries)
    summary_path = config.MODELS_DIR / "booster_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Booster summary → %s", summary_path)

    if not summary_df.empty:
        logger.info(
            "Avg MAE: lasso=%.2f → combined=%.2f  (avg Δ=%+.1f%%)",
            summary_df["mae_lasso"].mean(),
            summary_df["mae_combined"].mean(),
            summary_df["improvement_pct"].mean(),
        )
    logger.info("=" * 60)
    return summary_df


if __name__ == "__main__":
    train_all_boosters()
