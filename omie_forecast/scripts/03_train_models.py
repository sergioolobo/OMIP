"""
03_train_models.py — LASSO Model Training
==========================================
Trains 24 independent LassoCV models, one per clock hour (H00–H23), using
walk-forward cross-validation and sample-weighted downweighting of the storm
anomaly and price-spike periods.

Inputs:  data/processed/hourly_dataset_h{HH}.csv  (×24)
Outputs: models/hour_{HH}.pkl                     (×24)
         models/training_summary.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_models")
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


def _available_features(df_cols: list[str], feature_list: list[str]) -> list[str]:
    """Return only features that actually exist as columns in the dataset."""
    present = set(df_cols)
    missing = [f for f in feature_list if f not in present]
    if missing:
        logger.debug("  Features not in dataset (will skip): %s", missing[:10])
    return [f for f in feature_list if f in present]


def train_hour(hour: int) -> dict:
    """
    Train a LassoCV pipeline for a single clock hour.

    Args:
        hour: Clock hour 0–23.

    Returns:
        Model bundle dict (also saved to disk).
    """
    csv_path = config.DATA_PROCESSED / f"hourly_dataset_h{hour:02d}.csv"
    if not csv_path.exists():
        logger.warning("H%02d: dataset not found (%s)", hour, csv_path)
        return {}

    df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
    feats = _available_features(list(df.columns), config.ALL_FEATURES)

    if not feats:
        logger.warning("H%02d: no usable features — skipping", hour)
        return {}

    df_clean = df.dropna(subset=["price_es"] + feats).copy()
    if len(df_clean) < 200:
        logger.warning("H%02d: only %d clean rows — skipping", hour, len(df_clean))
        return {}

    X = df_clean[feats].values.astype(float)
    y = df_clean["price_es"].values.astype(float)

    # -----------------------------------------------------------------
    # Winsorization bounds (Option 4): 1st / 99th percentile per feature.
    # Applied at forecast time to stop LASSO from extrapolating wildly
    # when a raw feature lands in a region not seen during training.
    # -----------------------------------------------------------------
    feat_lo = np.nanpercentile(X, 1.0,  axis=0)
    feat_hi = np.nanpercentile(X, 99.0, axis=0)
    # Fallback: if lo == hi (constant feature) widen slightly
    _eq = feat_lo == feat_hi
    feat_lo = np.where(_eq, feat_lo - 1e-6, feat_lo)
    feat_hi = np.where(_eq, feat_hi + 1e-6, feat_hi)
    feature_bounds = {
        f: (float(feat_lo[i]), float(feat_hi[i]))
        for i, f in enumerate(feats)
    }

    # Sample weights — downweight storm + spike periods
    weights = np.ones(len(y))
    if "storm_flag" in df_clean.columns:
        weights[df_clean["storm_flag"].values == 1] = config.SAMPLE_WEIGHT_ANOMALY
    if "spike_flag" in df_clean.columns:
        weights[df_clean["spike_flag"].values == 1] = config.SAMPLE_WEIGHT_SPIKE

    # Walk-forward cross-validation splits
    tscv = TimeSeriesSplit(
        n_splits=config.WALK_FORWARD_SPLITS,
        gap=config.WALK_FORWARD_GAP_DAYS * 1,   # gap=N_hours, but each row = 1 day here
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(
            cv=tscv,
            max_iter=config.LASSO_MAX_ITER,
            n_alphas=config.LASSO_N_ALPHAS,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X, y, lasso__sample_weight=weights)

    lasso_model: LassoCV = pipeline.named_steps["lasso"]
    selected_mask = np.abs(lasso_model.coef_) > 1e-8
    selected_features = [f for f, s in zip(feats, selected_mask) if s]

    # Top-5 features by absolute coefficient
    coef_abs = np.abs(lasso_model.coef_)
    top5_idx = np.argsort(coef_abs)[::-1][:5]
    top5 = [feats[i] for i in top5_idx if coef_abs[i] > 1e-8]

    bundle: dict = {
        "pipeline":            pipeline,
        "all_features":        feats,
        "selected_features":   selected_features,
        "coef":                dict(zip(feats, lasso_model.coef_)),
        "alpha":               float(lasso_model.alpha_),
        "n_features_selected": int(selected_mask.sum()),
        "top5_features":       top5,
        "hour":                hour,
        "train_rows":          int(len(df_clean)),
        "trained_on":          str(pd.Timestamp.today().date()),
        # Option 4 — per-feature winsorization bounds (1st/99th percentile)
        "feature_bounds":      feature_bounds,
    }

    out_path = config.MODELS_DIR / f"hour_{hour:02d}.pkl"
    joblib.dump(bundle, out_path)

    logger.info(
        "H%02d: α=%.5f  kept=%d/%d  top=%s  (train=%d rows)",
        hour, bundle["alpha"],
        bundle["n_features_selected"], len(feats),
        ",".join(bundle["top5_features"][:3]),
        bundle["train_rows"],
    )
    return bundle


def train_all() -> pd.DataFrame:
    """Train all 24 hourly LASSO models and save a summary CSV."""
    logger.info("=" * 60)
    logger.info("TRAINING 24 HOURLY LASSO MODELS")
    logger.info("=" * 60)

    summaries: list[dict] = []
    for h in config.HOURS:
        bundle = train_hour(h)
        if bundle:
            summaries.append({
                "hour":               bundle["hour"],
                "alpha":              bundle["alpha"],
                "n_features_selected": bundle["n_features_selected"],
                "top_5_features":     ",".join(bundle["top5_features"]),
                "train_rows":         bundle["train_rows"],
            })

    summary_df = pd.DataFrame(summaries)
    summary_path = config.MODELS_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Training summary → %s", summary_path)

    if not summary_df.empty:
        logger.info("Models trained: %d/24", len(summary_df))
        logger.info("Avg alpha: %.5f  avg features kept: %.1f",
                    summary_df["alpha"].mean(),
                    summary_df["n_features_selected"].mean())
    logger.info("=" * 60)
    return summary_df


if __name__ == "__main__":
    train_all()
