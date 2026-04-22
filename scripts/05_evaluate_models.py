"""
05_evaluate_models.py — Model Evaluation & Reporting
=====================================================
Generates comprehensive evaluation charts, feature importance plots,
and summary statistics from the walk-forward validation results.

Inputs:  models/*.pkl
         outputs/forecasts/walkforward_results.csv
         data/processed/master_dataset.csv
Outputs: outputs/forecasts/model_evaluation.csv
         outputs/charts/feature_importance.png
         outputs/charts/actual_vs_pred_{contract}.png
         outputs/charts/residuals_walkforward.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("evaluate_models")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# ===================================================================
# Feature importance plot
# ===================================================================

def plot_feature_importance(contract: str, model_bundle: dict, path: Path) -> None:
    """Plot XGBoost feature importances and save as PNG."""
    xgb_model = model_bundle.get("xgb")
    feature_cols = model_bundle.get("feature_cols", [])

    if xgb_model is None or not feature_cols:
        logger.warning("No XGBoost model or features for %s — skipping importance plot.", contract)
        return

    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [feature_cols[i] for i in sorted_idx]
    values = importances[sorted_idx]
    ax.barh(range(len(names)), values[::-1], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"XGBoost Feature Importance — {contract}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance chart saved to %s", path)


# ===================================================================
# Actual vs Predicted chart
# ===================================================================

def plot_actual_vs_predicted(
    contract: str,
    df: pd.DataFrame,
    model_bundle: dict,
    path: Path,
) -> None:
    """Plot actual vs predicted for a contract and save as PNG."""
    target_col = model_bundle.get("target_col", "omip_yr1")
    feature_cols = model_bundle.get("feature_cols", [])
    ridge = model_bundle.get("ridge")
    xgb = model_bundle.get("xgb")
    ridge_w = model_bundle.get("ridge_weight", 0.5)
    xgb_w = model_bundle.get("xgb_weight", 0.5)

    if ridge is None or xgb is None:
        logger.warning("Models missing for %s — skipping actual vs pred chart.", contract)
        return

    available = [f for f in feature_cols if f in df.columns]
    missing = [f for f in feature_cols if f not in df.columns]
    for f in missing:
        df[f] = 0.0

    subset = df.dropna(subset=[target_col])
    if subset.empty:
        logger.warning("No valid target data for %s.", contract)
        return

    X = subset[feature_cols].ffill().fillna(0.0)
    y = subset[target_col]

    ridge_pred = ridge.predict(X)
    xgb_pred = xgb.predict(X)
    ensemble_pred = ridge_w * ridge_pred + xgb_w * (ridge_pred + xgb_pred)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y.index, y.values, label="Actual", color="black", linewidth=1)
    ax.plot(y.index, ensemble_pred, label="Predicted", color="crimson", linewidth=1, alpha=0.8)
    ax.set_title(f"Actual vs Predicted — {contract}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Actual vs predicted chart saved to %s", path)


# ===================================================================
# Walk-forward residuals plot
# ===================================================================

def plot_walkforward_residuals(
    df: pd.DataFrame,
    all_bundles: dict[str, dict],
    path: Path,
) -> None:
    """Plot walk-forward residuals across all contracts."""
    fig, axes = plt.subplots(len(all_bundles), 1, figsize=(14, 4 * len(all_bundles)), squeeze=False)

    for i, (contract, bundle) in enumerate(all_bundles.items()):
        ax = axes[i, 0]
        target_col = bundle.get("target_col", "omip_yr1")
        feature_cols = bundle.get("feature_cols", [])
        ridge = bundle.get("ridge")
        xgb = bundle.get("xgb")
        ridge_w = bundle.get("ridge_weight", 0.5)
        xgb_w = bundle.get("xgb_weight", 0.5)

        if ridge is None or xgb is None:
            ax.set_title(f"{contract} — no model")
            continue

        for f in feature_cols:
            if f not in df.columns:
                df[f] = 0.0

        subset = df.dropna(subset=[target_col])
        if subset.empty:
            ax.set_title(f"{contract} — no data")
            continue

        X = subset[feature_cols].ffill().fillna(0.0)
        y = subset[target_col]

        ridge_pred = ridge.predict(X)
        xgb_pred = xgb.predict(X)
        ensemble_pred = ridge_w * ridge_pred + xgb_w * (ridge_pred + xgb_pred)
        residuals = y.values - ensemble_pred

        ax.plot(y.index, residuals, color="steelblue", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"Residuals — {contract}")
        ax.set_ylabel("€/MWh")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Walk-forward residuals chart saved to %s", path)


# ===================================================================
# Main
# ===================================================================

def evaluate_all() -> None:
    """Generate all evaluation outputs."""
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    df = pd.read_csv(config.MASTER_DATASET, index_col="date", parse_dates=True)

    # Load walk-forward metrics if available
    wf_path = config.FORECASTS_DIR / "walkforward_results.csv"
    if wf_path.exists():
        wf_metrics = pd.read_csv(wf_path)
        eval_path = config.FORECASTS_DIR / "model_evaluation.csv"
        wf_metrics.to_csv(eval_path, index=False)
        logger.info("Evaluation metrics saved to %s", eval_path)
    else:
        wf_metrics = pd.DataFrame()
        logger.warning("No walk-forward results found at %s", wf_path)

    # Load model bundles and generate charts
    all_bundles: dict[str, dict] = {}
    for contract in config.CONTRACTS:
        model_path = config.MODELS_DIR / f"model_{contract}.pkl"
        if not model_path.exists():
            logger.warning("Model file not found: %s", model_path)
            continue
        bundle = joblib.load(model_path)
        all_bundles[contract] = bundle

        # Feature importance
        plot_feature_importance(
            contract, bundle,
            config.CHARTS_DIR / f"feature_importance_{contract}.png",
        )

        # Actual vs predicted
        plot_actual_vs_predicted(
            contract, df.copy(), bundle,
            config.CHARTS_DIR / f"actual_vs_pred_{contract}.png",
        )

    # Combined feature importance (first available)
    if all_bundles:
        first_contract = next(iter(all_bundles))
        plot_feature_importance(
            "ALL", all_bundles[first_contract],
            config.CHARTS_DIR / "feature_importance.png",
        )

    # Walk-forward residuals
    if all_bundles:
        plot_walkforward_residuals(
            df.copy(), all_bundles,
            config.CHARTS_DIR / "residuals_walkforward.png",
        )

    # Print summary table
    if not wf_metrics.empty:
        summary = wf_metrics.groupby("contract")[["mae", "rmse", "mape", "directional_accuracy"]].mean()
        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY (average across folds)")
        print("=" * 60)
        print(summary.to_string())
        print()
    else:
        print("No walk-forward metrics available to summarize.")

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    evaluate_all()
