"""
04_evaluate_models.py — Walk-Forward Evaluation
================================================
Runs out-of-sample walk-forward evaluation across all 24 hourly LASSO models
using TimeSeriesSplit(n_splits=12, gap=7).

Inputs:  data/processed/hourly_dataset_h{HH}.csv  (×24)
         models/hour_{HH}.pkl                      (×24)
Outputs: outputs/forecasts/evaluation_results.csv  — per-fold metrics
         outputs/forecasts/hourly_summary.csv      — per-hour averages
         outputs/charts/mae_by_hour.png
         outputs/charts/feature_importance_heatmap.png
         outputs/charts/residuals_by_hour.png
         outputs/charts/actual_vs_predicted_h{HH}.png  (×24)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("evaluate_models")
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


# ===================================================================
# Metrics
# ===================================================================

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero actuals."""
    mask = np.abs(y_true) > 1.0
    if mask.sum() == 0:
        return np.nan
    return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _directional(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of periods where predicted direction matches actual direction."""
    if len(y_true) < 2:
        return np.nan
    actual_dir = np.diff(y_true) > 0
    pred_dir   = np.diff(y_pred) > 0
    return float(100 * np.mean(actual_dir == pred_dir))


# ===================================================================
# Per-hour evaluation
# ===================================================================

def evaluate_hour(hour: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Walk-forward evaluation for a single clock hour.

    Returns:
        (fold_records, y_true_all, y_pred_all)
    """
    csv_path   = config.DATA_PROCESSED / f"hourly_dataset_h{hour:02d}.csv"
    model_path = config.MODELS_DIR / f"hour_{hour:02d}.pkl"

    if not csv_path.exists() or not model_path.exists():
        logger.warning("H%02d: missing data or model", hour)
        return [], np.array([]), np.array([])

    df = pd.read_csv(csv_path, index_col="datetime", parse_dates=True)
    bundle: dict = joblib.load(model_path)
    feats = bundle["all_features"]
    pipeline: Pipeline = bundle["pipeline"]

    feats = [f for f in feats if f in df.columns]
    df_clean = df.dropna(subset=["price_es"] + feats).copy()
    if len(df_clean) < 100:
        return [], np.array([]), np.array([])

    X = df_clean[feats].values.astype(float)
    y = df_clean["price_es"].values.astype(float)

    tscv = TimeSeriesSplit(n_splits=config.WALK_FORWARD_SPLITS,
                           gap=config.WALK_FORWARD_GAP_DAYS)

    fold_records: list[dict] = []
    y_true_all:  list[float] = []
    y_pred_all:  list[float] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]
        if len(X_tr) < 30 or len(X_te) < 5:
            continue
        try:
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_te)
        except Exception as exc:
            logger.debug("H%02d fold %d fit failed: %s", hour, fold, exc)
            continue

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mape = _mape(y_te, y_pred)
        da   = _directional(y_te, y_pred)

        fold_records.append({
            "hour": hour, "fold": fold,
            "mae": round(mae, 3), "rmse": round(rmse, 3),
            "mape": round(mape, 3), "directional_accuracy": round(da, 2),
            "n_test": len(y_te),
        })
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    logger.info("H%02d: %d folds  avg MAE=%.2f  avg RMSE=%.2f",
                hour,
                len(fold_records),
                np.mean([r["mae"] for r in fold_records]) if fold_records else np.nan,
                np.mean([r["rmse"] for r in fold_records]) if fold_records else np.nan)

    return fold_records, np.array(y_true_all), np.array(y_pred_all)


# ===================================================================
# Charts
# ===================================================================

def _chart_mae_by_hour(hourly_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = hourly_summary["hour"].values
    maes  = hourly_summary["mae_mean"].values
    bars  = ax.bar(hours, maes, color="steelblue", edgecolor="white", linewidth=0.5)
    # Highlight peak hours
    peak  = [10, 11, 12, 19, 20, 21]
    for bar, h in zip(bars, hours):
        if h in peak:
            bar.set_color("crimson")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Absolute Error (€/MWh)")
    ax.set_title("Walk-Forward MAE by Hour — OMIE Day-Ahead")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"H{h:02d}" for h in hours], fontsize=8, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "mae_by_hour.png", dpi=150)
    plt.close()
    logger.info("Saved mae_by_hour.png")


def _chart_actual_vs_predicted(hour: int, y_true: np.ndarray,
                                y_pred: np.ndarray) -> None:
    if len(y_true) < 10:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect forecast")
    # OLS trendline
    from numpy.polynomial.polynomial import polyfit
    try:
        c, m = polyfit(y_true, y_pred, 1)
        trend_x = np.linspace(lims[0], lims[1], 100)
        ax.plot(trend_x, c + m * trend_x, "g-", linewidth=1.5, label="OLS trend")
        r2 = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)
        ax.annotate(f"R² = {r2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=11, color="green")
    except Exception:
        pass
    ax.set_xlabel("Actual Price (€/MWh)")
    ax.set_ylabel("Predicted Price (€/MWh)")
    ax.set_title(f"Actual vs Predicted — H{hour:02d}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / f"actual_vs_predicted_h{hour:02d}.png", dpi=120)
    plt.close()


def _chart_feature_heatmap(bundles: dict[int, dict]) -> None:
    """Heatmap of absolute LASSO coefficients across all 24 hours."""
    # Collect all feature names that appear in at least one model
    all_feats: set[str] = set()
    for b in bundles.values():
        all_feats.update(b.get("selected_features", []))
    all_feats_sorted = sorted(all_feats)
    if not all_feats_sorted:
        return

    data = np.zeros((len(all_feats_sorted), 24))
    for h in range(24):
        b = bundles.get(h, {})
        coef = b.get("coef", {})
        for i, feat in enumerate(all_feats_sorted):
            data[i, h] = abs(coef.get(feat, 0.0))

    fig, ax = plt.subplots(figsize=(14, max(8, len(all_feats_sorted) * 0.28)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"H{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(all_feats_sorted)))
    ax.set_yticklabels(all_feats_sorted, fontsize=7)
    ax.set_xlabel("Hour of Day")
    ax.set_title("LASSO Feature Importance (|coefficient|) — 24 Hour Models")
    plt.colorbar(im, ax=ax, label="|coef|")
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "feature_importance_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    logger.info("Saved feature_importance_heatmap.png")


def _chart_residuals_by_hour(all_hour_residuals: dict[int, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    data   = [all_hour_residuals.get(h, np.array([])) for h in range(24)]
    labels = [f"H{h:02d}" for h in range(24)]
    bp = ax.boxplot([d for d in data], labels=labels, patch_artist=True,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
                    medianprops={"color": "red", "linewidth": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Residual (€/MWh)")
    ax.set_title("Forecast Residuals by Hour")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    plt.savefig(config.CHARTS_DIR / "residuals_by_hour.png", dpi=150)
    plt.close()
    logger.info("Saved residuals_by_hour.png")


# ===================================================================
# Main
# ===================================================================

def evaluate_all() -> pd.DataFrame:
    """Run walk-forward evaluation for all 24 models."""
    logger.info("=" * 60)
    logger.info("WALK-FORWARD EVALUATION (all 24 hours)")
    logger.info("=" * 60)

    all_records: list[dict]       = []
    all_residuals: dict[int, np.ndarray] = {}
    all_yt: dict[int, np.ndarray] = {}
    all_yp: dict[int, np.ndarray] = {}

    for h in config.HOURS:
        recs, y_true, y_pred = evaluate_hour(h)
        all_records.extend(recs)
        if len(y_true):
            all_residuals[h] = y_pred - y_true
            all_yt[h] = y_true
            all_yp[h] = y_pred
            _chart_actual_vs_predicted(h, y_true, y_pred)

    # Save residuals for use in 05_forecast.py prediction intervals
    residuals_df = pd.DataFrame({
        f"h{h:02d}": pd.Series(all_residuals.get(h, np.array([])))
        for h in config.HOURS
    })
    residuals_df.to_csv(config.FORECASTS_DIR / "eval_residuals.csv", index=False)

    # Per-fold results
    results_df = pd.DataFrame(all_records)
    results_path = config.FORECASTS_DIR / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    # Per-hour summary
    if results_df.empty:
        logger.warning("No evaluation results — check model and data files")
        return results_df

    summary = (results_df
               .groupby("hour")[["mae", "rmse", "mape", "directional_accuracy"]]
               .mean()
               .rename(columns={"mae": "mae_mean", "rmse": "rmse_mean",
                                "mape": "mape_mean",
                                "directional_accuracy": "da_mean"})
               .reset_index())
    summary.to_csv(config.FORECASTS_DIR / "hourly_summary.csv", index=False)

    # Charts
    _chart_mae_by_hour(summary)
    _chart_residuals_by_hour(all_residuals)

    # Load model bundles for heatmap
    bundles: dict[int, dict] = {}
    for h in config.HOURS:
        p = config.MODELS_DIR / f"hour_{h:02d}.pkl"
        if p.exists():
            bundles[h] = joblib.load(p)
    _chart_feature_heatmap(bundles)

    # Console summary
    print("\n" + "=" * 72)
    print(f"{'Hour':>5} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>7} | {'DA%':>6} | Features")
    print("-" * 72)
    for _, row in summary.iterrows():
        h = int(row["hour"])
        n_feat = bundles[h]["n_features_selected"] if h in bundles else "—"
        print(f"  H{h:02d} | {row['mae_mean']:>8.2f} | {row['rmse_mean']:>8.2f} | "
              f"{row['mape_mean']:>6.1f}% | {row['da_mean']:>5.1f}% | {n_feat}")
    mean_row = summary[["mae_mean", "rmse_mean", "mape_mean", "da_mean"]].mean()
    print("-" * 72)
    print(f"  MEAN | {mean_row['mae_mean']:>8.2f} | {mean_row['rmse_mean']:>8.2f} | "
          f"{mean_row['mape_mean']:>6.1f}% | {mean_row['da_mean']:>5.1f}%")
    print("=" * 72 + "\n")

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE — avg MAE %.2f EUR/MWh", mean_row["mae_mean"])
    logger.info("=" * 60)
    return results_df


if __name__ == "__main__":
    evaluate_all()
