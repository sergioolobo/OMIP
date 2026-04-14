"""
04_train_models.py — Model Training Module
============================================
Trains a two-layer ensemble (Ridge + XGBoost on residuals) for each
OMIP contract at two reliable horizons (7d and 30d).

Longer horizons (60d, 90d) are derived at forecast time by blending
the 7d and 30d model predictions with dampened trend extrapolation,
since shifting targets by 9-13 weeks produces unreliable models on
limited data.

Inputs:  data/processed/master_dataset.csv
Outputs: models/*.pkl
         outputs/forecasts/walkforward_results.csv
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# Horizons we actually train dedicated models for (safe shifts)
_TRAINED_HORIZONS = [7, 30]

# ---------------------------------------------------------------------------
# Predecessor contract helpers — stack same-type historical contracts
# ---------------------------------------------------------------------------
import re as _re


def _get_contract_family(contract: str) -> str:
    """Return the contract family prefix (e.g. 'q3' for Q3_26, 'yr' for YR27)."""
    c = contract.lower()
    if c.startswith("yr"):
        return "yr"
    m = _re.match(r"(q\d)", c)
    return m.group(1) if m else c


def _find_predecessor_columns(contract: str, df: pd.DataFrame) -> list[str]:
    """Find all columns for same-type contracts (e.g. all Q3 contracts for Q3_26).

    Returns list of column names sorted chronologically.
    """
    family = _get_contract_family(contract)
    prefix = f"omip_{family}_"

    predecessors = [
        c for c in df.columns
        if c.startswith(prefix)
        and "_lag_" not in c
        and "_momentum_" not in c
        and "_vol_" not in c
    ]
    return sorted(predecessors)


def _build_stacked_data(
    df: pd.DataFrame,
    contract: str,
    target_col: str,
    feature_cols: list[str],
    horizon_weeks: int,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Build a stacked training set using the target contract + all predecessors.

    Each predecessor contract contributes its own rows with the same feature
    columns (fuel prices, hydro, seasonality, etc.) but using that contract's
    own price for lag/momentum features.

    Returns (X_stacked, y_stacked, recency_weights).
    """
    predecessors = _find_predecessor_columns(contract, df)
    if not predecessors:
        return pd.DataFrame(), pd.Series(dtype=float), np.array([])

    # --- Price-regime filter: only keep predecessors in a similar price band ---
    # Avoid mixing Q2_19 at 5 EUR with Q2_27 at 35 EUR (useless for learning).
    target_median = df[target_col].dropna().median()
    if np.isnan(target_median) or target_median <= 0:
        target_median = 50.0  # safe fallback

    filtered_predecessors = []
    for pc in predecessors:
        pc_median = df[pc].dropna().median()
        if np.isnan(pc_median) or pc_median <= 0:
            continue
        ratio = max(pc_median, target_median) / min(pc_median, target_median)
        if ratio <= 2.5:  # within 2.5× of each other
            filtered_predecessors.append(pc)
        else:
            logger.debug("  Skipping predecessor %s (median %.1f vs target %.1f, ratio %.1f)",
                        pc, pc_median, target_median, ratio)
    predecessors = filtered_predecessors if filtered_predecessors else [target_col]

    base_features = [f for f in feature_cols
                     if "_lag_" not in f and "_momentum_" not in f and "_vol_" not in f]

    all_X = []
    all_y = []
    all_recency = []

    for pred_col in predecessors:
        pred_tag = pred_col  # e.g. omip_q3_24

        # Build lag/momentum/vol features for this predecessor
        pred_feature_map = {}
        for f in feature_cols:
            if f not in base_features:
                # Replace the target contract's lags with this predecessor's
                generic = f.replace(target_col, pred_col)
                if generic in df.columns:
                    pred_feature_map[f] = generic
                else:
                    pred_feature_map[f] = f  # fall back to original

        # Prepare feature matrix for this predecessor (deduplicated)
        use_cols = []
        rename_map = {}
        seen = set()
        for f in feature_cols:
            src = pred_feature_map.get(f, f)
            if src in df.columns and src not in seen:
                use_cols.append(src)
                seen.add(src)
                if src != f:
                    rename_map[src] = f
            elif f in df.columns and f not in seen:
                use_cols.append(f)
                seen.add(f)

        sub = df.copy()
        if horizon_weeks > 0:
            sub["_target"] = sub[pred_col].shift(-horizon_weeks)
        else:
            sub["_target"] = sub[pred_col]

        # Only keep rows where this predecessor has data
        sub = sub.dropna(subset=["_target"])
        if pred_col in sub.columns:
            sub = sub[sub[pred_col].notna()]

        if len(sub) < 5:
            continue

        X_pred = sub[[c for c in use_cols if c in sub.columns]].copy()
        # Rename predecessor-specific columns back to generic names
        X_pred = X_pred.rename(columns=rename_map)
        # Fill any missing columns
        for f in feature_cols:
            if f not in X_pred.columns:
                X_pred[f] = 0.0
        X_pred = X_pred[feature_cols].ffill().fillna(0.0)

        y_pred = sub["_target"]

        # Recency: weight based on row position globally + whether it's the current contract
        n = len(X_pred)
        is_current = (pred_col == target_col)
        base_weight = 1.0 if is_current else 0.6  # predecessors get 60% weight
        # Also apply time-based recency within each contract
        positions = np.arange(n, dtype=float)
        time_decay = np.exp(np.log(2) * (positions - n) / 260)  # ~1yr in biz days
        weights = base_weight * time_decay

        all_X.append(X_pred)
        all_y.append(y_pred)
        all_recency.append(weights)

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=float), np.array([])

    X_stacked = pd.concat(all_X, ignore_index=True)
    y_stacked = pd.concat(all_y, ignore_index=True)
    w_stacked = np.concatenate(all_recency)

    return X_stacked, y_stacked, w_stacked

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_models")
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
# Helpers
# ===================================================================

def _contract_to_column(contract: str) -> str:
    """Map a config contract identifier to its bulletin column name."""
    c = contract.lower()
    if c.startswith("yr"):
        return f"omip_yr_{c[2:]}"
    return f"omip_{c}"


def _get_target_col(contract: str, df: pd.DataFrame) -> str:
    """Return the target column name for a contract."""
    direct = _contract_to_column(contract)
    if direct in df.columns:
        return direct

    candidates = [
        f"omip_{contract.lower()}",
        f"omip_{contract.lower().replace('_', '')}",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    logger.warning("No column found for contract %s (tried %s)", contract, direct)
    return direct


def _get_features(contract: str, df: pd.DataFrame) -> list[str]:
    """Return the feature list for a contract, with per-contract lags."""
    target_col = _contract_to_column(contract)

    if contract in config.SHORT_HORIZON_CONTRACTS:
        base_features = list(config.SHORT_HORIZON_FEATURES)
    else:
        base_features = list(config.LONG_FEATURES)

    replacements = {
        "omip_lag_1d": f"{target_col}_lag_1d",
        "omip_lag_2d": f"{target_col}_lag_2d",
        "omip_lag_3d": f"{target_col}_lag_3d",
        "omip_lag_5d": f"{target_col}_lag_5d",
        "omip_lag_10d": f"{target_col}_lag_10d",
        "omip_lag_15d": f"{target_col}_lag_15d",
        "omip_lag_20d": f"{target_col}_lag_20d",
        "omip_lag_25d": f"{target_col}_lag_25d",
        "omip_lag_30d": f"{target_col}_lag_30d",
        "omip_lag_65d": f"{target_col}_lag_65d",
        "omip_momentum_20d": f"{target_col}_momentum_20d",
        "omip_momentum_65d": f"{target_col}_momentum_65d",
    }

    features = []
    for f in base_features:
        replacement = replacements.get(f)
        if replacement and replacement in df.columns:
            features.append(replacement)
        elif f in df.columns:
            features.append(f)
        else:
            features.append(f)

    # Add per-contract volatility features if available
    for suffix in ["_vol_20d", "_vol_40d"]:
        vol_col = f"{target_col}{suffix}"
        if vol_col in df.columns and vol_col not in features:
            features.append(vol_col)

    return features


def _horizon_to_shift(horizon_days: int) -> int:
    """Convert a horizon in calendar days to a shift in business days.

    Business-daily data: 1 calendar week ≈ 5 business days.
    """
    return max(1, round(horizon_days * 5 / 7))


# Keep legacy alias
_horizon_to_weeks = _horizon_to_shift


def _get_ensemble_weights(contract: str) -> tuple[float, float]:
    """Return (ridge_weight, xgb_weight) for the contract."""
    if contract in config.SHORT_HORIZON_CONTRACTS:
        return config.ENSEMBLE_WEIGHTS_QUARTERLY
    return config.ENSEMBLE_WEIGHTS_ANNUAL


def _build_ridge() -> Pipeline:
    """Create a Ridge regression pipeline with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=config.RIDGE_ALPHA)),
    ])


def _build_xgb(quantile_alpha: float | None = None) -> XGBRegressor:
    """Create an XGBRegressor with config hyperparameters."""
    params = dict(config.XGBOOST_PARAMS)
    es_rounds = params.pop("early_stopping_rounds", 40)
    eval_metric = params.pop("eval_metric", "mae")

    if quantile_alpha is not None:
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = quantile_alpha

    return XGBRegressor(
        **params,
        early_stopping_rounds=es_rounds,
        eval_metric=eval_metric,
    )


def _prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon_weeks: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare X, y, and sample_weight from the master dataset.

    If horizon_weeks > 0, shifts the target forward so that features at
    time t are aligned with the price at time t + horizon_weeks.
    """
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        logger.warning("Missing features (will be zero-filled): %s", missing_features)
        for f in missing_features:
            df[f] = 0.0

    all_feature_cols = available_features + missing_features

    if horizon_weeks > 0:
        df = df.copy()
        df["_target_shifted"] = df[target_col].shift(-horizon_weeks)
        effective_target = "_target_shifted"
    else:
        effective_target = target_col

    subset = df[all_feature_cols + [effective_target, "storm_anomaly_flag"]].copy()
    subset = subset.dropna(subset=[effective_target])

    X = subset[all_feature_cols].ffill().fillna(0.0)
    y = subset[effective_target]
    weights = subset["storm_anomaly_flag"].map(
        {1: config.STORM_ANOMALY_WEIGHT, 0: 1.0}
    ).fillna(1.0)

    return X, y, weights


# ===================================================================
# Walk-forward training + evaluation
# ===================================================================

def _train_single_horizon(
    df: pd.DataFrame,
    contract: str,
    target_col: str,
    feature_cols: list[str],
    horizon_days: int,
    ridge_w: float,
    xgb_w: float,
) -> tuple[dict, list[dict]]:
    """Train ensemble for one contract at one horizon.

    Strategy for quarterly contracts:
      • Walk-forward validation runs on the CURRENT contract's data only,
        so MAE reflects realistic performance on this contract's price regime.
      • Final production model trains on STACKED predecessor data
        (Q3_26 + Q3_25 + … + Q3_18) for better generalization.

    Returns (model_dict, fold_metrics).
    """
    horizon_shift = _horizon_to_shift(horizon_days)
    label = f"{contract}/h{horizon_days}d"

    # --- Always prepare current contract's own data (for walk-forward) ---
    X_own, y_own, weights_own = _prepare_data(df, target_col, feature_cols, horizon_shift)
    n_own = len(X_own)
    recency_halflife = 260  # ~1 year in business days (52 weeks × 5)
    row_positions = np.arange(n_own, dtype=float)
    recency_own = np.exp(np.log(2) * (row_positions - n_own) / recency_halflife)
    combined_own = weights_own.values * recency_own

    logger.info("  %s: %d own rows (target shifted -%d days)", label, n_own, horizon_shift)

    # --- Decide whether to use predecessor stacking for final model ---
    use_stacking = contract in config.SHORT_HORIZON_CONTRACTS
    X_final, y_final, w_final = X_own, y_own, combined_own  # default

    if use_stacking:
        X_stacked, y_stacked, w_stacked = _build_stacked_data(
            df, contract, target_col, feature_cols, horizon_shift,
        )
        if len(X_stacked) > 50:
            X_final, y_final, w_final = X_stacked, y_stacked, w_stacked
            logger.info("  %s: %d stacked rows for final model training",
                        label, len(X_stacked))

    min_rows = 100 if contract in config.SHORT_HORIZON_CONTRACTS else 200
    if n_own < min_rows:
        logger.warning("  %s: only %d rows (need %d) -- skipping.", label, n_own, min_rows)
        return {}, []

    # ── Walk-forward validation on CURRENT CONTRACT data only ──
    # Daily data: need ~100 business days (~5 months) minimum for training
    min_train_size = 500 if contract not in config.SHORT_HORIZON_CONTRACTS else 100
    max_splits = max(2, (n_own - min_train_size) // 50)
    n_splits = min(config.WF_N_SPLITS, max_splits)
    gap = min(config.WF_GAP_DAYS, max(5, n_own // (n_splits * 4)))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X_own)):
        if len(train_idx) < min_train_size:
            logger.debug("  %s fold %d: only %d train rows (need %d) -- skipping",
                        label, fold_i, len(train_idx), min_train_size)
            continue

        X_train, X_test = X_own.iloc[train_idx], X_own.iloc[test_idx]
        y_train, y_test = y_own.iloc[train_idx], y_own.iloc[test_idx]
        w_train = combined_own[train_idx]

        ridge_fold = _build_ridge()
        ridge_fold.fit(X_train, y_train, **{"ridge__sample_weight": w_train})
        ridge_pred_train = ridge_fold.predict(X_train)
        ridge_pred_test = ridge_fold.predict(X_test)

        residuals_train = y_train.values - ridge_pred_train
        xgb_fold = _build_xgb()
        eval_split = max(1, int(len(X_train) * 0.8))
        xgb_fold.fit(
            X_train.iloc[:eval_split],
            residuals_train[:eval_split],
            sample_weight=w_train[:eval_split],
            eval_set=[(X_train.iloc[eval_split:], residuals_train[eval_split:])],
            verbose=False,
        )
        xgb_pred_test = xgb_fold.predict(X_test)

        ensemble_pred = ridge_w * ridge_pred_test + xgb_w * (ridge_pred_test + xgb_pred_test)

        mae = np.mean(np.abs(y_test.values - ensemble_pred))
        rmse = np.sqrt(np.mean((y_test.values - ensemble_pred) ** 2))
        mape = np.mean(np.abs((y_test.values - ensemble_pred) / y_test.values.clip(min=1e-6))) * 100

        if len(y_test) > 1:
            actual_dir = np.sign(np.diff(y_test.values))
            pred_dir = np.sign(np.diff(ensemble_pred))
            dir_acc = np.mean(actual_dir == pred_dir) * 100
        else:
            dir_acc = np.nan

        fold_metrics.append({
            "contract": contract,
            "horizon_days": horizon_days,
            "fold": fold_i,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "train_size": len(X_train),
            "test_size": len(X_test),
        })

    if not fold_metrics:
        logger.warning("  %s: no valid folds — all below min training size.", label)
        return {}, []

    mean_mae = np.mean([m["mae"] for m in fold_metrics])
    logger.info("  %s: mean MAE=%.2f across %d folds", label, mean_mae, len(fold_metrics))

    # ── Final model: train on stacked data (if available) for generalization ──
    ridge_model = _build_ridge()
    ridge_model.fit(X_final, y_final, **{"ridge__sample_weight": w_final})
    ridge_pred_all = ridge_model.predict(X_final)
    residuals_all = y_final.values - ridge_pred_all

    xgb_model = _build_xgb()
    xgb_q10 = _build_xgb(quantile_alpha=config.QUANTILE_LOWER)
    xgb_q90 = _build_xgb(quantile_alpha=config.QUANTILE_UPPER)

    eval_split = max(1, int(len(X_final) * 0.8))
    for model in [xgb_model, xgb_q10, xgb_q90]:
        model.fit(
            X_final.iloc[:eval_split],
            residuals_all[:eval_split],
            sample_weight=w_final[:eval_split],
            eval_set=[(X_final.iloc[eval_split:], residuals_all[eval_split:])],
            verbose=False,
        )

    model_dict = {
        "ridge": ridge_model,
        "xgb": xgb_model,
        "xgb_q10": xgb_q10,
        "xgb_q90": xgb_q90,
    }

    return model_dict, fold_metrics


def train_contract(
    df: pd.DataFrame,
    contract: str,
) -> dict[str, Any]:
    """Train ensemble for a single contract at reliable horizons (7d, 30d).

    Longer horizons (60d, 90d) are derived at forecast time.
    """
    logger.info("Training models for %s ...", contract)

    target_col = _get_target_col(contract, df)
    feature_cols = _get_features(contract, df)
    ridge_w, xgb_w = _get_ensemble_weights(contract)

    if target_col not in df.columns:
        logger.warning("Target column %s not in dataset -- skipping %s.", target_col, contract)
        return {}

    # Only train models for 7d and 30d (reliable horizons)
    horizon_models: dict[int, dict] = {}
    all_fold_metrics: list[dict] = []

    for horizon in _TRAINED_HORIZONS:
        models, metrics = _train_single_horizon(
            df, contract, target_col, feature_cols, horizon, ridge_w, xgb_w,
        )
        if models:
            horizon_models[horizon] = models
        all_fold_metrics.extend(metrics)

    if not horizon_models:
        logger.warning("No horizon models trained for %s -- skipping.", contract)
        return {}

    # Compute historical stats for sanity bounds at forecast time
    price_series = df[target_col].dropna()
    hist_stats = {
        "mean": float(price_series.mean()),
        "std": float(price_series.std()),
        "min": float(price_series.min()),
        "max": float(price_series.max()),
    }

    # Backward compat: top-level models point to best available horizon (7d > 30d)
    compat_horizon = 7 if 7 in horizon_models else min(horizon_models.keys())
    model_bundle = {
        "contract": contract,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "horizon_models": horizon_models,
        "ridge_weight": ridge_w,
        "xgb_weight": xgb_w,
        "fold_metrics": all_fold_metrics,
        "hist_stats": hist_stats,
        "ridge": horizon_models[compat_horizon]["ridge"],
        "xgb": horizon_models[compat_horizon]["xgb"],
        "xgb_q10": horizon_models[compat_horizon]["xgb_q10"],
        "xgb_q90": horizon_models[compat_horizon]["xgb_q90"],
    }

    model_path = config.MODELS_DIR / f"model_{contract}.pkl"
    joblib.dump(model_bundle, model_path)
    logger.info("  Model saved to %s", model_path)

    return model_bundle


# ===================================================================
# Main
# ===================================================================

def train_all() -> None:
    """Train models for all contracts and save walk-forward results."""
    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    df = pd.read_csv(config.MASTER_DATASET, index_col="date", parse_dates=True)
    logger.info("Loaded master dataset: %d rows x %d cols", *df.shape)

    all_metrics: list[dict] = []

    for contract in config.CONTRACTS:
        result = train_contract(df, contract)
        if result and "fold_metrics" in result:
            all_metrics.extend(result["fold_metrics"])

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = config.FORECASTS_DIR / "walkforward_results.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Walk-forward results saved to %s", metrics_path)

        summary = metrics_df.groupby(["contract", "horizon_days"])[
            ["mae", "rmse", "mape", "directional_accuracy"]
        ].mean()
        print("\n" + "=" * 70)
        print("WALK-FORWARD VALIDATION SUMMARY (mean across folds)")
        print("=" * 70)
        print(summary.to_string())
        print()

    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_all()
