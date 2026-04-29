"""
05_forecast.py — 7-Day Hourly Forecast
=======================================
Generates a 168-hour (7-day) recursive multi-step forecast using the 24
trained LASSO models.  At each step, previously forecasted values fill the
lag slots that would otherwise require future prices.

80% prediction intervals are constructed from historical residuals computed
during walk-forward evaluation (outputs/forecasts/eval_residuals.csv).

Inputs:  data/processed/master_hourly.csv
         models/hour_{HH}.pkl           (×24)
         outputs/forecasts/eval_residuals.csv
Outputs: outputs/forecasts/omie_forecast_{date}.csv
         outputs/forecasts/omie_forecast_latest.csv  (always overwritten)
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("forecast")
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
# Helpers
# ===================================================================

def _load_models() -> dict[int, dict]:
    models: dict[int, dict] = {}
    for h in config.HOURS:
        p = config.MODELS_DIR / f"hour_{h:02d}.pkl"
        if p.exists():
            models[h] = joblib.load(p)
    logger.info("Loaded %d/24 hourly models", len(models))
    return models


def _load_boosters() -> dict[int, dict]:
    """Load LightGBM residual boosters (03b) if present. Optional — if the
    file is missing for an hour, the pipeline falls back to LASSO only."""
    boosters: dict[int, dict] = {}
    for h in config.HOURS:
        p = config.MODELS_DIR / f"hour_{h:02d}_booster.pkl"
        if p.exists():
            boosters[h] = joblib.load(p)
    if boosters:
        logger.info("Loaded %d/24 residual boosters (LightGBM)", len(boosters))
    else:
        logger.info("No residual boosters found — using LASSO only")
    return boosters


def _winsorize_row(X: np.ndarray, bounds: dict, feats: list[str]) -> np.ndarray:
    """Clip each feature in a (1, n_features) row to its (lo, hi) bound."""
    if not bounds:
        return X
    X = X.copy()
    for i, f in enumerate(feats):
        b = bounds.get(f)
        if b is None:
            continue
        lo, hi = b
        X[0, i] = float(np.clip(X[0, i], lo, hi))
    return X


def _load_residuals() -> dict[int, np.ndarray]:
    """Load bootstrap residuals from walk-forward evaluation per hour."""
    path = config.FORECASTS_DIR / "eval_residuals.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(col[1:]): df[col].dropna().values
            for col in df.columns if col.startswith("h")}


def _calendar_features(ts: pd.Timestamp, h: int) -> dict:
    dow  = ts.dayofweek
    mon  = ts.month
    return {
        "hour_of_day":   h,
        "day_of_week":   dow,
        "month":         mon,
        "is_weekend":    int(dow >= 5),
        "is_holiday":    int(ts.date() in config.IBERIAN_HOLIDAYS),
        "is_summer":     int(mon in [7, 8, 9]),
        "is_winter":     int(mon in [12, 1, 2]),
        "hour_x_weekend": h * int(dow >= 5),
    }


# ===================================================================
# Forecast
# ===================================================================

def forecast(forecast_days: int = config.FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """
    Generate a recursive multi-step 7-day forecast.

    For each future (date, hour):
      - Lags inside historical window → use actual prices.
      - Lags that fall in the future → use the model's own previous forecasts.
      - Fundamentals → forward-filled from last known value.
      - Prediction intervals → 10th/90th percentile of walk-forward residuals.

    Returns:
        DataFrame with columns [datetime, hour, point_forecast,
                                 lower_80ci, upper_80ci, lag_data_quality].
    """
    logger.info("=" * 60)
    logger.info("GENERATING %d-DAY HOURLY FORECAST", forecast_days)
    logger.info("=" * 60)

    master_path = config.DATA_PROCESSED / "master_hourly.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} not found — run 02_build_features.py first")

    hist = pd.read_csv(master_path, index_col="datetime", parse_dates=True)
    hist = hist.sort_index()

    models    = _load_models()
    boosters  = _load_boosters()
    residuals = _load_residuals()
    if not models:
        raise RuntimeError("No hourly models found — run 03_train_models.py first")

    # Last actual price timestamp
    actual_prices = hist["price_es"].dropna().copy()
    last_ts = actual_prices.index.max()
    logger.info("Last known price: %s  (%.2f €/MWh)",
                last_ts, actual_prices.iloc[-1])

    # Build a mutable price series combining history + forecasts
    # (used only for AR lags, NOT for rolling stats)
    price_series = actual_prices.copy()

    # Pre-compute rolling stats from ACTUAL prices only — frozen at forecast origin.
    # Rationale: rolling stats should reflect the market regime at forecast time,
    # not drift upward/downward as recursive forecasts accumulate.
    _r24  = actual_prices.iloc[-24:]
    _r168 = actual_prices.iloc[-168:]
    _frozen_roll = {
        "price_roll_mean_24h":  float(_r24.mean())  if len(_r24)  else 0.0,
        "price_roll_std_24h":   float(_r24.std())   if len(_r24)  > 1 else 0.0,
        "price_roll_mean_168h": float(_r168.mean()) if len(_r168) else 0.0,
        "price_roll_std_168h":  float(_r168.std())  if len(_r168) > 1 else 0.0,
    }
    logger.info("Frozen rolling stats: mean_24h=%.1f  std_24h=%.1f  "
                "mean_168h=%.1f  std_168h=%.1f",
                _frozen_roll["price_roll_mean_24h"],
                _frozen_roll["price_roll_std_24h"],
                _frozen_roll["price_roll_mean_168h"],
                _frozen_roll["price_roll_std_168h"])

    # Carry-forward fundamentals from last known row.  Includes the new
    # residual-load family so the forecasted hours don't lose those features.
    fund_cols = [c for c in hist.columns if c in (
        config.FUNDAMENTAL_FEATURES
        + config.ENTSOE_FEATURES
        + config.RESIDUAL_LOAD_FEATURES
        + ["spark_spread"]
    )]
    last_fund_row = hist[fund_cols].dropna(how="all").iloc[-1]

    # Forecast rows
    rows: list[dict] = []

    for step in range(1, forecast_days * 24 + 1):
        ts  = last_ts + pd.Timedelta(hours=step)
        h   = ts.hour
        bundle = models.get(h)
        if bundle is None:
            continue
        feats       = bundle["all_features"]
        pipeline    = bundle["pipeline"]
        bounds      = bundle.get("feature_bounds", {})
        target_mode = bundle.get("target_mode", "absolute")
        booster_bd  = boosters.get(h)

        # Build feature vector
        row: dict[str, float] = {}

        # --- AR lags (mix of actual + previously-forecasted) ---
        lag_available = 0
        for lag_h in config.LAG_HOURS:
            lag_ts = ts - pd.Timedelta(hours=lag_h)
            val = price_series.get(lag_ts, np.nan)
            row[f"price_lag_{lag_h}h"] = val
            if not np.isnan(val):
                lag_available += 1

        lag_quality = lag_available / len(config.LAG_HOURS)

        # Rolling stats — use frozen actual-only values to prevent recursive drift
        row.update(_frozen_roll)

        # Previous-day full curve
        prev_day = ts.normalize() - pd.Timedelta(days=1)
        for ph in range(24):
            lag_ts_pd = prev_day + pd.Timedelta(hours=ph)
            row[f"prev_day_h{ph:02d}"] = float(
                price_series.get(lag_ts_pd, np.nan) or np.nan
            )

        # Calendar
        row.update(_calendar_features(ts, h))

        # Fundamentals (carried forward)
        for col in fund_cols:
            row[col] = float(last_fund_row.get(col, np.nan) or np.nan)

        # Build array in model's expected feature order
        X = np.array([[row.get(f, np.nan) for f in feats]], dtype=float)
        # Impute any remaining NaN with 0 (scaler will normalise)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Option 4 — winsorize features to training-time 1st/99th percentile
        # bounds. Prevents LASSO from extrapolating linearly when a raw
        # feature lands outside its training distribution.
        X_w = _winsorize_row(X, bounds, feats)

        try:
            lasso_pred = float(pipeline.predict(X_w)[0])
        except Exception as exc:
            logger.debug("H%02d LASSO predict failed: %s", h, exc)
            lasso_pred = np.nan

        # Option 1 — add LightGBM residual correction if available. The
        # correction is bounded to ±2σ of the training residuals so a wild
        # tree leaf can't overwhelm LASSO's signal.
        correction = 0.0
        if booster_bd is not None and not np.isnan(lasso_pred):
            try:
                raw = float(booster_bd["booster"].predict(X_w)[0])
                res_std = float(booster_bd.get("residual_std", 10.0))
                correction = float(np.clip(raw, -2 * res_std, 2 * res_std))
            except Exception as exc:
                logger.debug("H%02d booster predict failed: %s", h, exc)
                correction = 0.0

        # Combine LASSO + booster correction.
        if np.isnan(lasso_pred):
            point = np.nan
        else:
            point = lasso_pred + correction
            # Step 2a — if the model was trained on (price - rolling_mean),
            # add the rolling mean back so `point` is in absolute €/MWh.
            if target_mode == "deviation_from_roll_mean_24h":
                point += _frozen_roll["price_roll_mean_24h"]

        # Prediction interval from residual bootstrap
        res = residuals.get(h, np.array([]))
        if len(res) > 10:
            lower = point + float(np.percentile(res, 10))
            upper = point + float(np.percentile(res, 90))
        else:
            lower = point * 0.85
            upper = point * 1.15

        # Clip to historically realistic bounds. OMIE PT/ES has never
        # traded below -9.83 €/MWh (72,448 hour history, 2015-2026) and
        # extreme peaks are around 650 €/MWh. LASSO can extrapolate
        # wildly when feature combinations fall outside training
        # distribution — a floor/cap is standard practice in EPF.
        # Tight business-realistic bounds. OMIE PT/ES has touched -9.83
        # once in 72,448 historical hours (2015-2026); clipping at -5 only
        # affects a handful of deeply-negative outliers. 300 €/MWh cap
        # covers 99.9 % of historical peaks (max was 651, single 2022 day).
        PRICE_FLOOR = -5.0    # €/MWh
        PRICE_CAP   = 300.0   # €/MWh
        if not np.isnan(point):
            point = float(np.clip(point, PRICE_FLOOR, PRICE_CAP))
        lower = float(np.clip(lower, PRICE_FLOOR, PRICE_CAP))
        upper = float(np.clip(upper, PRICE_FLOOR, PRICE_CAP))

        rows.append({
            "datetime":          ts,
            "hour":              h,
            "point_forecast":    round(point, 3) if not np.isnan(point) else np.nan,
            "lower_80ci":        round(lower, 3),
            "upper_80ci":        round(upper, 3),
            "lag_data_quality":  round(lag_quality * 100, 1),
        })

        # Feed forecast back into price_series for subsequent lags.
        # Using the *clipped* value prevents one wildly-negative hour
        # from dragging down downstream hours via AR lag features.
        if not np.isnan(point):
            price_series[ts] = point

    fc_df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    # Save
    today_str = date.today().isoformat()
    dated_path  = config.FORECASTS_DIR / f"omie_forecast_{today_str}.csv"
    latest_path = config.FORECASTS_DIR / "omie_forecast_latest.csv"
    fc_df.to_csv(dated_path,  index=False)
    fc_df.to_csv(latest_path, index=False)

    # Console summary
    _print_summary(fc_df)

    logger.info("Saved %d hourly forecasts → %s", len(fc_df), dated_path.name)
    logger.info("Horizon: %s → %s", fc_df["datetime"].iloc[0], fc_df["datetime"].iloc[-1])
    logger.info("=" * 60)
    return fc_df


def _print_summary(fc: pd.DataFrame) -> None:
    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["datetime"]).dt.normalize()
    grp = fc.groupby("date")

    def _min_hour_row(g: pd.DataFrame) -> str:
        idx = g["point_forecast"].idxmin()
        return f"H{g.loc[idx, 'hour']:02d} {g.loc[idx, 'point_forecast']:.1f}"

    def _max_hour_row(g: pd.DataFrame) -> str:
        idx = g["point_forecast"].idxmax()
        return f"H{g.loc[idx, 'hour']:02d} {g.loc[idx, 'point_forecast']:.1f}"

    print("\n" + "=" * 80)
    print(f"{'Date':<12}| {'Avg':>10} | {'Min Hour':>12} | {'Max Hour':>12} "
          f"| {'Lower CI':>10} | {'Upper CI':>10}")
    print("-" * 80)
    for day, g in grp:
        avg  = g["point_forecast"].mean()
        low  = g["lower_80ci"].mean()
        high = g["upper_80ci"].mean()
        print(f"  {str(day.date()):<10}| {avg:>8.1f}  | {_min_hour_row(g):>12} "
              f"| {_max_hour_row(g):>12} | {low:>10.1f} | {high:>10.1f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    forecast()
