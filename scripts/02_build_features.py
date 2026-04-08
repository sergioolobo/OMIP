"""
02_build_features.py — Feature Engineering Module
===================================================
Loads all raw CSVs from data/raw/, merges them into a single weekly
master dataset at Monday frequency, and engineers all features needed
for the forecasting models.

DATA SOURCE MAPPING (adapted):
  • PT generation (wind, solar, hydro gen)  → REN xlsx via generation_pt.csv
  • PT demand                               → REN xlsx via demand_pt.csv
  • ES generation                           → REMOVED
  • ES demand                               → REMOVED
  • Hydro reservoir (ES only)               → embalses.net via hydro_reservoirs.csv
  • German futures                          → investing.com CSV via eex_german_futures.csv
  • PT hydro reservoir                      → DROPPED

Inputs:  data/raw/*.csv
Outputs: data/processed/master_dataset.csv
         data/processed/data_quality_report.txt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("build_features")
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

def _read_raw_csv(path: Path, date_col: str = "date") -> pd.DataFrame:
    """Read a raw CSV (skipping the comment header line) and set DatetimeIndex."""
    if not path.exists():
        logger.warning("File not found: %s — returning empty DataFrame.", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, comment="#", parse_dates=[date_col], index_col=date_col)
        df = df.sort_index()
        logger.info("Loaded %s — %d rows", path.name, len(df))
        return df
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _find_front_contract(df: pd.DataFrame, prefix: str = "omip_q") -> str | None:
    """Find the quarterly or yearly contract column with the most observations."""
    candidates = [c for c in df.columns if c.startswith(prefix)
                  and not c.startswith(prefix + "1") and not c.startswith(prefix + "2")
                  and "lag" not in c and "momentum" not in c
                  and c not in ("omip_q1", "omip_yr1", "omip_yr2")]
    if not candidates:
        return None
    best = max(candidates, key=lambda c: df[c].notna().sum())
    logger.info("Front contract for %s*: %s (%d obs)", prefix, best, df[best].notna().sum())
    return best


def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily DataFrame to weekly (Monday) frequency using mean."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.resample("W-MON").mean()


# ===================================================================
# Load all raw datasets
# ===================================================================

def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load every raw CSV and return a dict of DataFrames."""
    datasets: dict[str, pd.DataFrame] = {}

    datasets["omip"] = _read_raw_csv(config.RAW_FILES["omip_futures"])
    datasets["omie"] = _read_raw_csv(config.RAW_FILES["omie_spot"])
    datasets["ttf"] = _read_raw_csv(config.RAW_FILES["ttf_gas"])
    datasets["co2"] = _read_raw_csv(config.RAW_FILES["eua_co2"])
    datasets["coal"] = _read_raw_csv(config.RAW_FILES["api2_coal"])
    datasets["hydro"] = _read_raw_csv(config.RAW_FILES["hydro_reservoirs"])
    datasets["gen_pt"] = _read_raw_csv(config.RAW_FILES["generation_pt"])
    datasets["eex"] = _read_raw_csv(config.RAW_FILES["eex_german_futures"])
    datasets["eurusd"] = _read_raw_csv(config.RAW_FILES["eurusd"])
    datasets["demand_pt"] = _read_raw_csv(config.RAW_FILES["demand_pt"])
    datasets["news"] = _read_raw_csv(config.RAW_FILES["news_sentiment"])

    return datasets


# ===================================================================
# Merge into weekly master
# ===================================================================

def merge_to_weekly(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all datasets into a single weekly (Monday) master DataFrame."""
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq="W-MON")
    master = pd.DataFrame(index=idx)
    master.index.name = "date"

    # --- OMIP futures ---
    # The bulletin parser already produces columns like omip_YR_27, omip_Q3_26.
    # Other sources may use long-form (settlement_price + contract_name).
    omip = datasets["omip"]
    if not omip.empty:
        if "settlement_price" in omip.columns and "contract_name" in omip.columns:
            omip_wide = omip.pivot_table(
                index=omip.index, columns="contract_name", values="settlement_price"
            )
            omip_wide = _resample_to_weekly(omip_wide)
            omip_wide.columns = [
                "omip_" + str(c).lower().replace(" ", "_").replace("/", "_")
                for c in omip_wide.columns
            ]
            master = master.join(omip_wide, how="left")
        else:
            omip_w = _resample_to_weekly(omip)
            # Normalise column names to lowercase; avoid double omip_ prefix
            new_cols = []
            for c in omip_w.columns:
                c_low = str(c).lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                if not c_low.startswith("omip_"):
                    c_low = "omip_" + c_low
                new_cols.append(c_low)
            omip_w.columns = new_cols
            master = master.join(omip_w, how="left")

    # Log which OMIP columns we have for debugging
    omip_cols = [c for c in master.columns if c.startswith("omip_")]
    logger.info("OMIP columns available: %s", omip_cols[:15])

    # Ensure essential OMIP proxy columns exist.
    # omip_yr1 = front annual contract, omip_yr2 = second annual,
    # omip_q1 = front quarterly contract.
    # The bulletin columns use 2-digit years: omip_yr_27, omip_q3_26, etc.
    for col in ["omip_yr1", "omip_yr2", "omip_q1"]:
        if col not in master.columns:
            mapping = {
                "omip_yr1": ["omip_yr_27", "omip_yr27", "omip_yr_26", "omip_yr26"],
                "omip_yr2": ["omip_yr_28", "omip_yr28", "omip_yr_27", "omip_yr27"],
                "omip_q1":  ["omip_q1_27", "omip_q1_26", "omip_q4_26",
                             "omip_q3_26", "omip_q2_26"],
            }
            found = False
            for alt in mapping.get(col, []):
                if alt in master.columns and master[alt].notna().any():
                    master[col] = master[alt]
                    logger.info("Mapped %s → %s", col, alt)
                    found = True
                    break
            if not found:
                master[col] = np.nan
                logger.warning("No suitable column found for %s", col)

    # --- Fuel prices (daily → weekly resample) ---
    for key, col_name in [("ttf", "ttf_gas"), ("co2", "eua_co2"), ("coal", "api2_coal")]:
        df = datasets[key]
        if not df.empty:
            df_w = _resample_to_weekly(df)
            if col_name in df_w.columns:
                master[col_name] = df_w[col_name].reindex(master.index)
            elif not df_w.empty:
                master[col_name] = df_w.iloc[:, 0].reindex(master.index)
            else:
                master[col_name] = np.nan
        else:
            master[col_name] = np.nan

    # --- OMIE spot ---
    omie = datasets["omie"]
    if not omie.empty:
        omie_w = _resample_to_weekly(omie)
        col = "omie_spot" if "omie_spot" in omie_w.columns else omie_w.columns[0]
        master["omie_spot"] = omie_w[col].reindex(master.index)
    else:
        master["omie_spot"] = np.nan

    # --- Hydro (Spain only — from embalses.net) ---
    hydro = datasets["hydro"]
    if not hydro.empty:
        for c in ["hydro_es", "hydro_iberia"]:
            if c in hydro.columns:
                master[c] = hydro[c].reindex(master.index, method="ffill")
    # hydro_iberia = hydro_es (Spain only, no PT)
    if "hydro_es" in master.columns and "hydro_iberia" not in master.columns:
        master["hydro_iberia"] = master["hydro_es"]
    for c in ["hydro_es", "hydro_iberia"]:
        if c not in master.columns:
            master[c] = np.nan

    # --- Wind + Solar generation (PT from REN) ---
    gen_pt = datasets["gen_pt"]
    if not gen_pt.empty:
        for c in ["wind_pt", "solar_pt", "hydro_gen_pt"]:
            if c in gen_pt.columns:
                master[c] = gen_pt[c].reindex(master.index)

    for c in ["wind_pt", "solar_pt"]:
        if c not in master.columns:
            master[c] = np.nan

    # --- EEX German futures (from investing.com CSV) ---
    eex = datasets["eex"]
    if not eex.empty:
        eex_w = _resample_to_weekly(eex)
        col = "german_cal_futures" if "german_cal_futures" in eex_w.columns else eex_w.columns[0]
        master["german_cal_futures"] = eex_w[col].reindex(master.index)
    else:
        master["german_cal_futures"] = np.nan

    # --- EUR/USD ---
    eurusd = datasets["eurusd"]
    if not eurusd.empty:
        eurusd_w = _resample_to_weekly(eurusd)
        col = "eurusd" if "eurusd" in eurusd_w.columns else eurusd_w.columns[0]
        master["eurusd"] = eurusd_w[col].reindex(master.index)
    else:
        master["eurusd"] = np.nan

    # --- Demand (PT from REN) ---
    demand_pt = datasets["demand_pt"]
    if not demand_pt.empty:
        col = "demand_pt" if "demand_pt" in demand_pt.columns else demand_pt.columns[0]
        master["demand_pt"] = demand_pt[col].reindex(master.index)
    else:
        master["demand_pt"] = np.nan

    # --- News sentiment (weekly, from NewsAPI + VADER) ---
    news = datasets.get("news", pd.DataFrame())
    if not news.empty:
        news_cols = ["news_sentiment", "news_volume", "news_bullish_pct",
                     "news_bearish_pct", "news_max_pos", "news_max_neg"]
        for c in news_cols:
            if c in news.columns:
                master[c] = news[c].reindex(master.index)
        logger.info("News sentiment: %d non-null weeks",
                    master["news_sentiment"].notna().sum() if "news_sentiment" in master.columns else 0)
    else:
        for c in ["news_sentiment", "news_volume", "news_bullish_pct",
                   "news_bearish_pct", "news_max_pos", "news_max_neg"]:
            master[c] = np.nan
        logger.info("No news sentiment data available.")

    logger.info("Merged master dataset: %d rows x %d cols", *master.shape)
    return master


# ===================================================================
# Feature engineering
# ===================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features to the master DataFrame."""

    # 1. Fuel marginal cost proxies
    df["gas_spark_spread"] = (
        (df["ttf_gas"] / config.TTF_MWH_FACTOR) / config.CCGT_EFFICIENCY
        + df["eua_co2"] * config.CCGT_EMISSION_FACTOR
    )
    df["coal_dark_spread"] = (
        (df["api2_coal"] / config.COAL_MWH_FACTOR) / config.COAL_EFFICIENCY
        + df["eua_co2"] * config.COAL_EMISSION_FACTOR
    )
    df["marginal_cost_proxy"] = df[["gas_spark_spread", "coal_dark_spread"]].max(axis=1)

    # 2. Hydro anomaly (deviation from seasonal average)
    #    Uses hydro_iberia which is now Spain-only (embalses.net)
    week_num = df.index.isocalendar().week.astype(int)
    df["hydro_seasonal_avg"] = df.groupby(week_num)["hydro_iberia"].transform("mean")
    df["hydro_anomaly"] = df["hydro_iberia"] - df["hydro_seasonal_avg"]
    df["hydro_anomaly_pct"] = df["hydro_anomaly"] / df["hydro_seasonal_avg"].replace(0, np.nan)

    # 3. Renewable surplus index (PT only)
    df["res_generation"] = (
        df["wind_pt"].fillna(0) + df["solar_pt"].fillna(0)
    )
    df["res_penetration"] = df["res_generation"] / df["demand_pt"].replace(0, np.nan)

    # 4. OMIE risk premium (spot vs. futures spread)
    df["omie_spot_monthly_avg"] = df["omie_spot"].rolling(4, min_periods=1).mean()
    # Compute risk premium for the front quarterly contract (pick most populated)
    front_q_col = _find_front_contract(df, prefix="omip_q")
    if front_q_col:
        df["risk_premium"] = df[front_q_col] - df["omie_spot_monthly_avg"]
    else:
        df["risk_premium"] = np.nan

    # 5. Forward curve slope (YR front vs YR second)
    yr_cols = sorted([c for c in df.columns if c.startswith("omip_yr_")
                      and df[c].notna().sum() > 50])
    if len(yr_cols) >= 2:
        df["curve_slope_yr"] = df[yr_cols[0]] - df[yr_cols[1]]
    else:
        df["curve_slope_yr"] = np.nan

    # 6. News sentiment derived features
    if "news_sentiment" in df.columns:
        df["news_sentiment_ma4w"] = df["news_sentiment"].rolling(4, min_periods=1).mean()
        df["news_sentiment_shift"] = df["news_sentiment"] - df["news_sentiment"].shift(1)
    else:
        df["news_sentiment_ma4w"] = np.nan
        df["news_sentiment_shift"] = np.nan

    # 7. Per-contract autoregressive lags, momentum, and technical features.
    #    For each OMIP contract column, generate lags and momentum so each
    #    contract model can use features derived from its own price history.
    contract_cols = [c for c in df.columns
                     if c.startswith("omip_") and not c.startswith("omip_m_")
                     and c not in ("omip_yr1", "omip_yr2", "omip_q1")
                     and "lag" not in c and "momentum" not in c]
    for col in contract_cols:
        tag = col.replace("omip_", "")  # e.g. "yr_27", "q3_26"
        for lag in [1, 2, 3, 4, 5, 6, 7, 13]:
            df[f"{col}_lag_{lag}w"] = df[col].shift(lag)
        df[f"{col}_momentum_4w"] = df[col] - df[col].shift(4)
        df[f"{col}_momentum_13w"] = df[col] - df[col].shift(13)

    # Also keep generic proxy lags for backward compatibility
    if "omip_yr1" in df.columns:
        for lag in [1, 2, 3, 4, 5, 6, 7, 13]:
            df[f"omip_lag_{lag}w"] = df["omip_yr1"].shift(lag)
        df["omip_momentum_4w"] = df["omip_yr1"] - df["omip_yr1"].shift(4)
        df["omip_momentum_13w"] = df["omip_yr1"] - df["omip_yr1"].shift(13)

    # 7b. Cross-contract term structure features
    #     Spreads between quarterly and annual contracts capture curve shape
    q_cols = sorted([c for c in df.columns if c.startswith("omip_q") and "_lag_" not in c
                     and "_momentum_" not in c and c not in ("omip_q1",)])
    if len(q_cols) >= 2:
        df["term_q_spread_1_2"] = df[q_cols[0]] - df[q_cols[1]]
        logger.info("Term structure: %s − %s", q_cols[0], q_cols[1])
    if len(q_cols) >= 3:
        df["term_q_spread_1_3"] = df[q_cols[0]] - df[q_cols[2]]
    # Annual spread (YR27 - YR28 if available)
    yr_sorted = sorted([c for c in df.columns if c.startswith("omip_yr_")
                        and "_lag_" not in c and "_momentum_" not in c])
    if len(yr_sorted) >= 2:
        df["term_yr_spread"] = df[yr_sorted[0]] - df[yr_sorted[1]]
    # Quarterly vs annual spread (front Q vs front YR)
    if q_cols and yr_sorted:
        df["term_q_yr_spread"] = df[q_cols[0]] - df[yr_sorted[0]]

    # 7c. Rolling volatility features (OMIP contract own-price volatility)
    for col in contract_cols:
        tag = col.replace("omip_", "")
        df[f"{col}_vol_4w"] = df[col].rolling(4, min_periods=2).std()
        df[f"{col}_vol_8w"] = df[col].rolling(8, min_periods=4).std()

    # 7d. Spot price momentum (OMIE spot trends for short-horizon contracts)
    if "omie_spot" in df.columns:
        df["spot_momentum_4w"] = df["omie_spot"] - df["omie_spot"].shift(4)
        df["spot_momentum_8w"] = df["omie_spot"] - df["omie_spot"].shift(8)
        df["spot_ma_ratio"] = (
            df["omie_spot"].rolling(4, min_periods=1).mean()
            / df["omie_spot"].rolling(13, min_periods=1).mean()
        )

    # 7e. German futures momentum & spread vs OMIP
    if "german_cal_futures" in df.columns:
        df["german_momentum_4w"] = df["german_cal_futures"] - df["german_cal_futures"].shift(4)
        df["german_momentum_13w"] = df["german_cal_futures"] - df["german_cal_futures"].shift(13)
        if yr_sorted:
            df["german_omip_spread"] = df["german_cal_futures"] - df[yr_sorted[0]]

    # 7. Rolling statistics
    df["ttf_vol_4w"] = df["ttf_gas"].rolling(4).std()
    df["ttf_vol_13w"] = df["ttf_gas"].rolling(13).std()

    # 8. Seasonality dummies
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["is_q1"] = (df["quarter"] == 1).astype(int)
    df["is_q3"] = (df["quarter"] == 3).astype(int)

    # 9. Storm anomaly flag
    df["storm_anomaly_flag"] = 0
    storm_mask = (df.index >= config.STORM_ANOMALY_START) & (df.index <= config.STORM_ANOMALY_END)
    df.loc[storm_mask, "storm_anomaly_flag"] = 1

    # Also flag weeks where OMIE spot is below 2nd percentile
    if df["omie_spot"].notna().sum() > 0:
        threshold = df["omie_spot"].quantile(config.SPOT_ANOMALY_PERCENTILE)
        spot_anomaly_mask = df["omie_spot"] < threshold
        df.loc[spot_anomaly_mask, "storm_anomaly_flag"] = 1

    logger.info("Engineered %d features.", df.shape[1])
    return df


# ===================================================================
# Data quality report
# ===================================================================

def write_quality_report(df: pd.DataFrame, path: Path) -> None:
    """Write a data quality report showing % missing per column."""
    total = len(df)
    lines = [
        "DATA QUALITY REPORT",
        "=" * 60,
        f"Total rows: {total}",
        f"Date range: {df.index.min()} to {df.index.max()}",
        "",
        f"{'Column':<35} {'Missing':>8} {'%':>8}",
        "-" * 55,
    ]
    for col in sorted(df.columns):
        n_missing = df[col].isna().sum()
        pct = 100.0 * n_missing / total if total > 0 else 0.0
        lines.append(f"{col:<35} {n_missing:>8d} {pct:>7.1f}%")

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    logger.info("Data quality report saved to %s", path)
    print(report)


# ===================================================================
# Main
# ===================================================================

def build_features() -> pd.DataFrame:
    """Full pipeline: load raw → merge → engineer features → save."""
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE DATASET")
    logger.info("=" * 60)

    datasets = load_all_raw()
    master = merge_to_weekly(datasets)
    master = engineer_features(master)

    # Initialize ECT term column (will be populated by 03_cointegration_check)
    if "ect_term" not in master.columns:
        master["ect_term"] = np.nan

    # Save
    master.to_csv(config.MASTER_DATASET)
    logger.info("Master dataset saved to %s", config.MASTER_DATASET)

    write_quality_report(master, config.DATA_QUALITY_REPORT)

    logger.info("=" * 60)
    logger.info("FEATURE BUILD COMPLETE — %d rows × %d cols", *master.shape)
    logger.info("=" * 60)

    return master


if __name__ == "__main__":
    build_features()
