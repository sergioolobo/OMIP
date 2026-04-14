"""
02_build_features.py — Feature Engineering Module
===================================================
Loads all raw CSVs from data/raw/, merges them into a single
**business-daily** master dataset and engineers all features needed
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
    """Resample a daily DataFrame to weekly (Monday) frequency using mean.
    LEGACY — only used when DATA_FREQ == 'W-MON'.
    """
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.resample("W-MON").mean()


def _ensure_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DataFrame has a DatetimeIndex suitable for daily reindexing."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


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

def merge_to_master(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all datasets into a single business-daily master DataFrame.

    All daily sources are joined directly; weekly/lower-frequency sources
    are forward-filled so every business day has a value.
    """
    freq = config.DATA_FREQ  # "B" (business-daily)
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq=freq)
    master = pd.DataFrame(index=idx)
    master.index.name = "date"

    # --- Helper: join a daily-frequency DataFrame to master (ffill gaps) ---
    def _join_daily(src: pd.DataFrame, cols: list[str] | None = None) -> None:
        nonlocal master
        if src.empty:
            return
        src = _ensure_daily_index(src)
        if cols:
            for c in cols:
                if c in src.columns:
                    master[c] = src[c].reindex(master.index, method="ffill")
        else:
            reindexed = src.reindex(master.index, method="ffill")
            master = master.join(reindexed, how="left", rsuffix="_dup")
            # Drop duplicate columns
            master = master[[c for c in master.columns if not c.endswith("_dup")]]

    # --- OMIP futures (business-daily from bulletins) ---
    omip = datasets["omip"]
    if not omip.empty:
        omip = _ensure_daily_index(omip)
        if "settlement_price" in omip.columns and "contract_name" in omip.columns:
            omip_wide = omip.pivot_table(
                index=omip.index, columns="contract_name", values="settlement_price"
            )
            omip_wide.columns = [
                "omip_" + str(c).lower().replace(" ", "_").replace("/", "_")
                for c in omip_wide.columns
            ]
            # Join daily OMIP directly (no resampling)
            master = master.join(omip_wide.reindex(master.index), how="left")
        else:
            # Normalise column names to lowercase; avoid double omip_ prefix
            new_cols = []
            for c in omip.columns:
                c_low = str(c).lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                if not c_low.startswith("omip_"):
                    c_low = "omip_" + c_low
                new_cols.append(c_low)
            omip.columns = new_cols
            master = master.join(omip.reindex(master.index), how="left")

    # Log which OMIP columns we have for debugging
    omip_cols = [c for c in master.columns if c.startswith("omip_")]
    logger.info("OMIP columns available (%d): %s", len(omip_cols), omip_cols[:15])

    # Ensure essential OMIP proxy columns exist.
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

    # --- Fuel prices (daily, joined directly with ffill for gaps) ---
    for key, col_name in [("ttf", "ttf_gas"), ("co2", "eua_co2"), ("coal", "api2_coal")]:
        df = datasets[key]
        if not df.empty:
            df = _ensure_daily_index(df)
            if col_name in df.columns:
                master[col_name] = df[col_name].reindex(master.index, method="ffill")
            elif not df.empty:
                master[col_name] = df.iloc[:, 0].reindex(master.index, method="ffill")
            else:
                master[col_name] = np.nan
        else:
            master[col_name] = np.nan

    # --- OMIE spot (daily) ---
    omie = datasets["omie"]
    if not omie.empty:
        omie = _ensure_daily_index(omie)
        col = "omie_spot" if "omie_spot" in omie.columns else omie.columns[0]
        master["omie_spot"] = omie[col].reindex(master.index, method="ffill")
    else:
        master["omie_spot"] = np.nan

    # --- Hydro (Spain only — weekly from embalses.net, forward-filled) ---
    hydro = datasets["hydro"]
    if not hydro.empty:
        hydro = _ensure_daily_index(hydro)
        for c in ["hydro_es", "hydro_iberia"]:
            if c in hydro.columns:
                master[c] = hydro[c].reindex(master.index, method="ffill")
    if "hydro_es" in master.columns and "hydro_iberia" not in master.columns:
        master["hydro_iberia"] = master["hydro_es"]
    for c in ["hydro_es", "hydro_iberia"]:
        if c not in master.columns:
            master[c] = np.nan

    # --- PT generation mix (from REN Production Breakdown, daily/sub-daily → ffill) ---
    gen_pt = datasets["gen_pt"]
    if not gen_pt.empty:
        gen_pt = _ensure_daily_index(gen_pt)
        ren_cols = [
            "wind_pt", "solar_pt", "hydro_gen_pt",
            "biomass_pt", "gas_ccgt_pt", "gas_cogen_pt", "coal_gen_pt",
            "imports_pt", "exports_pt", "pumping_pt", "demand_pt",
        ]
        for c in ren_cols:
            if c in gen_pt.columns:
                master[c] = gen_pt[c].reindex(master.index, method="ffill")

    for c in ["wind_pt", "solar_pt"]:
        if c not in master.columns:
            master[c] = np.nan

    # --- EEX German futures (daily from investing.com CSV) ---
    eex = datasets["eex"]
    if not eex.empty:
        eex = _ensure_daily_index(eex)
        col = "german_cal_futures" if "german_cal_futures" in eex.columns else eex.columns[0]
        master["german_cal_futures"] = eex[col].reindex(master.index, method="ffill")
    else:
        master["german_cal_futures"] = np.nan

    # --- EUR/USD (daily) ---
    eurusd = datasets["eurusd"]
    if not eurusd.empty:
        eurusd = _ensure_daily_index(eurusd)
        col = "eurusd" if "eurusd" in eurusd.columns else eurusd.columns[0]
        master["eurusd"] = eurusd[col].reindex(master.index, method="ffill")
    else:
        master["eurusd"] = np.nan

    # --- Demand (PT from REN) ---
    demand_pt = datasets["demand_pt"]
    if not demand_pt.empty:
        demand_pt = _ensure_daily_index(demand_pt)
        col = "demand_pt" if "demand_pt" in demand_pt.columns else demand_pt.columns[0]
        master["demand_pt"] = demand_pt[col].reindex(master.index, method="ffill")
    else:
        master["demand_pt"] = np.nan

    # --- News sentiment (weekly, forward-filled to daily) ---
    news = datasets.get("news", pd.DataFrame())
    if not news.empty:
        news = _ensure_daily_index(news)
        news_cols = ["news_sentiment", "news_volume", "news_bullish_pct",
                     "news_bearish_pct", "news_max_pos", "news_max_neg"]
        for c in news_cols:
            if c in news.columns:
                master[c] = news[c].reindex(master.index, method="ffill")
        logger.info("News sentiment: %d non-null rows",
                    master["news_sentiment"].notna().sum() if "news_sentiment" in master.columns else 0)
    else:
        for c in ["news_sentiment", "news_volume", "news_bullish_pct",
                   "news_bearish_pct", "news_max_pos", "news_max_neg"]:
            master[c] = np.nan
        logger.info("No news sentiment data available.")

    logger.info("Merged master dataset: %d rows x %d cols", *master.shape)
    return master


# Keep legacy alias
merge_to_weekly = merge_to_master


# ===================================================================
# Feature engineering
# ===================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features to the master DataFrame.

    All rolling windows and lags use **business-day** counts:
      1 week  ≈  5 days      4 weeks ≈ 20 days
      2 weeks ≈ 10 days     13 weeks ≈ 65 days
      8 weeks ≈ 40 days
    """

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
    #    Uses day-of-year grouping for daily frequency
    doy = df.index.dayofyear
    df["hydro_seasonal_avg"] = df.groupby(doy)["hydro_iberia"].transform("mean")
    df["hydro_anomaly"] = df["hydro_iberia"] - df["hydro_seasonal_avg"]
    df["hydro_anomaly_pct"] = df["hydro_anomaly"] / df["hydro_seasonal_avg"].replace(0, np.nan)

    # 3. Renewable surplus index (PT only)
    df["res_generation"] = (
        df["wind_pt"].fillna(0) + df["solar_pt"].fillna(0)
    )
    df["res_penetration"] = df["res_generation"] / df["demand_pt"].replace(0, np.nan)

    # 3b. Thermal generation features (PT from REN)
    if "gas_ccgt_pt" in df.columns:
        df["gas_total_pt"] = df["gas_ccgt_pt"].fillna(0) + df.get("gas_cogen_pt", pd.Series(0, index=df.index)).fillna(0)
        df["thermal_total_pt"] = df["gas_total_pt"] + df.get("coal_gen_pt", pd.Series(0, index=df.index)).fillna(0) + df.get("biomass_pt", pd.Series(0, index=df.index)).fillna(0)
        df["thermal_share"] = df["thermal_total_pt"] / df["demand_pt"].replace(0, np.nan)
        df["gas_ccgt_share"] = df["gas_ccgt_pt"].fillna(0) / df["demand_pt"].replace(0, np.nan)

    # 3c. Net import position (PT)
    if "imports_pt" in df.columns and "exports_pt" in df.columns:
        df["net_imports_pt"] = df["imports_pt"].fillna(0) - df["exports_pt"].fillna(0)
        df["net_import_share"] = df["net_imports_pt"] / df["demand_pt"].replace(0, np.nan)

    # 4. OMIE risk premium (spot vs. futures spread)
    df["omie_spot_monthly_avg"] = df["omie_spot"].rolling(20, min_periods=5).mean()
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
        df["news_sentiment_ma20d"] = df["news_sentiment"].rolling(20, min_periods=1).mean()
        df["news_sentiment_shift"] = df["news_sentiment"] - df["news_sentiment"].shift(1)
    else:
        df["news_sentiment_ma20d"] = np.nan
        df["news_sentiment_shift"] = np.nan

    # 7. Per-contract autoregressive lags, momentum, and technical features.
    #    Daily lags: 1d, 2d, 3d, 5d, 10d, 15d, 20d, 25d, 30d, 65d
    contract_cols = [c for c in df.columns
                     if c.startswith("omip_") and not c.startswith("omip_m_")
                     and c not in ("omip_yr1", "omip_yr2", "omip_q1")
                     and "lag" not in c and "momentum" not in c and "vol" not in c]
    lag_days = [1, 2, 3, 5, 10, 15, 20, 25, 30, 65]
    for col in contract_cols:
        for lag in lag_days:
            df[f"{col}_lag_{lag}d"] = df[col].shift(lag)
        df[f"{col}_momentum_20d"] = df[col] - df[col].shift(20)
        df[f"{col}_momentum_65d"] = df[col] - df[col].shift(65)

    # Generic proxy lags for backward compatibility
    if "omip_yr1" in df.columns:
        for lag in lag_days:
            df[f"omip_lag_{lag}d"] = df["omip_yr1"].shift(lag)
        df["omip_momentum_20d"] = df["omip_yr1"] - df["omip_yr1"].shift(20)
        df["omip_momentum_65d"] = df["omip_yr1"] - df["omip_yr1"].shift(65)

    # 7b. Cross-contract term structure features
    q_cols = sorted([c for c in df.columns if c.startswith("omip_q") and "_lag_" not in c
                     and "_momentum_" not in c and "_vol_" not in c
                     and c not in ("omip_q1",)])
    if len(q_cols) >= 2:
        df["term_q_spread_1_2"] = df[q_cols[0]] - df[q_cols[1]]
        logger.info("Term structure: %s − %s", q_cols[0], q_cols[1])
    if len(q_cols) >= 3:
        df["term_q_spread_1_3"] = df[q_cols[0]] - df[q_cols[2]]
    yr_sorted = sorted([c for c in df.columns if c.startswith("omip_yr_")
                        and "_lag_" not in c and "_momentum_" not in c
                        and "_vol_" not in c])
    if len(yr_sorted) >= 2:
        df["term_yr_spread"] = df[yr_sorted[0]] - df[yr_sorted[1]]
    if q_cols and yr_sorted:
        df["term_q_yr_spread"] = df[q_cols[0]] - df[yr_sorted[0]]

    # 7c. Rolling volatility features (daily windows)
    for col in contract_cols:
        df[f"{col}_vol_20d"] = df[col].rolling(20, min_periods=10).std()
        df[f"{col}_vol_40d"] = df[col].rolling(40, min_periods=20).std()

    # 7d. Spot price momentum (OMIE)
    if "omie_spot" in df.columns:
        df["spot_momentum_20d"] = df["omie_spot"] - df["omie_spot"].shift(20)
        df["spot_momentum_40d"] = df["omie_spot"] - df["omie_spot"].shift(40)
        df["spot_ma_ratio"] = (
            df["omie_spot"].rolling(20, min_periods=5).mean()
            / df["omie_spot"].rolling(65, min_periods=10).mean()
        )

    # 7e. German futures momentum & spread vs OMIP
    if "german_cal_futures" in df.columns:
        df["german_momentum_20d"] = df["german_cal_futures"] - df["german_cal_futures"].shift(20)
        df["german_momentum_65d"] = df["german_cal_futures"] - df["german_cal_futures"].shift(65)
        if yr_sorted:
            df["german_omip_spread"] = df["german_cal_futures"] - df[yr_sorted[0]]

    # 7f. TTF volatility
    df["ttf_vol_20d"] = df["ttf_gas"].rolling(20, min_periods=10).std()
    df["ttf_vol_65d"] = df["ttf_gas"].rolling(65, min_periods=30).std()

    # 8. Seasonality dummies + day-of-week
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["dow"] = df.index.dayofweek  # 0=Mon, 4=Fri
    df["is_q1"] = (df["quarter"] == 1).astype(int)
    df["is_q3"] = (df["quarter"] == 3).astype(int)

    # 9. Storm anomaly flag
    df["storm_anomaly_flag"] = 0
    storm_mask = (df.index >= config.STORM_ANOMALY_START) & (df.index <= config.STORM_ANOMALY_END)
    df.loc[storm_mask, "storm_anomaly_flag"] = 1

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
    master = merge_to_master(datasets)
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
