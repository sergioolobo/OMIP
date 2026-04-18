"""
02_build_features.py — Feature Engineering
===========================================
Loads all raw data from data/raw/, engineers the full feature set described
in config.py, and saves:

  - data/processed/master_hourly.csv         — single hourly DataFrame (all features)
  - data/processed/hourly_dataset_h{HH}.csv  (×24) — per-hour training sets
  - data/processed/data_quality_report.txt

Inputs:  data/raw/omie_hourly.csv                        (date, hour 1-24, price_eur_mwh)
         data/raw/ttf_gas.csv / eua_co2.csv / api2_coal.csv  (yfinance, # comment header)
         data/raw/entsoe_renewables_forecast_{PT,ES}_hourly.csv
         data/raw/entsoe_load_forecast_{PT,ES}_hourly.csv
         data/raw/entsoe_load_actual_{PT,ES}_hourly.csv
         data/raw/entsoe_generation_{PT,ES}_hourly.csv
         data/raw/entsoe_flows_ES_FR_hourly.csv
Outputs: data/processed/*.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("build_features")
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
# File loaders
# ===================================================================

def _entsoe_ts(path: Path, value_col: str | None = None) -> pd.Series:
    """
    Load an ENTSO-E hourly CSV (timestamp + value column).
    Returns a tz-naive UTC DatetimeIndex Series.
    """
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    # Coerce all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if value_col and value_col in df.columns:
        return df[value_col]
    # Return the first numeric column
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        return df[num_cols[0]]
    return pd.Series(dtype=float)


def _entsoe_df(path: Path) -> pd.DataFrame:
    """
    Load a multi-column ENTSO-E hourly CSV (generation, renewables forecast).
    Returns a tz-naive UTC DatetimeIndex DataFrame.
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _commodity(path: Path, col_name: str) -> pd.Series:
    """
    Load a yfinance commodity CSV that may have a # comment header.
    Returns a tz-naive DatetimeIndex Series.
    """
    if not path.exists():
        return pd.Series(dtype=float, name=col_name)
    # Detect comment header
    with path.open() as fh:
        first_line = fh.readline()
    skip = 1 if first_line.strip().startswith("#") else 0
    df = pd.read_csv(path, skiprows=skip, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    # Find the relevant column
    for col in df.columns:
        if col_name.lower() in col.lower() or col.lower() == col_name.lower():
            return df[col].rename(col_name)
    # Fall back to first numeric column
    num = df.select_dtypes(include="number")
    if not num.empty:
        return num.iloc[:, 0].rename(col_name)
    return pd.Series(dtype=float, name=col_name)


def _load_omie(path: Path) -> pd.Series:
    """
    Load OMIE prices. Columns: date, hour (1-24), price_eur_mwh.
    Returns a tz-naive UTC DatetimeIndex Series named 'price_es'.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    # OMIE hours are 1-based; convert to 0-based
    df["hour"] = df["hour"].astype(int) - 1
    df = df[df["hour"].between(0, 23)].copy()
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = (df.sort_values("datetime")
            .drop_duplicates("datetime")
            .set_index("datetime"))
    df.index.name = "datetime"
    return df["price_eur_mwh"].rename("price_es")


# ===================================================================
# Previous-day curve (vectorised)
# ===================================================================

def _add_prev_day_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prev_day_h00 … prev_day_h23: the 24 hourly prices of the preceding
    calendar day.  Uses vectorised pivot-shift-merge.
    """
    price = df["price_es"].copy()
    tmp = price.reset_index()
    tmp.columns = ["datetime", "price_es"]
    tmp["date"] = tmp["datetime"].dt.normalize()
    tmp["hour"] = tmp["datetime"].dt.hour

    wide = tmp.pivot(index="date", columns="hour", values="price_es")
    wide_prev = wide.shift(1)
    wide_prev.columns = [f"prev_day_h{h:02d}" for h in wide_prev.columns]

    merged = (tmp[["datetime", "date"]]
              .merge(wide_prev.reset_index(), on="date", how="left")
              .set_index("datetime")
              .drop(columns=["date"]))
    return df.join(merged, how="left")


# ===================================================================
# Build ENTSO-E aggregate features
# ===================================================================

def _build_entsoe_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Read all ENTSO-E files and return a DataFrame aligned to idx with:
      - wind_onshore_forecast_{es,pt}_mwh
      - wind_offshore_forecast_pt_mwh
      - solar_forecast_{es,pt}_mwh
      - load_forecast_{es,pt}_mwh
      - flow_es_fr_mwh
      - wind_total, solar_total, load_total, res_share  (derived aggregates)
      - hydro_anomaly  (from weekly reservoir proxy)
    """
    raw = config.DATA_RAW
    result: dict[str, pd.Series] = {}

    def _reindex(s: pd.Series, name: str) -> None:
        if s.empty:
            result[name] = pd.Series(np.nan, index=idx, name=name)
        else:
            # Resample to hourly mean first (handles 15-min resolution)
            s = s.resample("h").mean()
            result[name] = s.reindex(idx, method="ffill").rename(name)

    # Renewables forecasts
    rfc_pt = _entsoe_df(raw / "entsoe_renewables_forecast_PT_hourly.csv")
    rfc_es = _entsoe_df(raw / "entsoe_renewables_forecast_ES_hourly.csv")

    _reindex(rfc_pt.get("Wind Onshore",  pd.Series(dtype=float)), "wind_onshore_forecast_pt_mwh")
    _reindex(rfc_pt.get("Wind Offshore", pd.Series(dtype=float)), "wind_offshore_forecast_pt_mwh")
    _reindex(rfc_pt.get("Solar",         pd.Series(dtype=float)), "solar_forecast_pt_mwh")
    _reindex(rfc_es.get("Wind Onshore",  pd.Series(dtype=float)), "wind_onshore_forecast_es_mwh")
    _reindex(rfc_es.get("Solar",         pd.Series(dtype=float)), "solar_forecast_es_mwh")

    # Load forecasts
    _reindex(_entsoe_ts(raw / "entsoe_load_forecast_PT_hourly.csv"), "load_forecast_pt_mwh")
    _reindex(_entsoe_ts(raw / "entsoe_load_forecast_ES_hourly.csv"), "load_forecast_es_mwh")

    # Cross-border flow ES → FR
    _reindex(_entsoe_ts(raw / "entsoe_flows_ES_FR_hourly.csv"), "flow_es_fr_mwh")

    # Actual generation for aggregates (wind + solar + hydro totals)
    gen_pt = _entsoe_df(raw / "entsoe_generation_PT_hourly.csv")
    gen_es = _entsoe_df(raw / "entsoe_generation_ES_hourly.csv")

    def _sum_like(df: pd.DataFrame, keyword: str) -> pd.Series:
        cols = [c for c in df.columns if keyword.lower() in c.lower()]
        return df[cols].sum(axis=1) if cols else pd.Series(0.0, index=df.index)

    wind_pt_gen  = _sum_like(gen_pt, "wind")
    solar_pt_gen = _sum_like(gen_pt, "solar")
    wind_es_gen  = _sum_like(gen_es, "wind")
    solar_es_gen = _sum_like(gen_es, "solar")
    hydro_pt_gen = _sum_like(gen_pt, "hydro")
    hydro_es_gen = _sum_like(gen_es, "hydro")

    load_pt = _entsoe_ts(raw / "entsoe_load_actual_PT_hourly.csv")
    load_es = _entsoe_ts(raw / "entsoe_load_actual_ES_hourly.csv")

    wind_total  = (wind_pt_gen  + wind_es_gen).resample("h").mean().reindex(idx, method="ffill")
    solar_total = (solar_pt_gen + solar_es_gen).resample("h").mean().reindex(idx, method="ffill")
    load_total  = (load_pt + load_es.reindex(load_pt.index, method="ffill")
                   ).resample("h").mean().reindex(idx, method="ffill")

    result["wind_total"]  = wind_total.rename("wind_total")
    result["solar_total"] = solar_total.rename("solar_total")
    result["load_total"]  = load_total.rename("load_total")
    result["res_share"]   = ((wind_total + solar_total) / load_total.replace(0, np.nan)
                             ).rename("res_share")

    # Hydro anomaly: weekly deviation from 52-week rolling mean
    def _clean_series(s: pd.Series) -> pd.Series:
        s = s.sort_index()
        return s[~s.index.duplicated(keep="last")]

    hydro_pt_clean = _clean_series(hydro_pt_gen)
    hydro_es_clean = _clean_series(hydro_es_gen)
    # Combine on a shared hourly grid
    hydro_combined = pd.concat([hydro_pt_clean.rename("pt"), hydro_es_clean.rename("es")], axis=1)
    hydro_combined = hydro_combined.sort_index()
    hydro_combined = hydro_combined[~hydro_combined.index.duplicated(keep="last")]
    hydro_total    = hydro_combined.sum(axis=1, min_count=1)
    hydro_weekly   = hydro_total.resample("W").mean()
    roll52         = hydro_weekly.rolling(52, min_periods=4).mean()
    hydro_anom     = ((hydro_weekly - roll52) / roll52.replace(0, np.nan)).rename("hydro_anomaly")
    # Forward-fill weekly → hourly
    result["hydro_anomaly"] = hydro_anom.reindex(idx, method="ffill").fillna(0.0)

    out = pd.DataFrame(result, index=idx)
    return out


# ===================================================================
# Main feature engineering
# ===================================================================

def build_features() -> pd.DataFrame:
    """
    Build the full hourly feature matrix and save per-hour CSVs.

    Returns:
        Master hourly DataFrame with all features.
    """
    logger.info("=" * 60)
    logger.info("BUILDING FEATURE MATRIX")
    logger.info("=" * 60)

    omie_path = config.DATA_RAW / "omie_hourly.csv"
    if not omie_path.exists():
        raise FileNotFoundError(f"{omie_path} not found — run 01_collect_data.py first")

    # ---- 1. OMIE prices ----
    price = _load_omie(omie_path)
    logger.info("OMIE prices: %d rows, %s → %s",
                len(price), price.index.min().date(), price.index.max().date())

    df = price.to_frame()
    idx = df.index  # master hourly index

    # ---- 2. AR lags ----
    logger.info("Adding AR lags ...")
    for lag_h in config.LAG_HOURS:
        df[f"price_lag_{lag_h}h"] = df["price_es"].shift(lag_h)

    df["price_roll_mean_24h"]  = df["price_es"].rolling(24).mean()
    df["price_roll_std_24h"]   = df["price_es"].rolling(24).std()
    df["price_roll_mean_168h"] = df["price_es"].rolling(168).mean()
    df["price_roll_std_168h"]  = df["price_es"].rolling(168).std()

    # ---- 3. Previous-day curve ----
    logger.info("Adding previous-day curve ...")
    df = _add_prev_day_curve(df)

    # ---- 4. Calendar features ----
    logger.info("Adding calendar features ...")
    df["hour_of_day"]    = df.index.hour
    df["day_of_week"]    = df.index.dayofweek
    df["month"]          = df.index.month
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]     = df.index.normalize().to_series().dt.date.map(
        lambda d: int(d in config.IBERIAN_HOLIDAYS)
    ).values
    df["is_summer"]      = df["month"].isin([7, 8, 9]).astype(int)
    df["is_winter"]      = df["month"].isin([12, 1, 2]).astype(int)
    df["hour_x_weekend"] = df["hour_of_day"] * df["is_weekend"]

    # ---- 5. Commodity prices (daily → hourly forward-fill) ----
    logger.info("Loading commodity prices ...")
    ttf  = _commodity(config.DATA_RAW / "ttf_gas.csv",  "ttf_gas")
    co2  = _commodity(config.DATA_RAW / "eua_co2.csv",  "eua_co2")
    coal = _commodity(config.DATA_RAW / "api2_coal.csv", "api2_coal")

    for s in [ttf, co2, coal]:
        if not s.empty:
            s_h = s.resample("h").ffill().reindex(idx, method="ffill")
            df[s.name] = s_h.values
            logger.info("  %s: %d rows", s.name, s.notna().sum())
        else:
            df[s.name] = np.nan

    if "ttf_gas" in df.columns and "eua_co2" in df.columns:
        df["spark_spread"] = (df["ttf_gas"] / 293.07) / 0.52 - df["eua_co2"] * 0.202
    else:
        df["spark_spread"] = np.nan

    # ---- 6. ENTSO-E features ----
    logger.info("Building ENTSO-E features ...")
    entsoe_df = _build_entsoe_features(idx)
    for col in entsoe_df.columns:
        df[col] = entsoe_df[col].values
    logger.info("  ENTSO-E features added: %d", len(entsoe_df.columns))

    # ---- 7. Anomaly flags ----
    logger.info("Adding anomaly flags ...")
    df["storm_flag"] = 0
    storm_mask = (
        (df.index >= pd.Timestamp(config.STORM_ANOMALY_START))
        & (df.index <= pd.Timestamp(config.STORM_ANOMALY_END))
    )
    df.loc[storm_mask, "storm_flag"] = 1

    p2  = df["price_es"].quantile(0.02)
    p98 = df["price_es"].quantile(0.98)
    df["spike_flag"] = ((df["price_es"] < p2) | (df["price_es"] > p98)).astype(int)

    # ---- 8. Save ----
    df.index.name = "datetime"
    master_path = config.DATA_PROCESSED / "master_hourly.csv"
    df.to_csv(master_path)
    logger.info("Master dataset: %d rows × %d cols → %s",
                len(df), df.shape[1], master_path.name)

    logger.info("Saving 24 per-hour datasets ...")
    for h in config.HOURS:
        df_h = df[df.index.hour == h].copy()
        df_h.to_csv(config.DATA_PROCESSED / f"hourly_dataset_h{h:02d}.csv")

    _write_quality_report(df)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    return df


def _write_quality_report(df: pd.DataFrame) -> None:
    report = config.DATA_PROCESSED / "data_quality_report.txt"
    lines: list[str] = [
        "=" * 70,
        "DATA QUALITY REPORT",
        "=" * 70,
        f"Date range  : {df.index.min()} -> {df.index.max()}",
        f"Total rows  : {len(df):,}",
        f"Total cols  : {df.shape[1]}",
        "",
        f"  {'Column':<40} {'%NaN':>8}  {'N NaN':>8}",
        "  " + "-" * 60,
    ]
    for col in sorted(df.columns):
        n_nan = df[col].isna().sum()
        pct   = 100 * n_nan / len(df)
        lines.append(f"  {col:<40} {pct:>7.1f}%  {n_nan:>8,}")
    lines += ["", "Per-hour row counts (with non-null price):"]
    for h in config.HOURS:
        n = df[(df.index.hour == h) & df["price_es"].notna()].shape[0]
        lines.append(f"  H{h:02d}: {n:,} rows")
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Quality report -> %s", report.name)


if __name__ == "__main__":
    build_features()
