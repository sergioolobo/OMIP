"""
01_collect_data.py — Data Collection
=====================================
Downloads and caches all raw data needed for the OMIE forecasting pipeline:

  A) OMIE day-ahead spot prices (PT/ES) — hourly, via public ZIP files
  B) TTF gas + EUA CO2 + API2 coal — daily, via yfinance
  C) ENTSO-E: generation by type, load actual + forecast, wind/solar forecasts,
     cross-border flows (PT + ES, 2015-present)
  D) Hydro reservoir anomaly (Iberian) — weekly proxy from ENTSO-E hydro generation

Inputs:  none (downloads from public sources)
Outputs: data/raw/*.csv
         data/raw/entsoe_merged_hourly.csv
"""
from __future__ import annotations

import logging
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("collect_data")
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
# A) OMIE day-ahead prices
# ===================================================================

OMIE_YEARLY_URL = (
    "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbcpt"
    "&filename=marginalpdbcpt_{year}.zip"
)
OMIE_DAILY_URL = (
    "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbcpt"
    "&filename=marginalpdbcpt_{date}.1"
)

# Year boundary — yearly ZIPs available for complete years
_YEARLY_ZIP_CUTOFF = 2022   # up to and including 2022


def _fetch_with_retry(url: str, max_attempts: int = 3, timeout: int = 30) -> Optional[requests.Response]:
    """GET url with exponential backoff. Returns None on all failures."""
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            logger.debug("  %s → HTTP %d", url, r.status_code)
        except Exception as exc:
            logger.debug("  attempt %d/%d failed: %s", attempt, max_attempts, exc)
        if attempt < max_attempts:
            time.sleep(2 ** attempt)
    return None


def _parse_omie_file(text: str) -> pd.DataFrame:
    """Parse a single OMIE marginal price text file (semicolon-delimited)."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 4:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            hour_raw = int(parts[3])
        except ValueError:
            continue
        # OMIE hours are 1-25; hour 25 = DST duplicate → skip
        if hour_raw < 1 or hour_raw > 24:
            continue
        hour = hour_raw - 1   # convert to 0-based
        try:
            price = float(parts[4].replace(",", "."))
        except (ValueError, IndexError):
            continue
        rows.append({"year": year, "month": month, "day": day,
                     "hour": hour, "price_es": price})
    return pd.DataFrame(rows)


def _omie_yearly_zip(year: int, save_dir: Path) -> pd.DataFrame:
    """Download or use cached yearly OMIE ZIP."""
    cache = save_dir / f"omie_pt_{year}.parquet"
    if cache.exists():
        logger.info("  OMIE %d: cached", year)
        return pd.read_parquet(cache)

    url = OMIE_YEARLY_URL.format(year=year)
    logger.info("  OMIE %d: fetching yearly ZIP ...", year)
    resp = _fetch_with_retry(url)
    if resp is None:
        logger.warning("  OMIE %d: yearly ZIP unavailable", year)
        return pd.DataFrame()

    import zipfile, io
    parts = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.startswith("marginal") and not name.endswith("/"):
                try:
                    text = zf.read(name).decode("latin-1")
                    parts.append(_parse_omie_file(text))
                except Exception as exc:
                    logger.debug("    skip %s: %s", name, exc)

    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(cache, index=False)
    logger.info("  OMIE %d: %d rows saved", year, len(df))
    return df


def _omie_daily_range(start: pd.Timestamp, end: pd.Timestamp,
                      save_dir: Path) -> pd.DataFrame:
    """Download daily OMIE files from start to end, caching each."""
    parts = []
    date = start
    while date <= end:
        date_str = date.strftime("%Y%m%d")
        cache = save_dir / f"omie_daily_{date_str}.parquet"
        if cache.exists():
            parts.append(pd.read_parquet(cache))
            date += pd.Timedelta(days=1)
            continue
        url = OMIE_DAILY_URL.format(year=date.year, date=date_str)
        resp = _fetch_with_retry(url, timeout=15)
        if resp is not None:
            try:
                df_d = _parse_omie_file(resp.text)
                if not df_d.empty:
                    df_d.to_parquet(cache, index=False)
                    parts.append(df_d)
            except Exception as exc:
                logger.debug("  %s parse error: %s", date_str, exc)
        date += pd.Timedelta(days=1)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_omie(save_dir: Path) -> pd.Series:
    """
    Collect all OMIE PT day-ahead hourly prices from START_DATE to today.

    Returns a timezone-naive UTC DatetimeIndex Series named 'price_es'.
    """
    logger.info("=" * 60)
    logger.info("COLLECTING OMIE PT DAY-AHEAD PRICES")
    logger.info("=" * 60)

    start_year = int(config.START_DATE[:4])
    today = pd.Timestamp.today()
    parts: list[pd.DataFrame] = []

    # Yearly ZIPs for complete historical years
    for yr in range(start_year, min(_YEARLY_ZIP_CUTOFF + 1, today.year)):
        parts.append(_omie_yearly_zip(yr, save_dir))

    # Daily files from _YEARLY_ZIP_CUTOFF+1 through today
    daily_start = pd.Timestamp(f"{_YEARLY_ZIP_CUTOFF + 1}-01-01")
    daily_end   = today.normalize()
    logger.info("Fetching daily OMIE files %s → %s ...",
                daily_start.date(), daily_end.date())
    parts.append(_omie_daily_range(daily_start, daily_end, save_dir))

    if not any(not p.empty for p in parts):
        logger.error("No OMIE data collected")
        return pd.Series(dtype=float, name="price_es")

    raw = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    raw = raw.dropna(subset=["price_es"])

    # Build DatetimeIndex
    raw["datetime"] = pd.to_datetime(
        raw[["year", "month", "day"]].astype(int)
        .rename(columns={"year": "year", "month": "month", "day": "day"})
    ) + pd.to_timedelta(raw["hour"].astype(int), unit="h")

    series = (raw.sort_values("datetime")
                 .drop_duplicates("datetime")
                 .set_index("datetime")["price_es"])
    series.index.name = "datetime"

    out = save_dir / "omie_hourly.csv"
    series.to_csv(out)
    logger.info("OMIE prices saved: %d rows, %s → %s",
                len(series), series.index.min().date(), series.index.max().date())
    return series


# ===================================================================
# B) Commodity prices (TTF, EUA, API2)
# ===================================================================

_YFINANCE_TICKERS: dict[str, str] = {
    "ttf_gas": "TTF=F",
    "eua_co2": "ECF=F",
    "api2_coal": "MTF=F",
}


def collect_commodities(save_dir: Path) -> pd.DataFrame:
    """
    Download daily TTF gas, EUA CO2, API2 coal prices via yfinance.

    Returns a DataFrame indexed by date (UTC-normalised).
    """
    logger.info("COLLECTING COMMODITY PRICES (yfinance)")
    start = pd.Timestamp(config.START_DATE)
    dfs: dict[str, pd.Series] = {}

    for name, ticker in _YFINANCE_TICKERS.items():
        cache = save_dir / f"{name}.csv"
        try:
            raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError("empty response")
            s = raw["Close"].squeeze().rename(name)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.to_csv(cache)
            dfs[name] = s
            logger.info("  %s (%s): %d rows", name, ticker, len(s))
        except Exception as exc:
            logger.warning("  %s: failed (%s)", name, exc)
            if cache.exists():
                s = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                dfs[name] = s.rename(name)
                logger.info("  %s: loaded from cache (%d rows)", name, len(dfs[name]))

    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs.values(), axis=1)
    df.index.name = "date"
    df.to_csv(save_dir / "commodities_daily.csv")
    logger.info("Commodities saved: %d rows", len(df))
    return df


# ===================================================================
# C) ENTSO-E data
# ===================================================================

def _entsoe_chunk_dates(start: pd.Timestamp, end: pd.Timestamp,
                        chunk_days: int = 30) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into ≤chunk_days windows for ENTSO-E rate limiting."""
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _safe_entsoe(fn, *args, max_retries: int = 3, **kwargs):
    """Call an ENTSO-E client method with retry + graceful 204/NoData handling."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except NoMatchingDataError:
            return None
        except Exception as exc:
            msg = str(exc)
            if "204" in msg or "No Content" in msg:
                return None
            logger.debug("  ENTSO-E attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def _flatten_gen(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Flatten MultiIndex generation columns and normalise names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col if c).strip().lower()
                                                         .replace(" ", "_")
                      for col in df.columns]
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.index = df.index.tz_convert("UTC")
    return df


def _entsoe_series_chunks(client: EntsoePandasClient, fn_name: str,
                          country: str, start: pd.Timestamp,
                          end: pd.Timestamp, **kwargs) -> pd.Series:
    """Download a Series-returning ENTSO-E query in 30-day chunks."""
    fn = getattr(client, fn_name)
    parts: list[pd.Series] = []
    for s, e in _entsoe_chunk_dates(start, end):
        result = _safe_entsoe(fn, country, start=s, end=e, **kwargs)
        if result is None:
            continue
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        result = result.tz_convert("UTC")
        parts.append(result)
        time.sleep(2)
    if not parts:
        return pd.Series(dtype=float)
    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def _entsoe_df_chunks(client: EntsoePandasClient, fn_name: str,
                      country: str, start: pd.Timestamp,
                      end: pd.Timestamp, **kwargs) -> pd.DataFrame:
    """Download a DataFrame-returning ENTSO-E query in 30-day chunks."""
    fn = getattr(client, fn_name)
    parts: list[pd.DataFrame] = []
    for s, e in _entsoe_chunk_dates(start, end):
        result = _safe_entsoe(fn, country, start=s, end=e, **kwargs)
        if result is None:
            continue
        if isinstance(result, pd.Series):
            result = result.to_frame()
        result = _flatten_gen(result, country)
        parts.append(result)
        time.sleep(2)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def download_entsoe_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    save_dir: Path,
    api_key: str,
) -> pd.DataFrame:
    """
    Download generation per source, total load, load forecasts, wind/solar
    forecasts, and cross-border flows for Spain (ES) and Portugal (PT).

    Args:
        start:    Start timestamp (timezone-aware, UTC)
        end:      End timestamp (timezone-aware, UTC)
        save_dir: Directory to save raw CSVs
        api_key:  ENTSO-E API token

    Returns:
        Merged hourly DataFrame with all generation and load columns.
    """
    if not api_key:
        logger.error("ENTSOE_API_KEY not set — skipping ENTSO-E download")
        return pd.DataFrame()

    logger.info("=" * 60)
    logger.info("COLLECTING ENTSO-E TRANSPARENCY PLATFORM DATA")
    logger.info("=" * 60)

    client = EntsoePandasClient(api_key=api_key)
    results: dict[str, pd.Series | pd.DataFrame] = {}

    for country in ["ES", "PT"]:
        country_l = country.lower()
        logger.info("--- %s ---", country)

        # 1. Actual generation by type
        logger.info("  %s generation ...", country)
        cache = save_dir / f"entsoe_generation_{country_l}.csv"
        if cache.exists() and cache.stat().st_size > 200:
            gen = pd.read_csv(cache, index_col=0, parse_dates=True, low_memory=False)
            gen.index = pd.to_datetime(gen.index, utc=True)
        else:
            gen = _entsoe_df_chunks(client, "query_generation", country, start, end)
            if not gen.empty:
                gen.to_csv(cache)
        if not gen.empty:
            results[f"gen_{country_l}"] = gen
            logger.info("  %s generation: %s", country, gen.shape)

        # 2. Actual total load
        logger.info("  %s load actual ...", country)
        cache = save_dir / f"entsoe_load_{country_l}.csv"
        if cache.exists() and cache.stat().st_size > 200:
            load = pd.read_csv(cache, index_col=0, parse_dates=True, squeeze=False).iloc[:, 0]
            load.index = pd.to_datetime(load.index, utc=True)
        else:
            load = _entsoe_series_chunks(client, "query_load", country, start, end)
            if not load.empty:
                load.name = f"load_{country_l}_mwh"
                load.to_csv(cache)
        if isinstance(load, pd.Series) and not load.empty:
            load.name = f"load_{country_l}_mwh"
            results[f"load_{country_l}"] = load

        # 3. Day-ahead load forecast
        logger.info("  %s load forecast ...", country)
        cache = save_dir / f"entsoe_load_forecast_{country_l}.csv"
        if cache.exists() and cache.stat().st_size > 200:
            lfc = pd.read_csv(cache, index_col=0, parse_dates=True, squeeze=False).iloc[:, 0]
            lfc.index = pd.to_datetime(lfc.index, utc=True)
        else:
            lfc = _entsoe_series_chunks(client, "query_load_forecast", country, start, end)
            if not lfc.empty:
                lfc.name = f"load_forecast_{country_l}_mwh"
                lfc.to_csv(cache)
        if isinstance(lfc, pd.Series) and not lfc.empty:
            lfc.name = f"load_forecast_{country_l}_mwh"
            results[f"load_forecast_{country_l}"] = lfc

        # 4. Wind + solar day-ahead generation forecasts
        logger.info("  %s renewables forecast ...", country)
        cache = save_dir / f"entsoe_renewables_forecast_{country_l}.csv"
        if cache.exists() and cache.stat().st_size > 200:
            rfc = pd.read_csv(cache, index_col=0, parse_dates=True, low_memory=False)
            rfc.index = pd.to_datetime(rfc.index, utc=True)
        else:
            rfc = _entsoe_df_chunks(client, "query_wind_and_solar_forecast",
                                    country, start, end)
            if not rfc.empty:
                rfc.to_csv(cache)
        if isinstance(rfc, pd.DataFrame) and not rfc.empty:
            results[f"rfc_{country_l}"] = rfc

    # 5. Cross-border flows ES → FR
    logger.info("Cross-border flow ES → FR ...")
    cache = save_dir / "entsoe_flow_es_fr.csv"
    if cache.exists() and cache.stat().st_size > 200:
        flow = pd.read_csv(cache, index_col=0, parse_dates=True, squeeze=False).iloc[:, 0]
        flow.index = pd.to_datetime(flow.index, utc=True)
    else:
        fn = client.query_crossborder_flows
        parts: list[pd.Series] = []
        for s, e in _entsoe_chunk_dates(start, end):
            result = _safe_entsoe(fn, "ES", "FR", start=s, end=e)
            if result is not None:
                result = result.tz_convert("UTC")
                parts.append(result)
            time.sleep(2)
        flow = (pd.concat(parts).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
                if parts else pd.Series(dtype=float))
        if not flow.empty:
            flow.name = "flow_es_fr_mwh"
            flow.to_csv(cache)
    if isinstance(flow, pd.Series) and not flow.empty:
        flow.name = "flow_es_fr_mwh"
        results["flow_es_fr"] = flow

    # ---- Merge into hourly DataFrame ----
    logger.info("Merging ENTSO-E datasets ...")
    all_series: list[pd.Series] = []

    for key, data in results.items():
        if isinstance(data, pd.Series) and not data.empty:
            all_series.append(data.rename(data.name or key))
        elif isinstance(data, pd.DataFrame) and not data.empty:
            # Extract useful generation columns
            for col in data.columns:
                if any(src in col for src in
                       ["wind", "solar", "hydro", "nuclear", "gas", "coal",
                        "biomass", "load"]):
                    suffix = key.split("_")[-1]
                    all_series.append(data[col].rename(f"{col}_{suffix}"))

    if not all_series:
        logger.warning("No ENTSO-E series available — returning empty DataFrame")
        return pd.DataFrame()

    merged = pd.concat(all_series, axis=1)
    merged.index = pd.to_datetime(merged.index, utc=True)
    merged = merged.resample("h").mean()

    # DST handling: fill March gap (NaN after resample), average October duplicate
    merged = merged.interpolate(method="time", limit=2)

    out = save_dir / "entsoe_merged_hourly.csv"
    merged.to_csv(out)
    logger.info("ENTSO-E merged: %d rows × %d cols → %s",
                len(merged), merged.shape[1], out.name)
    return merged


# ===================================================================
# D) Hydro proxy from existing OMIP data (if available)
# ===================================================================

def collect_hydro_proxy(save_dir: Path) -> pd.DataFrame:
    """
    Build a weekly hydro anomaly proxy from ENTSO-E hydro generation.
    Falls back to zero if ENTSO-E data is not available.
    """
    cache = save_dir / "hydro_weekly.csv"
    entsoe_path = save_dir / "entsoe_merged_hourly.csv"
    if not entsoe_path.exists():
        pd.DataFrame({"hydro_anomaly": pd.Series(dtype=float)}).to_csv(cache)
        return pd.DataFrame()

    df = pd.read_csv(entsoe_path, index_col=0, parse_dates=True, low_memory=False)
    hydro_cols = [c for c in df.columns if "hydro" in c]
    if not hydro_cols:
        pd.DataFrame({"hydro_anomaly": pd.Series(dtype=float)}).to_csv(cache)
        return pd.DataFrame()

    hydro = df[hydro_cols].sum(axis=1)
    # Anomaly = deviation from rolling 52-week (364-day) mean
    hydro_weekly = hydro.resample("W").mean()
    hydro_weekly.name = "hydro_mwh"
    roll_mean = hydro_weekly.rolling(52, min_periods=4).mean()
    anom = (hydro_weekly - roll_mean) / (roll_mean.replace(0, np.nan))
    anom.name = "hydro_anomaly"
    out = pd.concat([hydro_weekly, anom], axis=1)
    out.to_csv(cache)
    logger.info("Hydro proxy: %d weekly rows", len(out))
    return out


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run the full data collection pipeline."""
    logger.info("=" * 60)
    logger.info("DATA COLLECTION PIPELINE")
    logger.info("=" * 60)

    # A) OMIE prices
    collect_omie(config.DATA_RAW)

    # B) Commodity prices
    collect_commodities(config.DATA_RAW)

    # C) ENTSO-E
    download_entsoe_data(
        start=pd.Timestamp(config.START_DATE, tz="UTC"),
        end=pd.Timestamp.now(tz="UTC"),
        save_dir=config.DATA_RAW,
        api_key=config.ENTSOE_API_KEY,
    )

    # D) Hydro proxy
    collect_hydro_proxy(config.DATA_RAW)

    logger.info("=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
