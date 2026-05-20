"""
01b_refresh_incremental.py — CI-safe incremental data refresh
=============================================================
Writes EXACTLY the raw files that 02_build_features.py reads, fetching only the
recent gap so it is fast and can never hang on permanently-missing history.

  OMIE prices  -> data/raw/omie_hourly.csv          (date, hour 1-24, price_eur_mwh)
  ENTSO-E      -> data/raw/entsoe_*_hourly.csv       (timestamp + value columns)

Why this exists instead of 01_collect_data.py:
  * 01 writes a different OMIE format (datetime,price_es) than 02 expects
    (date,hour,price_eur_mwh) and caches ENTSO-E under different filenames —
    so running it would not update (and could corrupt) what 02 reads.
  * 01's OMIE daily loop retries permanently-missing 2023 files every run and
    effectively hangs.

Strategy
  OMIE:   seed omie_hourly.csv from processed/master_hourly.csv if absent
          (cold start — no historical fetch, no hang), then fetch only daily
          files from the last present day to today (skip-on-404, fast).
  ENTSO-E: per series, if the *_hourly.csv exists fetch [last_ts-1d, now] and
          append; otherwise (cold start) chunked full fetch from START_DATE.

Idempotent: re-running only appends + de-dups on the time index.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from entsoe import EntsoePandasClient  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("refresh_incremental")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE))
logger.addHandler(_h)

RAW = config.DATA_RAW
# OMIE day-ahead marginal price (the real spot price the model targets).
# NOT marginalpdbcpt — that is the flat base-program (PDBC) price.
OMIE_DAILY_URL = (
    "https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc"
    "&filename=marginalpdbc_{date}.1"
)


def _parse_omie_native(text: str) -> list[tuple[pd.Timestamp, float]]:
    """Parse an OMIE marginalpdbc file -> list of (datetime, price).

    Since the EU 15-minute MTU switch (~Oct/Nov 2025) OMIE publishes 96
    quarter-hourly periods per day; before that it was 24 hourly periods.
    We detect the granularity per day from the period count and map each
    period to its interval-start timestamp in Iberian local time:

      * <=28 periods -> hourly:        period p -> (p-1):00
      * otherwise     -> quarter-hour: period p -> (p-1)*15 min

    DST days (23/25 hours -> 92/100 quarter-hours) fall out naturally.
    The caller resamples to hourly mean, which is the convention
    02_build_features._load_omie / ENTSO-E use for the hourly price.
    """
    by_day: dict[tuple[int, int, int], list[tuple[int, float]]] = {}
    for line in text.splitlines():
        p = line.split(";")
        if len(p) >= 5 and p[0].isdigit() and p[1].isdigit():
            try:
                key = (int(p[0]), int(p[1]), int(p[2]))
                by_day.setdefault(key, []).append((int(p[3]), float(p[4])))
            except (ValueError, IndexError):
                continue

    rows: list[tuple[pd.Timestamp, float]] = []
    for (y, m, d), pairs in by_day.items():
        step = (pd.Timedelta(hours=1) if len(pairs) <= 28
                else pd.Timedelta(minutes=15))
        base = pd.Timestamp(y, m, d)
        for period, price in pairs:
            rows.append((base + step * (period - 1), price))
    return rows


# ===========================================================================
# OMIE prices
# ===========================================================================

def _load_existing_omie() -> pd.Series:
    """Return the existing hourly price series (datetime-indexed), or seed it
    from the committed master_hourly.csv when the raw file is missing."""
    path = RAW / "omie_hourly.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(
            df["hour"].astype(int) - 1, unit="h"
        )
        s = df.set_index("datetime")["price_eur_mwh"].sort_index()
        return s[~s.index.duplicated(keep="last")]
    # Cold start — seed full history from the processed master (always committed)
    master = config.DATA_PROCESSED / "master_hourly.csv"
    if master.exists():
        logger.info("omie_hourly.csv absent — seeding from master_hourly.csv")
        m = pd.read_csv(master, index_col="datetime", parse_dates=True)
        return m["price_es"].dropna().sort_index()
    logger.warning("No OMIE source available — starting empty")
    return pd.Series(dtype=float)


def refresh_omie() -> None:
    series = _load_existing_omie()
    start = (series.index.max().normalize() if len(series)
             else pd.Timestamp(config.START_DATE))
    end = pd.Timestamp.today().normalize()
    logger.info("OMIE: fetching daily files %s -> %s", start.date(), end.date())

    rows, ok = [], 0
    d = start
    while d <= end:
        try:
            r = requests.get(OMIE_DAILY_URL.format(date=d.strftime("%Y%m%d")),
                             timeout=15)
            if r.status_code == 200 and "MARGINALPDBC" in r.text:
                day_rows = _parse_omie_native(r.text)
                if day_rows:
                    rows.extend(day_rows)
                    ok += 1
        except Exception as exc:        # noqa: BLE001 — never let one day kill the run
            logger.debug("OMIE %s failed: %s", d.date(), exc)
        d += pd.Timedelta(days=1)
    logger.info("OMIE: %d new daily files fetched", ok)

    if rows:
        new = pd.Series(dict(rows)).sort_index()
        new = new[~new.index.duplicated(keep="last")]
        # Aggregate quarter-hourly (96/day) to hourly mean; hourly days pass
        # through unchanged. This is the standard hourly-price convention.
        new = new.resample("h").mean().dropna()
        series = pd.concat([series, new])
        series = series[~series.index.duplicated(keep="last")].sort_index()

    out = pd.DataFrame({
        "date": series.index.normalize().date,
        "hour": series.index.hour + 1,
        "price_eur_mwh": series.values,
    })
    out.to_csv(RAW / "omie_hourly.csv", index=False)
    logger.info("OMIE: omie_hourly.csv -> %s (%d rows)",
                series.index.max(), len(out))


# ===========================================================================
# ENTSO-E
# ===========================================================================

_client = EntsoePandasClient(api_key=config.ENTSOE_API_KEY)


def _merge_hourly(path: Path, new: pd.DataFrame) -> None:
    """Resample new -> hourly UTC, align columns, append to existing, de-dup."""
    if new is None or new.empty:
        logger.info("  SKIP %s (empty fetch)", path.name)
        return
    new = new.loc[:, ~new.columns.duplicated()].copy()
    new.index = (new.index.tz_localize("UTC") if new.index.tz is None
                 else new.index.tz_convert("UTC"))
    new = new.apply(pd.to_numeric, errors="coerce").resample("h").mean()

    if path.exists():
        ex = pd.read_csv(path)
        ex["timestamp"] = pd.to_datetime(ex["timestamp"], utc=True)
        ex = ex.set_index("timestamp")
        ex = ex.loc[:, ~ex.columns.duplicated()]
        for col in new.columns:
            if col not in ex.columns:
                ex[col] = pd.NA
        new = new.reindex(columns=ex.columns)
        combined = pd.concat([ex, new])
    else:
        combined = new
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = "timestamp"
    combined.to_csv(path)
    logger.info("  OK %s -> %s (%d rows)", path.name, combined.index.max(),
                len(combined))


def _window(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) UTC: small overlap if file exists, else full history."""
    end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    if path.exists():
        ex = pd.read_csv(path, usecols=["timestamp"])
        last = pd.to_datetime(ex["timestamp"], utc=True).max()
        return last - pd.Timedelta(days=2), end
    return pd.Timestamp(config.START_DATE, tz="UTC"), end


def _fetch_chunked(fn, *args, start, end):
    """Call an entsoe query in <=120-day chunks (cold-start safe)."""
    parts = []
    s = start
    while s < end:
        e = min(s + pd.Timedelta(days=120), end)
        try:
            r = fn(*args, start=s, end=e)
            if r is not None and len(r):
                parts.append(r)
        except Exception as exc:        # noqa: BLE001
            logger.debug("  chunk %s-%s failed: %s", s.date(), e.date(), exc)
        s = e
    if not parts:
        return None
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def refresh_entsoe() -> None:
    for cc in ("ES", "PT"):
        logger.info("ENTSO-E %s ...", cc)
        specs = [
            (f"entsoe_load_actual_{cc}_hourly.csv",
             _client.query_load, (cc,), {"Actual Load": "load_actual"}),
            (f"entsoe_load_forecast_{cc}_hourly.csv",
             _client.query_load_forecast, (cc,), {"Forecasted Load": "load_forecast"}),
            (f"entsoe_renewables_forecast_{cc}_hourly.csv",
             _client.query_wind_and_solar_forecast, (cc,), {}),
            (f"entsoe_generation_{cc}_hourly.csv",
             _client.query_generation, (cc,), {}),
        ]
        for fname, fn, args, rename in specs:
            path = RAW / fname
            s, e = _window(path)
            try:
                df = _fetch_chunked(fn, *args, start=s, end=e)
            except Exception as exc:    # noqa: BLE001
                logger.warning("  ERR fetch %s: %s", fname, exc)
                continue
            if df is not None:
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in df.columns]
                if rename:
                    df = df.rename(columns=rename)
            _merge_hourly(path, df)

    logger.info("ENTSO-E flows ...")
    for src, dst in (("ES", "FR"), ("FR", "ES"), ("ES", "PT"), ("PT", "ES")):
        path = RAW / f"entsoe_flows_{src}_{dst}_hourly.csv"
        s, e = _window(path)
        try:
            fl = _fetch_chunked(_client.query_crossborder_flows, src, dst,
                                start=s, end=e)
        except Exception as exc:        # noqa: BLE001
            logger.warning("  ERR fetch %s->%s: %s", src, dst, exc)
            continue
        if fl is not None:
            _merge_hourly(path, fl.to_frame(name="flows"))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("INCREMENTAL DATA REFRESH")
    logger.info("=" * 60)
    refresh_omie()
    if config.ENTSOE_API_KEY:
        refresh_entsoe()
    else:
        logger.warning("ENTSOE_API_KEY not set — skipping ENTSO-E refresh")
    logger.info("=" * 60)
    logger.info("REFRESH COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
