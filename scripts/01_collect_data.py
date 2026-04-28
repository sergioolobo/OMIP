"""
01_collect_data.py — Data Collection Module
============================================
Downloads and caches all raw datasets required for the OMIP Futures
forecasting pipeline.

DATA SOURCE ADAPTATIONS (vs. original design):
  • OMIP futures  → commodity_data lib (OmipDownloader) if available,
                    else manual xlsx fallback
  • PT generation → REN Data Hub "Production Breakdown PT.xlsx" (already on disk)
  • ES generation → REMOVED
  • Hydro ES      → embalses.net weekly scraper + manual CSV fallback
  • PT hydro      → DROPPED (SNIRH unavailable)
  • German futures→ investing.com CSV already on disk + yfinance fallback
  • PT demand     → extracted from the same REN xlsx file
  • ES demand     → REMOVED

Inputs:  External APIs / web sources / local files placed by the user
Outputs: CSV files in  data/raw/
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("collect_data")
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

def _is_fresh(path: Path, max_age_days: int = config.CACHE_FRESHNESS_DAYS) -> bool:
    """Return True if *path* exists and was modified less than *max_age_days* ago.

    Skipped entirely when the FORCE_REFRESH environment variable is truthy
    (``"1"``, ``"true"``, etc.) — used by the GitHub Actions daily pipeline so
    that freshly-cloned files (all with today's mtime) don't spuriously short-
    circuit the collection step. Local runs without the env var keep the old
    cache-skipping behaviour.
    """
    if os.getenv("FORCE_REFRESH", "").lower() in {"1", "true", "yes"}:
        return False
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=max_age_days)


def _save_csv(df: pd.DataFrame, path: Path, source_label: str) -> None:
    """Save DataFrame to CSV with a header comment line."""
    header_line = f"# Source: {source_label} | Downloaded: {datetime.now().isoformat()}\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header_line)
        df.to_csv(fh)
    logger.info("Saved %s  (%d rows)", path.name, len(df))


def _retry(func: Callable, *args, max_retries: int = config.MAX_RETRIES, **kwargs):
    """Call *func* with exponential back-off.  Returns result or None."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            wait = config.RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "Attempt %d/%d for %s failed: %s — retrying in %.0fs",
                attempt, max_retries, func.__name__, exc, wait,
            )
            time.sleep(wait)
    logger.error("All %d attempts for %s exhausted.", max_retries, func.__name__)
    return None


def _yfinance_download(
    ticker: str,
    start: str,
    end: str,
    label: str,
) -> Optional[pd.DataFrame]:
    """Download a single ticker from yfinance with retry logic."""
    import yfinance as yf

    def _dl() -> pd.DataFrame:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {ticker}")
        return data

    result = _retry(_dl)
    if result is None:
        logger.warning("Could not download %s (%s) — returning empty frame.", label, ticker)
        return None
    # Flatten MultiIndex columns if present
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    return result


def _write_placeholder(path: Path, columns: list[str], source: str,
                       freq: str = "W-MON") -> None:
    """Write a placeholder CSV with NaN values so the pipeline continues."""
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq=freq)
    df = pd.DataFrame({c: np.nan for c in columns}, index=idx)
    df.index.name = "date"
    _save_csv(df, path, source)


# ===================================================================
# A) OMIP Historical Futures Prices
#    Primary: parse settlement prices from OMIP bulletin PDFs (parte1)
#    Fallback 1: commodity_data (OmipDownloader)
#    Fallback 2: manual xlsx at data/raw/omip_manual.xlsx
# ===================================================================

def _load_existing_omip_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load the previously-saved OMIP CSV, skipping the leading comment line.

    Returns None if the file is missing or unreadable. Used by
    ``collect_omip_futures`` to merge freshly-parsed bulletins with
    historical rows that aren't on disk in the current environment
    (e.g. CI runners where bulletin PDFs are gitignored)."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, comment="#", index_col="date", parse_dates=True)
        if df.empty:
            return None
        return df
    except Exception as exc:
        logger.warning("Could not read existing OMIP CSV (%s): %s", path.name, exc)
        return None


def _merge_omip(existing: Optional[pd.DataFrame],
                fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine an existing OMIP CSV with a fresh bulletin-parse result.

    Fresh data overrides existing values on duplicate (date, instrument)
    pairs.  Both indexes are normalised to dates and the union of
    instrument columns is preserved so we never *lose* a column when
    the fresh slice happens to not contain it.
    """
    if existing is None or existing.empty:
        return fresh
    # Align column union so dropping/adding instruments doesn't blow away rows
    all_cols = sorted(set(existing.columns) | set(fresh.columns))
    existing = existing.reindex(columns=all_cols)
    fresh    = fresh.reindex(columns=all_cols)
    # combine_first(other) = self where not NaN, else other  →  we want the
    # OPPOSITE: prefer fresh, fall back to existing.  fresh.combine_first(existing)
    merged = fresh.combine_first(existing)
    merged.sort_index(inplace=True)
    merged.index.name = "date"
    return merged


def collect_omip_futures() -> None:
    """Collect OMIP settlement prices — bulletins → commodity_data → manual.

    Merges newly-parsed bulletins with the previously-saved CSV so that
    environments which only have a partial bulletin archive on disk
    (e.g. GitHub Actions runners) cannot accidentally truncate the
    historical record. The resulting CSV is *always* a superset of what
    was on disk before the run.
    """
    path = config.RAW_FILES["omip_futures"]
    if _is_fresh(path):
        logger.info("OMIP futures file is fresh — skipping.")
        return

    logger.info("Collecting OMIP futures data …")

    existing = _load_existing_omip_csv(path)
    if existing is not None:
        logger.info("Existing OMIP CSV: %d dates × %d instruments — will merge.",
                    *existing.shape)

    # --- Strategy 1: parse bulletin PDFs (primary source) ---
    try:
        fresh = _parse_omip_bulletins()
        if fresh is not None and not fresh.empty:
            merged = _merge_omip(existing, fresh)
            logger.info("OMIP merge: existing=%d  fresh=%d  merged=%d rows",
                        0 if existing is None else len(existing),
                        len(fresh), len(merged))
            _save_csv(merged, path, "OMIP bulletin PDFs (FTB settlement prices)")
            return
    except Exception as exc:
        logger.warning("OMIP bulletin parsing failed: %s", exc)

    # --- Strategy 2: commodity_data library ---
    try:
        fresh = _download_omip_commodity_data()
        if fresh is not None and not fresh.empty:
            merged = _merge_omip(existing, fresh)
            _save_csv(merged, path, "commodity_data OmipDownloader")
            return
    except Exception as exc:
        logger.warning("commodity_data OMIP download failed: %s", exc)

    # --- Strategy 3: manual Excel fallback ---
    fresh = manual_load_omip()
    if fresh is not None and not fresh.empty:
        merged = _merge_omip(existing, fresh)
        _save_csv(merged, path, "omip_manual.xlsx")
        return

    # --- Last resort: no fresh data, but existing CSV exists → keep it ---
    if existing is not None:
        logger.warning(
            "No fresh OMIP data this run — preserving existing %d-row CSV.",
            len(existing),
        )
        return

    logger.error(
        "No OMIP futures data available.\n"
        "  Option A: Place bulletin PDFs in configured dirs (see config.OMIP_BULLETIN_DIRS).\n"
        "  Option B: Set up commodity_data + ong_tsdb (see README).\n"
        "  Option C: Place a manual file at %s",
        config.RAW_FILES["omip_manual"],
    )


def _parse_omip_bulletins() -> Optional[pd.DataFrame]:
    """Parse FTB settlement prices from all OMIP bulletin parte1 PDFs.

    Scans config.OMIP_BULLETIN_DIRS for files matching
    boletim_YYYYMMDD_parte1.pdf, extracts section 1.1 (MIBEL SPEL Base
    Load Futures — FTB), and returns a wide-format DataFrame indexed by
    date with one column per instrument (e.g. omip_Q3-25, omip_YR-27).
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed — cannot parse bulletin PDFs. "
                       "Install with: pip install pdfplumber")
        return None

    import re

    # Discover all parte1 PDF files
    pdf_files: list[Path] = []
    for bdir in config.OMIP_BULLETIN_DIRS:
        if bdir.exists():
            pdf_files.extend(sorted(bdir.glob("boletim_*_parte1.pdf")))
        else:
            logger.warning("Bulletin directory not found: %s", bdir)

    if not pdf_files:
        logger.warning("No OMIP bulletin PDFs found.")
        return None

    logger.info("Found %d OMIP bulletin parte1 PDFs to parse.", len(pdf_files))

    all_records: list[dict] = []
    errors = 0

    for pdf_path in pdf_files:
        try:
            records = _parse_single_bulletin(pdf_path, pdfplumber, re)
            all_records.extend(records)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                logger.warning("Failed to parse %s: %s", pdf_path.name, exc)

    if errors > 5:
        logger.warning("… and %d more bulletin parse errors.", errors - 5)

    if not all_records:
        logger.warning("No records extracted from bulletins.")
        return None

    logger.info("Extracted %d price records from %d bulletins (%d errors).",
                len(all_records), len(pdf_files), errors)

    # Build long-form DataFrame
    df_long = pd.DataFrame(all_records)
    df_long["date"] = pd.to_datetime(df_long["date"])

    # Pivot to wide format: one column per instrument
    df_wide = df_long.pivot_table(
        index="date",
        columns="instrument",
        values="settlement_price",
        aggfunc="last",  # if duplicate dates, take last
    )
    df_wide.columns = ["omip_" + c.replace("-", "_") for c in df_wide.columns]
    df_wide.sort_index(inplace=True)
    df_wide.index.name = "date"

    logger.info("OMIP bulletin dataset: %d dates × %d instruments.", *df_wide.shape)
    return df_wide


def _parse_single_bulletin(pdf_path: Path, pdfplumber, re) -> list[dict]:
    """Parse FTB settlement prices from a single OMIP bulletin PDF.

    Only reads the first 2 pages (where FTB data lives) for speed.
    Returns list of dicts with keys: date, instrument, settlement_price, change.
    """
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages[:2]:  # FTB section is always on pages 1-2
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # Extract bulletin date from header: "2023/01/02"
    date_match = re.search(r"(\d{4}/\d{2}/\d{2})", full_text)
    if not date_match:
        return []
    bulletin_date = date_match.group(1).replace("/", "-")

    # Extract only section 1.1 (FTB) — stop at section 1.2 (FPB) or 1.3
    ftb_section = re.split(r"1\.[23]\s+MIBEL", full_text)[0]

    records: list[dict] = []
    for line in ftb_section.split("\n"):
        line = line.strip()
        if not line.startswith("FTB "):
            continue
        # Skip daily (D), weekend (WE), weekly (Wk), and PPA instruments
        if re.match(r"FTB\s+(D|WE|Wk)\s", line) or "PPA" in line:
            continue

        # Match instrument: 'M Feb-23', 'Q2-23', 'Q3-23', 'YR-24'
        inst_match = re.match(r"FTB\s+(M\s+\w+-\d{2}|Q\d-\d{2}|YR-\d{2})", line)
        if not inst_match:
            continue

        instrument = inst_match.group(1).strip()
        rest = line[inst_match.end():]

        # Extract comma-decimal numbers (European format: 89,50)
        # This skips pure integers (like nominal MWh 2184, or open interest 620)
        nums = [float(n.replace(",", ".")) for n in re.findall(r"-?[\d]+,[\d]+", rest)]
        if not nums:
            continue

        # Settlement price is always the second-to-last comma-decimal number;
        # the last one is the daily change.  If only one number, it IS the
        # settlement (change was "n.a.").
        if len(nums) >= 2:
            settlement = nums[-2]
            change = nums[-1]
        else:
            settlement = nums[0]
            change = None

        if settlement <= 0:
            continue

        records.append({
            "date": bulletin_date,
            "instrument": instrument,
            "settlement_price": settlement,
            "change": change,
        })

    return records


def _download_omip_commodity_data() -> Optional[pd.DataFrame]:
    """Download OMIP settlement prices using the commodity_data library.

    Requires:
      pip install commodity-data   (from GitHub if not on PyPI)
      A running ong_tsdb instance with config at
      ~/.config/ongpi/commodity_data.yml
    """
    try:
        from commodity_data.downloaders import OmipDownloader
    except ImportError:
        logger.info("commodity_data not installed — skipping OmipDownloader.")
        return None

    logger.info("Using commodity_data OmipDownloader …")
    omip = OmipDownloader()
    omip.download(config.START_DATE)

    df = omip.settlement_df
    if df is None or df.empty:
        logger.warning("OmipDownloader returned no data.")
        return None

    try:
        settle = omip.settle_xs(market="Omip", commodity="Power", offset=1)
        if settle is not None and not settle.empty:
            df = settle
    except Exception as exc:
        logger.warning("settle_xs failed, using raw settlement_df: %s", exc)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

    df.index.name = "date"
    logger.info("commodity_data returned %d rows.", len(df))
    return df


def manual_load_omip() -> Optional[pd.DataFrame]:
    """Load OMIP data from a user-placed Excel file at data/raw/omip_manual.xlsx."""
    path = config.RAW_FILES["omip_manual"]
    if not path.exists():
        logger.warning("Manual OMIP file not found at %s", path)
        return None
    logger.info("Loading manual OMIP file from %s", path)
    try:
        df = pd.read_excel(path, engine="openpyxl")
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as exc:
        logger.error("Failed to read manual OMIP file: %s", exc)
        return None


# ===================================================================
# B) OMIE Day-Ahead Spot Prices
# ===================================================================

def collect_omie_spot() -> None:
    """Download OMIE day-ahead spot prices (marginalpdbc files)."""
    path = config.RAW_FILES["omie_spot"]
    if _is_fresh(path):
        logger.info("OMIE spot file is fresh — skipping.")
        return

    logger.info("Collecting OMIE day-ahead spot prices …")

    try:
        df = _download_omie_bulk()
        if df is not None and not df.empty:
            _save_csv(df, path, "omie.es bulk download")
            return
    except Exception as exc:
        logger.warning("OMIE bulk download failed: %s", exc)

    logger.warning("Creating placeholder OMIE spot file with NaN values.")
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    df = pd.DataFrame({"date": idx, "omie_spot": np.nan})
    df.set_index("date", inplace=True)
    _save_csv(df, path, "placeholder — OMIE data unavailable")


def _download_omie_bulk() -> Optional[pd.DataFrame]:
    """Fetch OMIE day-ahead spot prices — weekly sampling for dense coverage.

    Downloads one file per week (every 7 days) across the full date range.
    Each file contains that day's hourly prices; we take the daily mean.
    This gives ~400+ data points for 2018-2026 in ~2 minutes.
    """
    frames: list[pd.DataFrame] = []
    start = pd.Timestamp(config.START_DATE)
    end = pd.Timestamp(config.END_DATE)
    current = start
    fetched = 0

    logger.info("OMIE bulk download: %s to %s (weekly sampling)", start.date(), end.date())

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = (
            f"https://www.omie.es/es/file-download"
            f"?parents%5B0%5D=marginalpdbc&filename=marginalpdbc_{date_str}.1"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 100:
                lines = resp.text.strip().split("\n")
                rows = []
                for line in lines[1:]:
                    parts = line.strip().split(";")
                    if len(parts) >= 5:
                        try:
                            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                            price = float(parts[4].replace(",", "."))
                            rows.append({"date": dt, "omie_spot": price})
                        except (ValueError, IndexError):
                            continue
                if rows:
                    frames.append(pd.DataFrame(rows))
                    fetched += 1
        except Exception:
            pass

        # Weekly sampling: dense enough for weekly resampling downstream
        current += timedelta(days=7)

        if fetched % 50 == 0 and fetched > 0:
            logger.info("  OMIE: fetched %d files so far ...", fetched)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date")["omie_spot"].mean().reset_index()
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    logger.info("OMIE bulk: %d daily prices from %s to %s",
                len(df), df.index.min().date(), df.index.max().date())
    return df


# ===================================================================
# C) TTF Natural Gas Futures
# ===================================================================

def collect_ttf_gas() -> None:
    """Download TTF natural gas front-month prices via yfinance."""
    path = config.RAW_FILES["ttf_gas"]
    if _is_fresh(path):
        logger.info("TTF gas file is fresh — skipping.")
        return

    logger.info("Collecting TTF natural gas futures …")
    df = _yfinance_download(
        config.YFINANCE_TICKERS["ttf_front"],
        config.START_DATE, config.END_DATE, "TTF Gas",
    )

    if df is not None and not df.empty:
        out = pd.DataFrame({"ttf_gas": df["Close"]})
        out.index.name = "date"
        _save_csv(out, path, f"yfinance {config.YFINANCE_TICKERS['ttf_front']}")
    else:
        logger.warning("TTF gas data unavailable — writing placeholder.")
        _write_placeholder(path, ["ttf_gas"], "placeholder — TTF unavailable", freq="B")


# ===================================================================
# D) EUA CO2 Allowances
# ===================================================================

def collect_eua_co2() -> None:
    """Download EUA CO2 allowance prices via yfinance.

    Primary ticker (ECF=F) has very sparse data.  Fallback chain:
      1. ECF=F  (ICE EUA front future)
      2. KRBN   (KraneShares Global Carbon ETF — strong EUA proxy)
      3. GRN    (iPath Series B Carbon ETN — secondary proxy)

    When using a proxy ETF, prices are scaled to approximate EUR/tCO2
    via a linear mapping calibrated to known EUA levels.
    """
    path = config.RAW_FILES["eua_co2"]
    if _is_fresh(path):
        logger.info("EUA CO2 file is fresh — skipping.")
        return

    logger.info("Collecting EUA CO2 allowance prices …")

    # --- Strategy 1: direct EUA futures ticker ---
    df = _yfinance_download(
        config.YFINANCE_TICKERS["eua_co2"],
        config.START_DATE, config.END_DATE, "EUA CO2 (ECF=F)",
    )
    if df is not None and len(df.dropna(subset=["Close"])) >= 30:
        out = pd.DataFrame({"eua_co2": df["Close"]})
        out.index.name = "date"
        _save_csv(out, path, "yfinance ECF=F")
        return

    # --- Strategy 2: KRBN ETF proxy (KraneShares Global Carbon) ---
    # KRBN tracks the IHS Markit Global Carbon Index, heavily weighted to EUA.
    # Scale factor: KRBN ~29 USD ≈ EUA ~68 EUR.  Use a linear scaling so
    # the model captures *movements*, not absolute levels.
    logger.info("ECF=F insufficient — trying KRBN carbon ETF proxy …")
    for proxy_ticker, label in [("KRBN", "KRBN carbon ETF"), ("GRN", "GRN carbon ETN")]:
        df = _yfinance_download(proxy_ticker, config.START_DATE, config.END_DATE, label)
        if df is not None and len(df.dropna(subset=["Close"])) >= 30:
            # Scale KRBN/GRN (USD) to approximate EUA (EUR/tCO2).
            # Rough mapping: EUA ≈ KRBN * 2.35  (calibrated mid-2025)
            scale_factor = 2.35 if proxy_ticker == "KRBN" else 2.45
            out = pd.DataFrame({"eua_co2": df["Close"] * scale_factor})
            out.index.name = "date"
            _save_csv(out, path, f"yfinance {proxy_ticker} (scaled proxy)")
            return

    logger.warning("EUA CO2 data unavailable — writing placeholder.")
    _write_placeholder(path, ["eua_co2"], "placeholder — EUA CO2 unavailable", freq="B")


# ===================================================================
# E) API2 Coal Futures
# ===================================================================

def collect_api2_coal() -> None:
    """Download API2 coal futures prices via yfinance."""
    path = config.RAW_FILES["api2_coal"]
    if _is_fresh(path):
        logger.info("API2 coal file is fresh — skipping.")
        return

    logger.info("Collecting API2 coal futures …")
    df = _yfinance_download(
        config.YFINANCE_TICKERS["api2_coal"],
        config.START_DATE, config.END_DATE, "API2 Coal",
    )

    if df is not None and not df.empty:
        out = pd.DataFrame({"api2_coal": df["Close"]})
        out.index.name = "date"
        _save_csv(out, path, f"yfinance {config.YFINANCE_TICKERS['api2_coal']}")
    else:
        logger.warning("API2 coal data unavailable — writing placeholder.")
        _write_placeholder(path, ["api2_coal"], "placeholder — API2 coal unavailable", freq="B")


# ===================================================================
# F) Iberian Hydro Reservoir Levels — Spain only (embalses.net)
#    PT hydro dropped.
# ===================================================================

def collect_hydro_reservoirs() -> None:
    """Collect Spanish hydro reservoir levels from embalses.net.

    embalses.net only exposes the *current week* as scrapeable data.
    This function:
      1. Scrapes the latest weekly total from /cuencas.php
      2. Appends it to the existing hydro CSV (building history over time)
      3. Falls back to a manual CSV if scraping fails

    For a full backfill, place a manual CSV at data/raw/hydro_reservoirs.csv
    with columns: date, hydro_es (total hm³), hydro_iberia.
    """
    path = config.RAW_FILES["hydro_reservoirs"]

    # Load existing history if present
    if path.exists():
        try:
            existing = pd.read_csv(path, comment="#", parse_dates=["date"], index_col="date")
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    logger.info("Collecting Spanish hydro reservoir levels from embalses.net …")

    new_row = _scrape_embalses_current()
    if new_row is not None:
        new_df = pd.DataFrame([new_row]).set_index("date")
        if existing.empty:
            combined = new_df
        else:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)

        combined["hydro_iberia"] = combined["hydro_es"]  # Spain only
        _save_csv(combined, path, "embalses.net weekly scrape (appended)")
    else:
        if existing.empty:
            logger.warning(
                "Hydro data unavailable. Place a manual CSV at %s with "
                "columns: date, hydro_es (hm³).",
                path,
            )
            _write_placeholder(path, ["hydro_es", "hydro_iberia"],
                               "placeholder — hydro data unavailable")
        else:
            logger.info("Scrape failed but existing hydro history preserved (%d rows).", len(existing))


def _scrape_embalses_current() -> Optional[dict]:
    """Scrape the current-week total reservoir volume from embalses.net.

    Returns a dict {"date": <Monday of this week>, "hydro_es": <hm³>}
    or None on failure.
    """
    from bs4 import BeautifulSoup

    url = "https://www.embalses.net/cuencas.php"
    logger.info("Fetching %s …", url)

    def _fetch():
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.text

    html = _retry(_fetch)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # The page shows a summary table with basin rows.  The total is usually
    # in the page text as "Agua embalsada total: XX.XXX hm³" or similar.
    # We try multiple strategies.
    text = soup.get_text(separator=" ")

    # Strategy 1: look for "total" + numeric pattern in hm³
    import re
    patterns = [
        r"total[:\s]*([\d.,]+)\s*hm",
        r"([\d.,]+)\s*hm[³3]?\s*total",
        r"agua\s+embalsada[:\s]*([\d.,]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            value_str = m.group(1).replace(".", "").replace(",", ".")
            try:
                value = float(value_str)
                today = pd.Timestamp.today()
                monday = today - pd.Timedelta(days=today.weekday())
                logger.info("embalses.net: scraped total = %.1f hm³", value)
                return {"date": monday.normalize(), "hydro_es": value}
            except ValueError:
                continue

    # Strategy 2: parse all tables and sum the volume column
    tables = soup.find_all("table")
    for tbl in tables:
        try:
            dfs = pd.read_html(StringIO(str(tbl)))
            for tdf in dfs:
                if tdf.shape[1] >= 3:
                    # Try to find a numeric column that looks like hm³ values
                    for col in tdf.columns:
                        vals = pd.to_numeric(
                            tdf[col].astype(str).str.replace(".", "", regex=False)
                                     .str.replace(",", ".", regex=False),
                            errors="coerce",
                        )
                        total = vals.sum()
                        if 5000 < total < 80000:  # plausible range for Spain (hm³)
                            today = pd.Timestamp.today()
                            monday = today - pd.Timedelta(days=today.weekday())
                            logger.info("embalses.net: table sum = %.1f hm³", total)
                            return {"date": monday.normalize(), "hydro_es": total}
        except Exception:
            continue

    logger.warning("Could not extract hydro total from embalses.net.")
    return None


# ===================================================================
# G) Wind + Solar Generation
#    PT → REN "Production Breakdown PT.xlsx" (on disk)
# ===================================================================

def collect_generation() -> None:
    """Collect wind + solar generation for PT (REN)."""
    _collect_generation_pt()


def _collect_generation_pt() -> None:
    """Extract full generation mix for Portugal from REN xlsx.

    Extracts: hydro, wind, solar, biomass, gas CCGT, gas cogen, coal,
    imports, exports, and demand.
    """
    path = config.RAW_FILES["generation_pt"]
    ren_path = config.RAW_FILES["ren_production_pt"]

    if _is_fresh(path):
        logger.info("PT generation file is fresh — skipping.")
        return

    if not ren_path.exists():
        logger.error("REN Production Breakdown PT.xlsx not found at %s", ren_path)
        _write_placeholder(path, ["wind_pt", "solar_pt", "hydro_gen_pt"],
                           "placeholder — REN PT file missing")
        return

    logger.info("Extracting PT generation from REN Data Hub xlsx …")
    try:
        raw = pd.read_excel(ren_path, engine="openpyxl")

        # Build column mapping — Portuguese names with encoding quirks
        col_map: dict[str, str] = {}
        for orig_col in raw.columns:
            low = str(orig_col).lower()
            if "hora" in low or "data" in low:
                col_map[orig_col] = "datetime"
            elif "drica" in low or "hidrica" in low or "hídrica" in low:
                col_map[orig_col] = "hydro_gen_pt"
            elif "lica" in low or "eolica" in low or "eólica" in low:
                col_map[orig_col] = "wind_pt"
            elif "solar" in low:
                col_map[orig_col] = "solar_pt"
            elif "biomassa" in low:
                col_map[orig_col] = "biomass_pt"
            elif "ciclo combinado" in low or "ciclo_combinado" in low:
                col_map[orig_col] = "gas_ccgt_pt"
            elif "cogera" in low:
                col_map[orig_col] = "gas_cogen_pt"
            elif "carv" in low:
                col_map[orig_col] = "coal_gen_pt"
            elif "importa" in low:
                col_map[orig_col] = "imports_pt"
            elif "exporta" in low:
                col_map[orig_col] = "exports_pt"
            elif "bombagem" in low:
                col_map[orig_col] = "pumping_pt"
            elif low.strip() == "consumo":
                col_map[orig_col] = "demand_pt"

        raw = raw.rename(columns=col_map)
        logger.info("REN column mapping: %s", col_map)

        needed = ["datetime", "wind_pt", "solar_pt"]
        missing = [c for c in needed if c not in raw.columns]
        if missing:
            logger.error("Missing expected columns after mapping: %s", missing)
            return

        raw["datetime"] = pd.to_datetime(raw["datetime"])
        raw = raw.set_index("datetime").sort_index()

        # Resample 15-min → weekly Monday mean
        gen_cols = [c for c in [
            "wind_pt", "solar_pt", "hydro_gen_pt",
            "biomass_pt", "gas_ccgt_pt", "gas_cogen_pt", "coal_gen_pt",
            "imports_pt", "exports_pt", "pumping_pt", "demand_pt",
        ] if c in raw.columns]
        weekly = raw[gen_cols].resample("W-MON").mean()
        weekly.index.name = "date"

        _save_csv(weekly, path, "REN Data Hub Production Breakdown PT.xlsx")

    except Exception as exc:
        logger.error("Failed to process REN PT file: %s", exc, exc_info=True)
        _write_placeholder(path, ["wind_pt", "solar_pt", "hydro_gen_pt"],
                           "placeholder — REN processing failed")



# ===================================================================
# H) German Cal/Quarter EEX Power Futures
#    Primary: investing.com CSV already on disk
#    Fallback: yfinance proxy
# ===================================================================

def collect_eex_german_futures() -> None:
    """Load German power futures from investing.com xlsx/CSV on disk."""
    path = config.RAW_FILES["eex_german_futures"]
    if _is_fresh(path):
        logger.info("EEX German futures file is fresh — skipping.")
        return

    investing_path = config.RAW_FILES["german_futures_investing"]

    # --- Strategy 1: xlsx version of investing.com data (clean, pre-processed) ---
    investing_xlsx = investing_path.with_suffix(".xlsx")
    if investing_xlsx.exists():
        logger.info("Parsing German Power Futures from investing.com xlsx …")
        try:
            raw = pd.read_excel(investing_xlsx, engine="openpyxl")
            raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
            raw["date"] = pd.to_datetime(raw["date"])
            raw["german_cal_futures"] = pd.to_numeric(
                raw["price"].astype(str).str.replace(",", ""), errors="coerce"
            )
            df = raw[["date", "german_cal_futures"]].dropna(subset=["german_cal_futures"])
            df = df.set_index("date").sort_index()
            df.index.name = "date"
            logger.info("Parsed investing.com xlsx: %d rows, %s to %s",
                        len(df), df.index.min().date(), df.index.max().date())
            _save_csv(df, path, f"investing.com xlsx ({investing_xlsx.name})")
            return
        except Exception as exc:
            logger.warning("Failed to parse investing.com xlsx: %s", exc)

    # --- Strategy 2: parse the investing.com CSV already on disk ---
    if investing_path.exists():
        logger.info("Parsing German Power Futures from investing.com CSV …")
        try:
            df = _parse_investing_csv(investing_path)
            if df is not None and not df.empty:
                _save_csv(df, path, f"investing.com CSV ({investing_path.name})")
                return
        except Exception as exc:
            logger.warning("Failed to parse investing.com CSV: %s", exc)

    # --- Strategy 3: manual xlsx ---
    manual_path = config.RAW_DIR / "eex_german_manual.xlsx"
    if manual_path.exists():
        try:
            df = pd.read_excel(manual_path, engine="openpyxl")
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            if "german_cal_futures" not in df.columns and len(df.columns) >= 1:
                df = df.rename(columns={df.columns[0]: "german_cal_futures"})
            _save_csv(df, path, "eex_german_manual.xlsx")
            return
        except Exception as exc:
            logger.warning("Failed to read EEX manual file: %s", exc)

    # --- Strategy 3: yfinance proxy ---
    df = _yfinance_download("DEFB.DE", config.START_DATE, config.END_DATE, "EEX German proxy")
    if df is not None and not df.empty:
        out = pd.DataFrame({"german_cal_futures": df["Close"]})
        out.index.name = "date"
        _save_csv(out, path, "yfinance EEX proxy")
    else:
        logger.warning("EEX German futures unavailable — writing placeholder.")
        _write_placeholder(path, ["german_cal_futures"],
                           "placeholder — EEX data unavailable", freq="B")


def _parse_investing_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Parse an investing.com-style historical data CSV.

    investing.com exports a non-standard format where each entire line is
    wrapped in outer double-quotes and internal fields use escaped
    double-double-quotes as separators:

        "Date,""Price"",""Open"",…"
        "03/24/2026,""100.10"",""100.10"",…"

    This function handles that encoding.
    """
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        raw_lines = fh.readlines()

    # Strip outer quotes from each line and replace "","" with a clean separator
    cleaned_lines: list[str] = []
    for line in raw_lines:
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]               # remove outer wrapper quotes
        # Now fields are separated by  ,""value""  — replace "" with nothing
        line = line.replace('""', '')       # remove all remaining double-quotes
        cleaned_lines.append(line)

    from io import StringIO
    raw = pd.read_csv(StringIO("\n".join(cleaned_lines)))

    # Normalise column names
    raw.columns = [c.strip().lower().replace(" ", "_").replace(".", "") for c in raw.columns]

    logger.debug("investing.com CSV columns after cleanup: %s", list(raw.columns))

    if "date" not in raw.columns or "price" not in raw.columns:
        logger.warning("investing.com CSV missing 'date' or 'price' columns: %s",
                       list(raw.columns))
        return None

    raw["date"] = pd.to_datetime(raw["date"], format="mixed", dayfirst=False)
    raw["german_cal_futures"] = pd.to_numeric(
        raw["price"].astype(str).str.replace(",", ""), errors="coerce"
    )

    df = raw[["date", "german_cal_futures"]].dropna(subset=["german_cal_futures"])
    df = df.set_index("date").sort_index()
    df.index.name = "date"

    logger.info("Parsed investing.com CSV: %d rows, %s to %s",
                len(df), df.index.min().date(), df.index.max().date())
    return df


# ===================================================================
# I) EUR/USD Exchange Rate
# ===================================================================

def collect_eurusd() -> None:
    """Download EUR/USD exchange rate via yfinance."""
    path = config.RAW_FILES["eurusd"]
    if _is_fresh(path):
        logger.info("EUR/USD file is fresh — skipping.")
        return

    logger.info("Collecting EUR/USD exchange rate …")
    df = _yfinance_download(
        config.YFINANCE_TICKERS["eurusd"],
        config.START_DATE, config.END_DATE, "EUR/USD",
    )

    if df is not None and not df.empty:
        out = pd.DataFrame({"eurusd": df["Close"]})
        out.index.name = "date"
        _save_csv(out, path, f"yfinance {config.YFINANCE_TICKERS['eurusd']}")
    else:
        logger.warning("EUR/USD data unavailable — writing placeholder.")
        _write_placeholder(path, ["eurusd"], "placeholder — EURUSD unavailable", freq="B")


# ===================================================================
# J) Electricity Demand
#    PT → REN xlsx
# ===================================================================

def collect_demand() -> None:
    """Collect electricity demand for PT (REN)."""
    _collect_demand_pt()


def _collect_demand_pt() -> None:
    """Extract Portuguese demand from the REN xlsx."""
    path = config.RAW_FILES["demand_pt"]
    ren_path = config.RAW_FILES["ren_production_pt"]

    if _is_fresh(path):
        logger.info("PT demand file is fresh — skipping.")
        return

    if not ren_path.exists():
        logger.warning("REN xlsx missing — writing PT demand placeholder.")
        _write_placeholder(path, ["demand_pt"], "placeholder — REN file missing")
        return

    logger.info("Extracting PT demand from REN Data Hub xlsx …")
    try:
        raw = pd.read_excel(ren_path, engine="openpyxl")

        # Map columns — only the standalone "Consumo" column is total demand;
        # "Consumo Baterias" is a different metric and must be excluded.
        col_map: dict[str, str] = {}
        for orig_col in raw.columns:
            low = str(orig_col).lower()
            if "hora" in low or "data" in low:
                col_map[orig_col] = "datetime"
            elif low.strip() == "consumo":
                col_map[orig_col] = "demand_pt"

        raw = raw.rename(columns=col_map)
        if "datetime" not in raw.columns or "demand_pt" not in raw.columns:
            logger.error("Could not map datetime/demand columns from REN file.")
            _write_placeholder(path, ["demand_pt"], "placeholder — REN mapping failed")
            return

        raw["datetime"] = pd.to_datetime(raw["datetime"])
        raw = raw.set_index("datetime").sort_index()

        weekly = raw[["demand_pt"]].resample("W-MON").mean()
        weekly.index.name = "date"
        _save_csv(weekly, path, "REN Data Hub (Consumo)")

    except Exception as exc:
        logger.error("Failed to extract PT demand: %s", exc)
        _write_placeholder(path, ["demand_pt"], "placeholder — REN demand extraction failed")



# ===================================================================
# Master collection function
# ===================================================================

ALL_COLLECTORS: list[tuple[str, Callable]] = [
    ("OMIP Futures", collect_omip_futures),
    ("OMIE Spot", collect_omie_spot),
    ("TTF Gas", collect_ttf_gas),
    ("EUA CO2", collect_eua_co2),
    ("API2 Coal", collect_api2_coal),
    ("Hydro Reservoirs (ES)", collect_hydro_reservoirs),
    ("Generation (PT + ES)", collect_generation),
    ("EEX German Futures", collect_eex_german_futures),
    ("EUR/USD", collect_eurusd),
    ("Demand (PT + ES)", collect_demand),
]


def collect_all() -> None:
    """Run every data collector in sequence, logging status for each."""
    logger.info("=" * 60)
    logger.info("STARTING DATA COLLECTION PIPELINE")
    logger.info("=" * 60)

    for name, func in ALL_COLLECTORS:
        logger.info("--- %s ---", name)
        try:
            func()
        except Exception as exc:
            logger.error("Unhandled error in %s collector: %s", name, exc, exc_info=True)
        logger.info("")

    logger.info("=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(config.PROJECT_ROOT / ".env")
    collect_all()
