"""
00_download_bulletins.py -- OMIP Market Bulletin Downloader
============================================================
Downloads OMIP market bulletins from omip.pt and saves/extracts the
parte1.pdf files (which contain FTB settlement prices) into the
configured bulletin directories.

Three URL patterns are used depending on the date range:

  Period 1 (2018-01 to ~2020-01):
    https://www.omip.pt/sites/default/files/boletim_DDMMYYYY.pdf
    → single PDF (contains both parte1+parte2 content)

  Period 2 (~2020-02 to ~2021-08):
    https://www.omip.pt/sites/default/files/YYYY-MM/boletim_DDMMYYYY.pdf
    → single PDF

  Period 3 (~2021-08 to present):
    https://www.omip.pt/sites/default/files/market_bulletins/boletim_YYYYMMDD.zip
    → ZIP containing boletim_YYYYMMDD_parte1.pdf + boletim_YYYYMMDD_parte2.pdf

The script:
  - Scans existing bulletin dirs to find what's already downloaded
  - Downloads only missing dates (business days between start and end)
  - For ZIPs, extracts PDFs into the year-appropriate directory
  - For single PDFs, renames to boletim_YYYYMMDD_parte1.pdf for consistency
  - Tries all applicable URL patterns with fallbacks
  - Skips weekends and holidays gracefully (HTTP 404)

Run:
  python scripts/00_download_bulletins.py                          # last 30 days
  python scripts/00_download_bulletins.py --from-date 2018-01-02   # from specific date
  python scripts/00_download_bulletins.py --full-backfill           # all since 2018-01-02
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("download_bulletins")
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OMIP_BASE = "https://www.omip.pt/sites/default/files"
EARLIEST_DATE = date(2018, 1, 2)
REQUEST_TIMEOUT = 30
THROTTLE_SECONDS = 0.4  # polite delay between downloads


# ===================================================================
# URL pattern helpers
# ===================================================================

def _url_candidates(d: date) -> list[tuple[str, str]]:
    """Return a list of (url, format_label) candidates for a given date.

    Tries all patterns in order of likelihood based on the date.
    Each URL pattern is tried until one returns HTTP 200.
    """
    dd_mm_yyyy = d.strftime("%d%m%Y")
    yyyy_mm_dd = d.strftime("%Y%m%d")
    yyyy_mm = d.strftime("%Y-%m")

    candidates = []

    if d < date(2020, 2, 1):
        # Period 1: flat path, DDMMYYYY, single PDF
        candidates.append(
            (f"{OMIP_BASE}/boletim_{dd_mm_yyyy}.pdf", "pdf_flat")
        )
        # Fallback: try period 2 pattern
        candidates.append(
            (f"{OMIP_BASE}/{yyyy_mm}/boletim_{dd_mm_yyyy}.pdf", "pdf_dated")
        )
    elif d < date(2021, 8, 16):
        # Period 2: year-month subfolder, DDMMYYYY, single PDF
        candidates.append(
            (f"{OMIP_BASE}/{yyyy_mm}/boletim_{dd_mm_yyyy}.pdf", "pdf_dated")
        )
        # Fallback: try period 1
        candidates.append(
            (f"{OMIP_BASE}/boletim_{dd_mm_yyyy}.pdf", "pdf_flat")
        )
        # Fallback: try period 3
        candidates.append(
            (f"{OMIP_BASE}/market_bulletins/boletim_{yyyy_mm_dd}.zip", "zip")
        )
    else:
        # Period 3: market_bulletins folder, YYYYMMDD, ZIP
        candidates.append(
            (f"{OMIP_BASE}/market_bulletins/boletim_{yyyy_mm_dd}.zip", "zip")
        )
        # Fallback: try period 2
        candidates.append(
            (f"{OMIP_BASE}/{yyyy_mm}/boletim_{dd_mm_yyyy}.pdf", "pdf_dated")
        )

    return candidates


# ===================================================================
# Helpers
# ===================================================================

def _get_bulletin_dir(d: date) -> Path:
    """Return the bulletin directory for a given date's year."""
    for bdir in config.OMIP_BULLETIN_DIRS:
        if str(d.year) in bdir.name and bdir.exists():
            return bdir

    # Create new dir alongside existing bulletin dirs
    if config.OMIP_BULLETIN_DIRS:
        parent = config.OMIP_BULLETIN_DIRS[0].parent
        new_dir = parent / f"Boletins {d.year}"
        new_dir.mkdir(parents=True, exist_ok=True)
        if new_dir not in config.OMIP_BULLETIN_DIRS:
            config.OMIP_BULLETIN_DIRS.append(new_dir)
        return new_dir

    # Last resort: use raw data dir
    fallback = config.RAW_DIR / f"bulletins_{d.year}"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _existing_dates() -> set[date]:
    """Scan all bulletin directories and return dates already downloaded."""
    existing: list[date] = []
    for bdir in config.OMIP_BULLETIN_DIRS:
        if not bdir.exists():
            continue
        # Match both formats: boletim_YYYYMMDD_parte1.pdf and boletim_YYYYMMDD.pdf
        for pdf in bdir.glob("boletim_*parte1.pdf"):
            name = pdf.stem  # boletim_20230102_parte1
            parts = name.split("_")
            if len(parts) >= 2:
                try:
                    d = datetime.strptime(parts[1], "%Y%m%d").date()
                    existing.append(d)
                except ValueError:
                    continue
    return set(existing)


def _business_days(start: date, end: date) -> list[date]:
    """Generate all weekday dates between start and end (inclusive)."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0 ... Friday=4
            days.append(current)
        current += timedelta(days=1)
    return days


def _save_single_pdf(content: bytes, d: date, target_dir: Path) -> bool:
    """Save a single-file bulletin PDF as boletim_YYYYMMDD_parte1.pdf."""
    date_str = d.strftime("%Y%m%d")
    out_path = target_dir / f"boletim_{date_str}_parte1.pdf"
    out_path.write_bytes(content)
    return True


def _extract_zip(content: bytes, d: date, target_dir: Path) -> bool:
    """Extract bulletin ZIP, saving PDFs with standardised names."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
        extracted = 0
        for member in zf.namelist():
            if member.endswith(".pdf"):
                zf.extract(member, target_dir)
                extracted += 1
        return extracted > 0
    except zipfile.BadZipFile:
        return False


def download_bulletin(d: date, target_dir: Path) -> bool:
    """Download a single bulletin for date d, trying all URL patterns.

    Returns True if successful, False otherwise.
    """
    date_str = d.strftime("%Y%m%d")
    candidates = _url_candidates(d)

    for url, fmt in candidates:
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.debug("  %s network error: %s", date_str, exc)
            continue

        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            logger.debug("  %s -> HTTP %d from %s", date_str, resp.status_code, fmt)
            continue

        # Got a 200 — process the content
        if fmt == "zip":
            ok = _extract_zip(resp.content, d, target_dir)
        else:
            ok = _save_single_pdf(resp.content, d, target_dir)

        if ok:
            logger.info("  %s -> OK [%s] (%d KB -> %s)",
                        date_str, fmt, len(resp.content) // 1024, target_dir.name)
            return True

    # All patterns returned 404 — likely a holiday
    logger.debug("  %s -> not found (holiday/non-trading)", date_str)
    return False


# ===================================================================
# Main
# ===================================================================

def download_bulletins(from_date: date | None = None, to_date: date | None = None) -> None:
    """Download all missing OMIP bulletins between from_date and to_date."""
    if to_date is None:
        to_date = date.today()
    if from_date is None:
        from_date = to_date - timedelta(days=30)

    if from_date < EARLIEST_DATE:
        from_date = EARLIEST_DATE

    logger.info("=" * 60)
    logger.info("OMIP BULLETIN DOWNLOADER")
    logger.info("  Range: %s to %s", from_date, to_date)
    logger.info("=" * 60)

    existing = _existing_dates()
    logger.info("Already have %d bulletins on disk.", len(existing))

    all_days = _business_days(from_date, to_date)
    missing = [d for d in all_days if d not in existing]
    logger.info("Business days in range: %d, missing: %d", len(all_days), len(missing))

    if not missing:
        logger.info("All bulletins are up to date. Nothing to download.")
        return

    downloaded = 0
    skipped = 0

    for i, d in enumerate(missing):
        target_dir = _get_bulletin_dir(d)
        success = download_bulletin(d, target_dir)

        if success:
            downloaded += 1
        else:
            skipped += 1

        # Progress update every 50 files
        if (i + 1) % 50 == 0:
            logger.info("  Progress: %d/%d checked, %d downloaded, %d skipped ...",
                        i + 1, len(missing), downloaded, skipped)

        # Polite throttle
        if i < len(missing) - 1:
            time.sleep(THROTTLE_SECONDS)

    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("  Downloaded: %d new bulletins", downloaded)
    logger.info("  Skipped (holidays/404): %d", skipped)
    logger.info("  Date range covered: %s to %s", from_date, to_date)
    logger.info("=" * 60)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download OMIP market bulletins from omip.pt"
    )
    parser.add_argument(
        "--from-date", type=str, default=None,
        help="Start date (YYYY-MM-DD). Default: 30 days ago.",
    )
    parser.add_argument(
        "--to-date", type=str, default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--full-backfill", action="store_true",
        help="Download ALL bulletins since 2018-01-02.",
    )
    args = parser.parse_args()

    from_date = None
    to_date = None

    if args.full_backfill:
        from_date = EARLIEST_DATE
    elif args.from_date:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()

    if args.to_date:
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    download_bulletins(from_date=from_date, to_date=to_date)


if __name__ == "__main__":
    main()
