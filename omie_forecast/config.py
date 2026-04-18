"""
config.py — OMIEForecast Central Configuration
================================================
All paths, constants, feature lists, and model hyperparameters live here.
No hardcoded values in any other script.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import date

import holidays as hol
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# ENTSO-E API key
# ---------------------------------------------------------------------------
ENTSOE_API_KEY: str = os.getenv("ENTSOE_API_KEY", "")

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
FORECASTS_DIR  = PROJECT_ROOT / "outputs" / "forecasts"
CHARTS_DIR     = PROJECT_ROOT / "outputs" / "charts"
LOGS_DIR       = PROJECT_ROOT / "logs"

for _d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FORECASTS_DIR, CHARTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

LOG_FILE   = LOGS_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE   = date.today().isoformat()

# ---------------------------------------------------------------------------
# Hours
# ---------------------------------------------------------------------------
HOURS: list[int] = list(range(24))
FORECAST_HORIZON_DAYS: int = 7

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
LAG_HOURS: list[int] = [24, 48, 72, 96, 120, 144, 168]   # 1-day to 7-day same-hour lags
PREV_DAY_HOURS: list[int] = list(range(24))

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
LASSO_CV_FOLDS: int    = 5
LASSO_MAX_ITER: int    = 20_000
LASSO_N_ALPHAS: int    = 100

WALK_FORWARD_SPLITS: int   = 12
WALK_FORWARD_GAP_DAYS: int = 7

# ---------------------------------------------------------------------------
# Storm anomaly — downweight during training (extreme storms Jan-Feb 2026)
# ---------------------------------------------------------------------------
STORM_ANOMALY_START:   str   = "2026-01-23"
STORM_ANOMALY_END:     str   = "2026-02-15"
SAMPLE_WEIGHT_ANOMALY: float = 0.1
SAMPLE_WEIGHT_SPIKE:   float = 0.3

# ---------------------------------------------------------------------------
# Iberian holidays — Portugal + Spain, 2015–2028
# ---------------------------------------------------------------------------
def _build_iberian_holidays(years: range) -> set[date]:
    days: set[date] = set()
    for yr in years:
        days.update(hol.Portugal(years=yr).keys())
        days.update(hol.Spain(years=yr).keys())
    return days

IBERIAN_HOLIDAYS: set[date] = _build_iberian_holidays(range(2015, 2029))

# ---------------------------------------------------------------------------
# Feature column names (referenced by multiple scripts)
# ---------------------------------------------------------------------------
AR_FEATURES: list[str] = (
    [f"price_lag_{h}h" for h in LAG_HOURS]
    + [f"prev_day_h{h:02d}" for h in range(24)]
    + ["price_roll_mean_24h", "price_roll_std_24h",
       "price_roll_mean_168h", "price_roll_std_168h"]
)

FUNDAMENTAL_FEATURES: list[str] = [
    "wind_total", "solar_total", "load_total", "res_share",
    "ttf_gas", "eua_co2", "spark_spread",
    "hydro_anomaly", "is_weekend", "is_holiday",
    "is_summer", "is_winter", "hour_x_weekend", "month",
]

ENTSOE_FEATURES: list[str] = [
    "wind_onshore_forecast_es_mwh",
    "wind_onshore_forecast_pt_mwh",
    "wind_offshore_forecast_pt_mwh",
    "solar_forecast_es_mwh",
    "solar_forecast_pt_mwh",
    "load_forecast_es_mwh",
    "load_forecast_pt_mwh",
    "flow_es_fr_mwh",
]

ALL_FEATURES: list[str] = AR_FEATURES + FUNDAMENTAL_FEATURES + ENTSOE_FEATURES
