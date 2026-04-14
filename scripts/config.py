"""
OMIP Futures Price Forecasting System — Configuration
=====================================================
Central configuration for all paths, parameters, contract identifiers,
feature lists, model hyperparameters, and validation settings.

All scripts import from this module. No hardcoded values elsewhere.
"""

from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# Project root (this file sits at project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
CHARTS_DIR = OUTPUTS_DIR / "charts"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure directories exist at import time
for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, FORECASTS_DIR, CHARTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw data file paths
# ---------------------------------------------------------------------------
RAW_FILES = {
    "omip_futures": RAW_DIR / "omip_futures_historical.csv",
    "omip_manual": RAW_DIR / "omip_manual.xlsx",
    "omie_spot": RAW_DIR / "omie_spot_daily.csv",
    "ttf_gas": RAW_DIR / "ttf_gas.csv",
    "eua_co2": RAW_DIR / "eua_co2.csv",
    "api2_coal": RAW_DIR / "api2_coal.csv",
    "hydro_reservoirs": RAW_DIR / "hydro_reservoirs.csv",
    "ren_production_pt": RAW_DIR / "Production Breakdown PT.xlsx",
    "generation_pt": RAW_DIR / "generation_pt.csv",
    "eex_german_futures": RAW_DIR / "eex_german_futures.csv",
    "german_futures_investing": RAW_DIR / "German Power Futures Historical Data.csv",
    "eurusd": RAW_DIR / "eurusd.csv",
    "demand_pt": RAW_DIR / "demand_pt.csv",
    "news_sentiment": RAW_DIR / "news_sentiment.csv",
}

# ---------------------------------------------------------------------------
# OMIP Bulletin PDF directories (parte1 files contain settlement prices)
# ---------------------------------------------------------------------------
_OMIP_BASE_DIR = Path(r"C:\Users\User\NOSSA ENERGIA, COMERCIO ENERGIA,LDA\Estratégia - Documentos\9.OMIP")
OMIP_BULLETIN_DIRS: list[Path] = [
    _OMIP_BASE_DIR / "Boletins 2018",
    _OMIP_BASE_DIR / "Boletins 2019",
    _OMIP_BASE_DIR / "Boletins 2020",
    _OMIP_BASE_DIR / "Boletins 2021",
    _OMIP_BASE_DIR / "Boletins 2022",
    _OMIP_BASE_DIR / "3. Boletins 2023",
    _OMIP_BASE_DIR / "2. Boletins 2024",
    _OMIP_BASE_DIR / "1. Boletins 2025",
    _OMIP_BASE_DIR / "Boletins 2026",
]

# Processed data file paths
MASTER_DATASET = PROCESSED_DIR / "master_dataset.csv"
DATA_QUALITY_REPORT = PROCESSED_DIR / "data_quality_report.txt"
COINTEGRATION_REPORT = PROCESSED_DIR / "cointegration_report.txt"

# ---------------------------------------------------------------------------
# OMIP Contract identifiers
# ---------------------------------------------------------------------------
CONTRACTS: list[str] = ["Q3_26", "Q4_26", "Q1_27", "Q2_27", "YR27", "YR28"]

SHORT_HORIZON_CONTRACTS: list[str] = ["Q3_26", "Q4_26", "Q1_27", "Q2_27"]
LONG_HORIZON_CONTRACTS: list[str] = ["YR27", "YR28"]

# Delivery start dates — last trading day is 2 business days before this
CONTRACT_DELIVERY_START: dict[str, date] = {
    "Q3_26": date(2026, 7, 1),
    "Q4_26": date(2026, 10, 1),
    "Q1_27": date(2027, 1, 1),
    "Q2_27": date(2027, 4, 1),
    "YR27":  date(2027, 1, 1),
    "YR28":  date(2028, 1, 1),
}

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
START_DATE: str = "2018-01-01"  # earliest OMIP bulletin: 2018-01-02
END_DATE: str = date.today().isoformat()

# ---------------------------------------------------------------------------
# Cache freshness (skip re-download if file is newer than this many days)
# ---------------------------------------------------------------------------
CACHE_FRESHNESS_DAYS: int = 7

# Hydro reservoir: Spain only (from embalses.net).  PT hydro dropped.
HYDRO_WEIGHTS: dict[str, float] = {"ES": 1.00}

# ---------------------------------------------------------------------------
# REN Data Hub — Portuguese production breakdown
# Column name mapping (original Portuguese → internal names)
# ---------------------------------------------------------------------------
REN_COLUMN_MAP: dict[str, str] = {
    "Data e Hora": "datetime",
    "Hídrica": "hydro_gen_pt",
    "Eólica": "wind_pt",
    "Solar": "solar_pt",
    "Consumo": "demand_pt",
}

# ---------------------------------------------------------------------------
# Fuel → electricity conversion constants
# ---------------------------------------------------------------------------
CCGT_EFFICIENCY: float = 0.52
CCGT_EMISSION_FACTOR: float = 0.202   # tCO2/MWh gas
COAL_EFFICIENCY: float = 0.38
COAL_EMISSION_FACTOR: float = 0.34    # tCO2/MWh coal
TTF_MWH_FACTOR: float = 293.07       # kWh per therm → MWh divisor
COAL_MWH_FACTOR: float = 7.14        # GJ per tonne → MWh divisor

# ---------------------------------------------------------------------------
# Storm anomaly exclusion period
# ---------------------------------------------------------------------------
STORM_ANOMALY_START: str = "2026-01-23"
STORM_ANOMALY_END: str = "2026-02-15"
STORM_ANOMALY_WEIGHT: float = 0.1
SPOT_ANOMALY_PERCENTILE: float = 0.02  # 2nd percentile threshold

# ---------------------------------------------------------------------------
# Data frequency — "B" for business-daily, "W-MON" for weekly Monday
# ---------------------------------------------------------------------------
DATA_FREQ: str = "B"  # business-daily master dataset

# ---------------------------------------------------------------------------
# Feature lists  (lag/momentum names now use "d" suffix for daily)
# ---------------------------------------------------------------------------
LONG_FEATURES: list[str] = [
    "gas_spark_spread",
    "coal_dark_spread",
    "eua_co2",
    "hydro_anomaly_pct",
    "omip_lag_20d",
    "omip_lag_65d",
    "german_cal_futures",
    "german_momentum_20d",
    "german_momentum_65d",
    "german_omip_spread",
    "res_penetration",
    "eurusd",
    "ect_term",
    "is_q1",
    "is_q3",
    # Term structure
    "term_yr_spread",
    # PT generation mix (REN)
    "thermal_share",
    "gas_ccgt_share",
    "net_import_share",
    # News sentiment features
    "news_sentiment",
    "news_sentiment_ma20d",
]

SHORT_FEATURES: list[str] = [
    "omip_lag_1d",
    "omip_lag_5d",
    "omip_lag_10d",
    "ttf_vol_20d",
    "omip_momentum_20d",
    "risk_premium",
    "hydro_anomaly",
    "curve_slope_yr",
    # Cross-contract term structure
    "term_q_spread_1_2",
    "term_q_yr_spread",
    # Spot price momentum
    "spot_momentum_20d",
    "spot_ma_ratio",
    # Day-of-week effect
    "dow",
    # News
    "news_bullish_pct",
    "news_volume",
]

# Full feature set for quarterly (short-horizon) contracts
SHORT_HORIZON_FEATURES: list[str] = LONG_FEATURES + SHORT_FEATURES

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
XGBOOST_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 40,
    "eval_metric": "mae",
    "random_state": 42,
}

RIDGE_ALPHA: float = 1.0  # legacy — LassoCV now auto-tunes alpha

# Ensemble weights — (ridge_weight, xgb_weight)
ENSEMBLE_WEIGHTS_QUARTERLY: tuple[float, float] = (0.40, 0.60)
ENSEMBLE_WEIGHTS_ANNUAL: tuple[float, float] = (0.45, 0.55)

# Quantile regression alphas for prediction intervals
QUANTILE_LOWER: float = 0.10
QUANTILE_UPPER: float = 0.90

# ---------------------------------------------------------------------------
# Walk-forward validation settings
# ---------------------------------------------------------------------------
WF_N_SPLITS: int = 12
WF_GAP_WEEKS: int = 4        # legacy (unused now)
WF_GAP_DAYS: int = 20        # ~4 weeks of business days

# ---------------------------------------------------------------------------
# Forecast horizons (days ahead)
# ---------------------------------------------------------------------------
FORECAST_HORIZONS: list[int] = [30, 60, 90]

# Directional signal thresholds (absolute %)
SIGNAL_THRESHOLD_PCT: float = 3.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# yfinance tickers
# ---------------------------------------------------------------------------
YFINANCE_TICKERS: dict[str, str] = {
    "ttf_front": "TTF=F",
    "eua_co2": "ECF=F",
    "eua_co2_proxy": "KRBN",       # KraneShares Global Carbon ETF (EUA proxy)
    "api2_coal": "MTF=F",
    "eurusd": "EURUSD=X",
}

# ---------------------------------------------------------------------------
# Retry settings for API calls
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 2.0  # seconds; actual wait = base ** attempt
