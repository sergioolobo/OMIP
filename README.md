# OMIP Futures Price Forecasting System

Production-quality forecasting system for **OMIP forward contracts** on the Iberian electricity market (MIBEL). Predicts settlement / mark-to-market prices for quarterly and annual contracts (Q3/26, Q4/26, Q1/27, Q2/27, YR27, YR28).

## Dashboard Trading NOSSA Energia (main tab)

The Streamlit app (`scripts/07_dashboard.py`) opens on **Dashboard Trading NOSSA Energia**; the OMIP and
OMIE price forecasts are the two secondary tabs. The trading tab shows, for any period:

| Sub-tab | What it answers |
|---------|-----------------|
| Resumo | Bought vs. consumed, net and absolute imbalance, cost of imbalances, regulation charges, CGS and total energy cost per MWh — each against the previous period — plus conclusions written from the data |
| Compras | One line per day (purchase, consumption, imbalance, prices, costs), a 15-minute view of any day (offer sent, final position, verified consumption, prices) and the offers that did not match |
| Desvios | Excess and deficit, what each side cost, bias by hour and weekday, day × hour heat map |
| Preços | OMIE day-ahead price, imbalance prices and spreads, hourly profile, the price we actually paid |
| CGS | Imbalances + regulation charges per MWh consumed, against a reference value, by day/week/month |
| Mercado | Imbalance cost per MWh of every market agent (public REN data) and where NOSSA Energia ranks |
| Dados | Coverage of each source and the warnings raised while reading it |

Code: `scripts/trading_tab.py` (UI) and `scripts/trading_dados.py` (loading + maths, no Streamlit).
Test: `python scripts/teste_trading.py` (hand-made numbers; must print `0 falha(s)`).

### Where the data comes from

The data is **not** produced by this repo's pipeline. It is generated on the company PC by
`trading_dados.py` in the «Compra OMIE» project, from the REN settlement documents pasted into the
«Cálculo do CGS» workbooks (final position, verified quantity, prices, charges — per quarter-hour since
30/12/2024), the day-ahead offers sent to OMIE, and public OMIE/REN data:

```bash
python trading_dados.py               # writes trading/dados/*.csv (plain; git-ignored; local use)
python trading_dados.py --cifrar      # also writes trading/pub/*.bin (encrypted, one file per month)
python trading_dados.py --publicar    # encrypts, then commits and pushes trading/pub only
```

**This repository is public and so is the server**, so company data only ever reaches git encrypted
(`trading/pub/`, Fernet with a key derived from a password by PBKDF2-HMAC-SHA256). The dashboard asks
for that password before showing the trading tab; the forecast tabs stay open. `trading/dados/` (plain
CSV) is git-ignored and exists only on the PC that generates it — when it is present the dashboard
reads it and asks for nothing. The password is typed in a terminal (`--palavra-passe`), never passed
as an argument, and is stored on that PC with Windows DPAPI.

## Use Cases

1. **Price level forecasting** — What will Q4/26 settle at in 30/60/90 days?
2. **Directional signal** — Should I close my position now or wait?

## Architecture

Two-layer ensemble per contract:

| Layer | Model | Role |
|-------|-------|------|
| 1 | Ridge Regression (scaled) | Captures fundamental linear relationships |
| 2 | XGBoost on Ridge residuals | Captures non-linear patterns and interactions |
| 3 | Weighted average | Quarterly: 40% Ridge + 60% XGB · Annual: 65% Ridge + 35% XGB |

Quantile XGBoost models provide 80% prediction intervals.

## Setup

```bash
# 1. Clone / navigate to the project
cd OMIPForecast

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file with API keys
echo "ENTSOE_API_KEY=your_key_here" > .env
echo "ESIOS_API_KEY=your_key_here" >> .env
```

### Environment Variables

| Variable | Required For | How to Obtain |
|----------|-------------|---------------|
| `ENTSOE_API_KEY` | Wind/solar generation, demand data | Register at [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) |
| `ESIOS_API_KEY` | Spanish hydro reservoir levels | Request from [REE/ESIOS](https://api.esios.ree.es/) |

## Running the Pipeline

Execute each script **in order** from the project root:

```bash
# Step 1 — Download all raw data
python scripts/01_collect_data.py

# Step 2 — Build feature dataset
python scripts/02_build_features.py

# Step 3 — Cointegration & stationarity diagnostics
python scripts/03_cointegration_check.py

# Step 4 — Train models (walk-forward validation)
python scripts/04_train_models.py

# Step 5 — Generate evaluation charts and metrics
python scripts/05_evaluate_models.py

# Step 6 — Generate forward-looking forecasts
python scripts/06_forecast.py

# Step 7 — Launch the interactive dashboard
streamlit run scripts/07_dashboard.py
```

The pipeline is **resumable** — each step saves intermediate outputs so you can re-run from any point.

## Manual Data Sources

Some data sources require manual CSV/Excel placement:

| File | Location | Notes |
|------|----------|-------|
| OMIP settlement prices | `data/raw/omip_manual.xlsx` | Download from [omip.pt](https://www.omip.pt/en/dados-mercado), or use `commodity_data` library (see below). Columns: `date`, `contract_name`, `settlement_price`, `volume` |
| PT production breakdown | `data/raw/Production Breakdown PT.xlsx` | Download from [REN Data Hub](https://datahub.ren.pt/). 15-min resolution. Provides wind, solar, hydro generation, and demand for Portugal. |
| German power futures | `data/raw/German Power Futures Historical Data.csv` | Download from [investing.com](https://www.investing.com/commodities/german-power-baseload-futures-historical-data). Investing.com CSV format. |
| Spanish hydro reservoirs | `data/raw/hydro_reservoirs.csv` | Auto-appended weekly from [embalses.net](https://www.embalses.net/). For full backfill, place a manual CSV with columns: `date`, `hydro_es` (hm³). |

### OMIP data via commodity_data (optional)

The [`commodity_data`](https://github.com/Oneirag/commodity_data) library can automate OMIP settlement price downloads. It requires a running `ong_tsdb` database:

```bash
# 1. Install from GitHub
pip install git+https://github.com/Oneirag/ong_tsdb.git
pip install git+https://github.com/Oneirag/commodity_data.git

# 2. Create config file at ~/.config/ongpi/commodity_data.yml
# 3. Start the ong_tsdb server
# 4. Run 01_collect_data.py — it will use OmipDownloader automatically
```

If `commodity_data` is not installed, the pipeline falls back to `omip_manual.xlsx`.

## Configuration

All parameters are centralised in `config.py`:

- **CONTRACTS** — list of active OMIP contract identifiers
- **XGBOOST_PARAMS** — hyperparameters for the XGBoost layer
- **ENSEMBLE_WEIGHTS** — Ridge vs XGB blend per horizon
- **WF_N_SPLITS / WF_GAP_WEEKS** — walk-forward validation settings
- **STORM_ANOMALY_START/END** — anomaly exclusion period (down-weighted, not removed)
- **SIGNAL_THRESHOLD_PCT** — directional signal threshold (default 3%)
- **CACHE_FRESHNESS_DAYS** — skip re-downloads if files are newer than this

## Project Structure

```
OMIPForecast/
├── config.py                   # Central configuration
├── requirements.txt
├── data/
│   ├── raw/                    # Downloaded raw files
│   └── processed/              # Cleaned master dataset
├── models/                     # Saved .pkl model files
├── outputs/
│   ├── forecasts/              # Forecast CSVs
│   └── charts/                 # PNG charts
├── scripts/
│   ├── 01_collect_data.py
│   ├── 02_build_features.py
│   ├── 03_cointegration_check.py
│   ├── 04_train_models.py
│   ├── 05_evaluate_models.py
│   ├── 06_forecast.py
│   └── 07_dashboard.py
└── logs/
    └── pipeline.log
```

## Key Drivers Modelled

- **Fuel costs**: TTF natural gas, API2 coal, EUA CO2 allowances
- **Hydro**: Spanish reservoir anomaly (embalses.net)
- **Renewables**: Wind + solar penetration ratio (PT: REN Data Hub, ES: ENTSO-E)
- **Demand**: Weekly Iberian electricity load (PT: REN Data Hub, ES: ENTSO-E)
- **Anchors**: German EEX Cal/Quarter futures, EUR/USD FX
- **Technical**: Autoregressive lags, momentum, volatility, curve slope, risk premium
