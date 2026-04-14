"""
07_dashboard.py — Streamlit Dashboard for OMIP Futures Forecasts
=================================================================
Interactive dashboard with three tabs: Forecasts, Fundamentals, and
Model Performance.  Uses Plotly for all vizualisations.

Run with:  streamlit run scripts/07_dashboard.py
Inputs:  data/processed/master_dataset.csv
         outputs/forecasts/omip_forecast_*.csv
         outputs/forecasts/walkforward_results.csv
         models/*.pkl
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# ===================================================================
# Branding colours
# ===================================================================
ORANGE = "#E8611A"
ORANGE_LIGHT = "#F28C52"
ORANGE_PALE = "#FFF3EB"
WHITE = "#FFFFFF"
DARK_TEXT = "#2D2D2D"
GRAY_BG = "#F7F7F7"

# ===================================================================
# Page config + custom CSS
# ===================================================================
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"
_FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "favicon.png"

st.set_page_config(
    page_title="OMIP Futures Forecast — Nossa Energia",
    layout="wide",
    page_icon=str(_FAVICON_PATH) if _FAVICON_PATH.exists() else None,
)

# Inject orange/white theme via CSS
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {WHITE};
    }}

    /* Header bar */
    header[data-testid="stHeader"] {{
        background-color: {ORANGE} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {ORANGE_PALE} !important;
        border-right: 3px solid {ORANGE};
    }}
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {ORANGE} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid {ORANGE};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {WHITE};
        color: {DARK_TEXT};
        border-radius: 8px 8px 0 0;
        border: 1px solid #ddd;
        padding: 8px 20px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ORANGE} !important;
        color: {WHITE} !important;
        border: 1px solid {ORANGE};
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {ORANGE_PALE};
        border: 1px solid {ORANGE_LIGHT};
        border-radius: 8px;
        padding: 12px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {ORANGE} !important;
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {ORANGE} !important;
    }}

    /* DataFrame styling */
    .stDataFrame {{
        border: 1px solid {ORANGE_LIGHT};
        border-radius: 8px;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {ORANGE};
        color: {WHITE};
        border: none;
        border-radius: 6px;
    }}
    .stButton > button:hover {{
        background-color: {ORANGE_LIGHT};
        color: {WHITE};
    }}

    /* Multiselect tags */
    span[data-baseweb="tag"] {{
        background-color: {ORANGE} !important;
    }}

    /* Footer */
    .footer-text {{
        color: {ORANGE};
        font-size: 0.85em;
    }}
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Logo + Title header
# ===================================================================
_header_cols = st.columns([2, 6, 2])
with _header_cols[0]:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=200)
with _header_cols[1]:
    st.markdown(
        f'<h1 style="margin-bottom:0; color:{ORANGE} !important;">'
        'OMIP Futures Forecast Dashboard</h1>'
        f'<p style="color:{ORANGE_LIGHT}; margin-top:0;">'
        'Iberian Electricity Market Intelligence</p>',
        unsafe_allow_html=True,
    )
with _header_cols[2]:
    st.write("")  # spacer

st.markdown("---")

# ===================================================================
# Plotly layout defaults (orange theme)
# ===================================================================
_PLOTLY_LAYOUT = dict(
    paper_bgcolor=WHITE,
    plot_bgcolor=GRAY_BG,
    font=dict(color=DARK_TEXT),
    title_font=dict(color=ORANGE, size=16),
    xaxis=dict(gridcolor="#E0E0E0"),
    yaxis=dict(gridcolor="#E0E0E0"),
    colorway=[ORANGE, "#1A73E8", ORANGE_LIGHT, "#34A853", "#EA4335", "#9C27B0"],
)


# ===================================================================
# Data loading (cached)
# ===================================================================

@st.cache_data(ttl=300)
def load_master() -> pd.DataFrame:
    if not config.MASTER_DATASET.exists():
        st.error(f"Master dataset not found at {config.MASTER_DATASET}")
        return pd.DataFrame()
    return pd.read_csv(config.MASTER_DATASET, index_col="date", parse_dates=True)


@st.cache_data(ttl=60)
def load_latest_forecast() -> pd.DataFrame:
    forecast_files = sorted(config.FORECASTS_DIR.glob("omip_forecast_*.csv"))
    if not forecast_files:
        return pd.DataFrame()
    return pd.read_csv(forecast_files[-1])


@st.cache_data(ttl=300)
def load_walkforward() -> pd.DataFrame:
    path = config.FORECASTS_DIR / "walkforward_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model_bundle(contract: str) -> dict | None:
    path = config.MODELS_DIR / f"model_{contract}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


# ===================================================================
# Sidebar controls
# ===================================================================

with st.sidebar:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=180)
    st.markdown(f'<h2 style="color:{ORANGE};">Controls</h2>', unsafe_allow_html=True)

master = load_master()
forecast_df = load_latest_forecast()
wf_results = load_walkforward()

available_contracts = config.CONTRACTS
selected_contracts = st.sidebar.multiselect(
    "Contracts",
    options=available_contracts,
    default=available_contracts[:3],
)

selected_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=config.FORECAST_HORIZONS,
    format_func=lambda x: f"{x} days",
)

if not master.empty:
    min_date = master.index.min().date()
    max_date = master.index.max().date()
    date_range = st.sidebar.slider(
        "Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
else:
    date_range = None

# ===================================================================
# Tab 1 — Forecasts
# ===================================================================

tab1, tab2, tab3 = st.tabs(["Forecasts", "Fundamentals", "Model Performance"])

with tab1:
    st.header("Price Forecasts")

    if master.empty:
        st.warning("No master dataset available. Run the pipeline first.")
    else:
        filtered = master.copy()
        if date_range:
            filtered = filtered.loc[str(date_range[0]):str(date_range[1])]

        for contract in selected_contracts:
            bundle = load_model_bundle(contract)
            target_col = bundle["target_col"] if bundle else "omip_yr1"

            if target_col not in filtered.columns:
                st.info(f"No price data for {contract} (column: {target_col})")
                continue

            fig = go.Figure()

            # Historical price
            fig.add_trace(go.Scatter(
                x=filtered.index,
                y=filtered[target_col],
                name="Historical",
                line=dict(color=ORANGE, width=2),
            ))

            # Add forecast point + CI if available
            if not forecast_df.empty:
                fc_contract = forecast_df[forecast_df["contract"] == contract]
                # Find the horizon closest to the selected one (handles capped horizons)
                if not fc_contract.empty:
                    available_h = fc_contract["horizon_days"].unique()
                    best_h = min(available_h, key=lambda h: abs(h - selected_horizon))
                    fc = fc_contract[fc_contract["horizon_days"] == best_h]
                else:
                    fc = fc_contract
                if not fc.empty:
                    row = fc.iloc[0]
                    forecast_date = pd.Timestamp(row["forecast_date"])
                    actual_horizon = int(row["horizon_days"])
                    horizon_date = forecast_date + pd.Timedelta(days=actual_horizon)

                    # Shaded CI band
                    fig.add_trace(go.Scatter(
                        x=[forecast_date, horizon_date, horizon_date, forecast_date],
                        y=[row["current_price"], row["upper_80ci"],
                           row["lower_80ci"], row["current_price"]],
                        fill="toself",
                        fillcolor="rgba(232,97,26,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="90% CI",
                        showlegend=True,
                    ))

                    # Point forecast line
                    fig.add_trace(go.Scatter(
                        x=[forecast_date, horizon_date],
                        y=[row["current_price"], row["point_forecast"]],
                        name="Forecast",
                        line=dict(color=DARK_TEXT, width=2, dash="dash"),
                    ))

            # Determine actual horizon label for title
            _title_hz = selected_horizon
            if not forecast_df.empty:
                _fc_h = forecast_df[forecast_df["contract"] == contract]
                if not _fc_h.empty:
                    _avail = _fc_h["horizon_days"].unique()
                    _title_hz = int(min(_avail, key=lambda h: abs(h - selected_horizon)))

            fig.update_layout(
                title=f"{contract} — Historical + Forecast ({_title_hz}d)",
                xaxis_title="Date",
                yaxis_title="EUR/MWh",
                height=400,
                **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, width="stretch")

        # Forecast table
        if not forecast_df.empty:
            st.subheader("Forecast Details")
            # For each contract, pick the horizon closest to the selected one
            _table_rows = []
            for _c in selected_contracts:
                _fc_c = forecast_df[forecast_df["contract"] == _c]
                if _fc_c.empty:
                    continue
                _avail_h = _fc_c["horizon_days"].unique()
                _best = min(_avail_h, key=lambda h: abs(h - selected_horizon))
                _table_rows.append(_fc_c[_fc_c["horizon_days"] == _best])
            display_fc = pd.concat(_table_rows) if _table_rows else pd.DataFrame()
            if not display_fc.empty:
                def _color_signal(val: str) -> str:
                    if "Opportunity" in val:
                        return "background-color: #d4edda; color: #155724"
                    elif "Wait" in val:
                        return f"background-color: {ORANGE_PALE}; color: {ORANGE}"
                    return ""

                styled = display_fc.style.map(_color_signal, subset=["signal"])
                st.dataframe(styled, width="stretch")
            else:
                st.info("No forecast data for selected contracts / horizon.")

# ===================================================================
# Tab 2 — Fundamentals
# ===================================================================

with tab2:
    st.header("Fundamental Drivers")

    if master.empty:
        st.warning("No data loaded.")
    else:
        filtered = master.copy()
        if date_range:
            filtered = filtered.loc[str(date_range[0]):str(date_range[1])]

        fundamentals = {
            "TTF Gas (EUR/MWh)": "ttf_gas",
            "EUA CO2 (EUR/t)": "eua_co2",
            "API2 Coal": "api2_coal",
            "Hydro Anomaly": "hydro_anomaly",
        }

        cols = st.columns(2)
        for i, (title, col_name) in enumerate(fundamentals.items()):
            with cols[i % 2]:
                if col_name in filtered.columns and filtered[col_name].notna().any():
                    fig = px.line(
                        filtered,
                        y=col_name,
                        title=title,
                        labels={col_name: title, "date": "Date"},
                    )
                    fig.update_traces(line_color=ORANGE)
                    fig.update_layout(height=300, **_PLOTLY_LAYOUT)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info(f"No data for {title}")

        # --- News Sentiment ---
        st.subheader("News Sentiment")
        if "news_sentiment" in filtered.columns and filtered["news_sentiment"].notna().any():
            news_data = filtered[filtered["news_sentiment"].notna()]

            # Sentiment chart
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                x=news_data.index,
                y=news_data["news_sentiment"],
                marker_color=[
                    "#34A853" if v > 0.05 else ("#EA4335" if v < -0.05 else "#AAAAAA")
                    for v in news_data["news_sentiment"]
                ],
                name="Weekly Sentiment",
            ))
            if "news_sentiment_ma4w" in news_data.columns:
                fig_sent.add_trace(go.Scatter(
                    x=news_data.index,
                    y=news_data["news_sentiment_ma4w"],
                    name="4-Week MA",
                    line=dict(color=ORANGE, width=2),
                ))
            fig_sent.add_hline(y=0, line_dash="dash", line_color="#AAAAAA")
            fig_sent.update_layout(
                title="Energy News Sentiment (VADER compound score)",
                yaxis_title="Sentiment (-1 bearish to +1 bullish)",
                height=350, **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_sent, width="stretch")

            # Latest headlines from raw articles
            articles_path = Path(__file__).resolve().parent / "data" / "raw" / "news_articles_latest.csv"
            if articles_path.exists():
                arts = pd.read_csv(articles_path)
                arts = arts.sort_values("compound", key=abs, ascending=False).head(10)
                st.markdown("**Top Recent Headlines (by sentiment strength):**")
                for _, row in arts.iterrows():
                    score = row["compound"]
                    icon = "🟢" if score > 0.05 else ("🔴" if score < -0.05 else "⚪")
                    st.markdown(
                        f"{icon} **{score:+.2f}** — [{row['title']}]({row['url']}) "
                        f"*({row['source']})*"
                    )
        else:
            st.info("No news sentiment data yet. Run 01b_collect_news.py to fetch articles.")

        # Correlation table
        st.subheader("Correlation: Fundamentals vs OMIP Contracts")
        fund_cols = ["ttf_gas", "eua_co2", "api2_coal", "hydro_anomaly",
                     "res_penetration", "eurusd", "german_cal_futures",
                     "news_sentiment"]
        omip_cols = [c for c in filtered.columns if c.startswith("omip_") and "lag" not in c]
        available_fund = [c for c in fund_cols if c in filtered.columns]
        available_omip = [c for c in omip_cols if c in filtered.columns]

        if available_fund and available_omip:
            corr = filtered[available_fund + available_omip].corr()
            corr_subset = corr.loc[available_fund, available_omip]
            fig = px.imshow(
                corr_subset,
                text_auto=".2f",
                color_continuous_scale=[[0, "#1A73E8"], [0.5, WHITE], [1, ORANGE]],
                zmin=-1,
                zmax=1,
                title="Correlation Matrix",
            )
            fig.update_layout(height=400, **_PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")

# ===================================================================
# Tab 3 — Model Performance
# ===================================================================

with tab3:
    st.header("Model Performance")

    if wf_results.empty:
        st.warning("No walk-forward results available. Run training first.")
    else:
        # MAE per fold chart — filter by horizon closest to selected
        st.subheader("Walk-Forward MAE per Fold")
        wf_filtered = wf_results[wf_results["contract"].isin(selected_contracts)]
        # Pick the trained horizon closest to the selected one (we train 7d and 30d)
        if "horizon_days" in wf_filtered.columns:
            available_h = sorted(wf_filtered["horizon_days"].unique())
            best_h = min(available_h, key=lambda h: abs(h - selected_horizon)) if available_h else selected_horizon
            wf_filtered = wf_filtered[wf_filtered["horizon_days"] == best_h]
            horizon_label = f"{best_h}d"
        else:
            horizon_label = ""
        if not wf_filtered.empty:
            fig = px.line(
                wf_filtered,
                x="fold",
                y="mae",
                color="contract",
                markers=True,
                title=f"MAE across Walk-Forward Folds ({horizon_label} model)",
            )
            fig.update_layout(height=400, **_PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")

            # Average MAE summary table per contract
            avg_mae = (
                wf_filtered.groupby("contract")["mae"]
                .agg(["mean", "min", "max", "count"])
                .rename(columns={"mean": "Avg MAE", "min": "Best Fold", "max": "Worst Fold", "count": "Folds"})
                .round(2)
                .sort_values("Avg MAE")
            )
            st.markdown(f"**Average MAE per Position ({horizon_label})**")
            st.dataframe(avg_mae, use_container_width=True)

        # Feature importance bar chart
        st.subheader("Feature Importance (XGBoost)")
        for contract in selected_contracts:
            bundle = load_model_bundle(contract)
            if bundle is None:
                continue
            xgb = bundle.get("xgb")
            features = bundle.get("feature_cols", [])
            if xgb is None or not features:
                continue

            importances = xgb.feature_importances_
            imp_df = pd.DataFrame({"feature": features, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(15)

            fig = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Feature Importance — {contract}",
                color_discrete_sequence=[ORANGE],
            )
            layout_kw = {**_PLOTLY_LAYOUT}
            layout_kw["yaxis"] = {**layout_kw.get("yaxis", {}), "autorange": "reversed"}
            fig.update_layout(height=400, **layout_kw)
            st.plotly_chart(fig, width="stretch")

        # Actual vs Predicted scatter
        st.subheader("Actual vs Predicted")
        for contract in selected_contracts:
            bundle = load_model_bundle(contract)
            if bundle is None:
                continue

            target_col = bundle["target_col"]
            feature_cols = bundle["feature_cols"]
            ridge = bundle.get("ridge")
            xgb_model = bundle.get("xgb")

            if ridge is None or xgb_model is None or target_col not in master.columns:
                continue

            df_temp = master.copy()
            for f in feature_cols:
                if f not in df_temp.columns:
                    df_temp[f] = 0.0

            subset = df_temp.dropna(subset=[target_col])
            if subset.empty:
                continue

            X = subset[feature_cols].ffill().fillna(0.0)
            y = subset[target_col]

            ridge_pred = ridge.predict(X)
            xgb_pred = xgb_model.predict(X)
            ensemble = bundle["ridge_weight"] * ridge_pred + bundle["xgb_weight"] * (ridge_pred + xgb_pred)

            scatter_df = pd.DataFrame({"Actual": y.values, "Predicted": ensemble})
            fig = px.scatter(
                scatter_df,
                x="Actual",
                y="Predicted",
                title=f"Actual vs Predicted — {contract}",
                trendline="ols",
                color_discrete_sequence=[ORANGE],
            )
            # Perfect prediction line
            mn, mx = min(y.min(), ensemble.min()), max(y.max(), ensemble.max())
            fig.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(dash="dash", color="#AAAAAA"),
                name="Perfect",
            ))
            fig.update_layout(height=450, **_PLOTLY_LAYOUT)
            st.plotly_chart(fig, width="stretch")


# ===================================================================
# Footer
# ===================================================================
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;" class="footer-text">'
    f'Nossa Energia &mdash; OMIP Futures Forecast Dashboard &mdash; '
    f'Built with Streamlit + Plotly</div>',
    unsafe_allow_html=True,
)
