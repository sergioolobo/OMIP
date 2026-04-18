"""
07_dashboard.py — Combined OMIE Spot + OMIP Futures Dashboard
=============================================================
Unified Iberian electricity market dashboard with two top-level tabs:

  ⚡ OMIE Spot Price  — 7-day hourly day-ahead price forecast
                        (24 independent LASSO models, one per clock hour)
                        Sub-tabs: Forecast | Model Diagnostics | Historical Analysis

  📊 OMIP Futures     — Iberian electricity futures forecasts
                        Sub-tabs: Forecasts | Fundamentals | Model Performance

Run with:
    streamlit run scripts/07_dashboard.py

Data sources:
    OMIE tab  →  ../OMIEForecast/  (sibling project)
    OMIP tab  →  ./data/processed/  (this project)
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent          # OMIPForecast/scripts/
_OMIP_ROOT   = _SCRIPTS_DIR.parent                      # OMIPForecast/
_OMIE_ROOT   = _OMIP_ROOT / "omie_forecast"             # embedded subfolder

# Ensure scripts/ is on sys.path so "import config" resolves to scripts/config.py
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import config as omip_cfg  # noqa: E402  (OMIPForecast config)

# ---------------------------------------------------------------------------
# Load OMIEForecast config via importlib  (avoids module-name collision)
# ---------------------------------------------------------------------------
_OMIE_AVAILABLE  = False
_OMIE_LOAD_ERROR = ""
omie_cfg         = None

_omie_cfg_file = _OMIE_ROOT / "config.py"
if _omie_cfg_file.exists():
    try:
        _spec    = importlib.util.spec_from_file_location("omie_config", _omie_cfg_file)
        omie_cfg = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(omie_cfg)
        _OMIE_AVAILABLE = True
    except Exception as _err:
        _OMIE_LOAD_ERROR = str(_err)
else:
    _OMIE_LOAD_ERROR = f"OMIEForecast directory not found at {_OMIE_ROOT}"

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
_LOGO_PATH    = _SCRIPTS_DIR / "assets" / "logo.png"
_FAVICON_PATH = _SCRIPTS_DIR / "assets" / "favicon.png"

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Iberian Power Dashboard — Nossa Energia",
    layout="wide",
    page_icon=str(_FAVICON_PATH) if _FAVICON_PATH.exists() else "⚡",
)

# ===================================================================
# Branding colours
# ===================================================================
ORANGE       = "#E8611A"
ORANGE_LIGHT = "#F28C52"
ORANGE_PALE  = "#FFF3EB"
WHITE        = "#FFFFFF"
DARK_TEXT    = "#2D2D2D"
GRAY_BG      = "#F7F7F7"

# ===================================================================
# CSS — orange/white Nossa Energia theme
# ===================================================================
st.markdown(f"""
<style>
    .stApp {{ background-color: {WHITE}; }}
    header[data-testid="stHeader"] {{ background-color: {ORANGE} !important; }}
    section[data-testid="stSidebar"] {{
        background-color: {ORANGE_PALE} !important;
        border-right: 3px solid {ORANGE};
    }}
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: {ORANGE} !important;
    }}
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
    [data-testid="stMetric"] {{
        background-color: {ORANGE_PALE};
        border: 1px solid {ORANGE_LIGHT};
        border-radius: 8px;
        padding: 12px;
    }}
    [data-testid="stMetricLabel"] {{ color: {ORANGE} !important; }}
    h1, h2, h3 {{ color: {ORANGE} !important; }}
    .stDataFrame {{ border: 1px solid {ORANGE_LIGHT}; border-radius: 8px; }}
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
    span[data-baseweb="tag"] {{ background-color: {ORANGE} !important; }}
    .footer-text {{ color: {ORANGE}; font-size: 0.85em; }}
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Header
# ===================================================================
_hdr = st.columns([2, 6, 2])
with _hdr[0]:
    if _LOGO_PATH.exists():
        st.image(str(_LOGO_PATH), width=200)
with _hdr[1]:
    st.markdown(
        f'<h1 style="margin-bottom:0; color:{ORANGE} !important;">'
        "Iberian Power Dashboard</h1>"
        f'<p style="color:{ORANGE_LIGHT}; margin-top:0;">'
        "OMIE Day-Ahead Spot &amp; OMIP Futures — Nossa Energia</p>",
        unsafe_allow_html=True,
    )
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
# ── OMIE data loaders (all prefixed omie_)
# ===================================================================

@st.cache_data(ttl=600)
def omie_load_historical() -> pd.DataFrame:
    if not _OMIE_AVAILABLE:
        return pd.DataFrame()
    path = omie_cfg.DATA_PROCESSED / "master_hourly.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="datetime", parse_dates=True,
                     usecols=["datetime", "price_es"])
    return df.sort_index()


@st.cache_data(ttl=600)
def omie_load_forecast() -> pd.DataFrame:
    if not _OMIE_AVAILABLE:
        return pd.DataFrame()
    path = omie_cfg.FORECASTS_DIR / "omie_forecast_latest.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")


@st.cache_data(ttl=3600)
def omie_load_hourly_summary() -> pd.DataFrame:
    if not _OMIE_AVAILABLE:
        return pd.DataFrame()
    path = omie_cfg.FORECASTS_DIR / "hourly_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def omie_load_models() -> dict[int, dict]:
    if not _OMIE_AVAILABLE:
        return {}
    bundles: dict[int, dict] = {}
    for h in range(24):
        p = omie_cfg.MODELS_DIR / f"hour_{h:02d}.pkl"
        if p.exists():
            try:
                bundles[h] = joblib.load(p)
            except Exception:
                pass
    return bundles


@st.cache_data(ttl=3600)
def omie_load_master_full() -> pd.DataFrame:
    if not _OMIE_AVAILABLE:
        return pd.DataFrame()
    path = omie_cfg.DATA_PROCESSED / "master_hourly.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col="datetime", parse_dates=True, low_memory=False)


# ===================================================================
# ── OMIP data loaders (all prefixed omip_)
# ===================================================================

@st.cache_data(ttl=300)
def omip_load_master() -> pd.DataFrame:
    if not omip_cfg.MASTER_DATASET.exists():
        return pd.DataFrame()
    return pd.read_csv(omip_cfg.MASTER_DATASET, index_col="date", parse_dates=True)


@st.cache_data(ttl=60)
def omip_load_forecast() -> pd.DataFrame:
    files = sorted(omip_cfg.FORECASTS_DIR.glob("omip_forecast_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


@st.cache_data(ttl=300)
def omip_load_walkforward() -> pd.DataFrame:
    path = omip_cfg.FORECASTS_DIR / "walkforward_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def omip_load_model_bundle(contract: str) -> dict | None:
    path = omip_cfg.MODELS_DIR / f"model_{contract}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


# ===================================================================
# Pre-load data
# ===================================================================
_omie_hist   = omie_load_historical()
_omie_fc     = omie_load_forecast()
_omip_master = omip_load_master()
_omip_fc     = omip_load_forecast()
_omip_wf     = omip_load_walkforward()


# ===================================================================
# Sidebar
# ===================================================================
st.sidebar.title("⚡ Nossa Energia")
st.sidebar.markdown("---")

# ── OMIE sidebar controls ──
with st.sidebar.expander("⚡ OMIE Spot Controls", expanded=True):
    if not _omie_hist.empty and not _omie_fc.empty:
        _omie_fc_max   = _omie_fc["datetime"].max().date()
        _omie_def_start = (_omie_hist.index.max().date() - pd.Timedelta(days=90))
        omie_date_range = st.slider(
            "Date range",
            min_value=_omie_hist.index.min().date(),
            max_value=_omie_fc_max,
            value=(_omie_def_start, _omie_fc_max),
            format="YYYY-MM-DD",
            key="omie_date_range",
        )
    else:
        omie_date_range = (None, None)

    omie_hour_sel = st.multiselect(
        "Hours to highlight",
        options=list(range(24)),
        default=[7, 10, 13, 19],
        format_func=lambda h: f"H{h:02d}",
        key="omie_hours",
    )
    omie_show_ci = st.toggle("Show 80% CI band", value=True, key="omie_ci")
    omie_show_fi = st.toggle("Show feature importance", value=False, key="omie_fi")

    if st.button("🔄 Re-run OMIE forecast", key="omie_rerun"):
        if _OMIE_AVAILABLE:
            _fc_script = _OMIE_ROOT / "scripts" / "05_forecast.py"
            with st.spinner("Running OMIE forecast…"):
                _res = subprocess.run(
                    [sys.executable, str(_fc_script)],
                    capture_output=True, text=True, timeout=120,
                )
            if _res.returncode == 0:
                st.success("OMIE forecast updated!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Error: {_res.stderr[-400:]}")
        else:
            st.warning(f"OMIEForecast unavailable: {_OMIE_LOAD_ERROR}")

# ── OMIP sidebar controls ──
with st.sidebar.expander("📊 OMIP Futures Controls", expanded=True):
    omip_contracts = st.multiselect(
        "Contracts",
        options=omip_cfg.CONTRACTS,
        default=omip_cfg.CONTRACTS[:3],
        key="omip_contracts",
    )
    omip_horizon = st.selectbox(
        "Forecast Horizon",
        options=omip_cfg.FORECAST_HORIZONS,
        format_func=lambda x: f"{x} days",
        key="omip_horizon",
    )
    if not _omip_master.empty:
        _omip_min = _omip_master.index.min().date()
        _omip_max = _omip_master.index.max().date()
        omip_date_range = st.slider(
            "Date range",
            min_value=_omip_min,
            max_value=_omip_max,
            value=(_omip_min, _omip_max),
            key="omip_date_range",
        )
    else:
        omip_date_range = None


# ===================================================================
# TOP-LEVEL TABS
# ===================================================================
tab_omie, tab_omip = st.tabs(["⚡ OMIE Spot Price", "📊 OMIP Futures"])


# ===================================================================
# ═══ TAB 1 — OMIE SPOT PRICE ═══════════════════════════════════════
# ===================================================================
with tab_omie:
    if not _OMIE_AVAILABLE:
        st.error(f"OMIEForecast project not found.\n\n**Error**: {_OMIE_LOAD_ERROR}")
        st.info(
            "Expected location: `" + str(_OMIE_ROOT) + "`\n\n"
            "Run the OMIEForecast pipeline to generate data and models:\n"
            "```\npython scripts/01_collect_data.py\n"
            "python scripts/02_build_features.py\n"
            "python scripts/03_train_models.py\n"
            "python scripts/04_evaluate_models.py\n"
            "python scripts/05_forecast.py\n```"
        )
        st.stop()

    st_o1, st_o2, st_o3 = st.tabs([
        "📈 7-Day Forecast", "🔬 Model Diagnostics", "📊 Historical Analysis"
    ])

    # ─────────────────────────────────────────────────────────────────
    # OMIE sub-tab 1 — 7-Day Forecast
    # ─────────────────────────────────────────────────────────────────
    with st_o1:
        st.header("OMIE Portugal/Spain Day-Ahead Price Forecast")

        if _omie_hist.empty:
            st.warning("No historical data. Run `scripts/01_collect_data.py` in OMIEForecast.")
        elif _omie_fc.empty:
            st.warning("No forecast found. Run `scripts/05_forecast.py` in OMIEForecast.")
        else:
            _t0 = (pd.Timestamp(omie_date_range[0])
                   if omie_date_range[0] else _omie_hist.index.min())

            _hist_slice = _omie_hist[
                (_omie_hist.index >= _t0)
                & (_omie_hist.index <= _omie_fc["datetime"].min())
            ]
            _now_line = _omie_hist["price_es"].dropna().index.max()

            fig = go.Figure()
            # Historical
            fig.add_trace(go.Scatter(
                x=_hist_slice.index, y=_hist_slice["price_es"],
                mode="lines", name="Historical",
                line=dict(color="black", width=1.5),
            ))
            # Forecast
            fig.add_trace(go.Scatter(
                x=_omie_fc["datetime"], y=_omie_fc["point_forecast"],
                mode="lines", name="Forecast",
                line=dict(color="crimson", width=2, dash="dash"),
            ))
            # 80% CI shading
            if omie_show_ci and "lower_80ci" in _omie_fc.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([_omie_fc["datetime"],
                                 _omie_fc["datetime"].iloc[::-1]]),
                    y=pd.concat([_omie_fc["upper_80ci"],
                                 _omie_fc["lower_80ci"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(30,100,200,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="80% CI",
                    hoverinfo="skip",
                ))
            # "Now" vertical line — use add_shape + add_annotation instead
            # of add_vline to avoid a pandas/plotly arithmetic incompatibility
            # ("Addition/subtraction of integers and integer-arrays with
            #  Timestamp is no longer supported") on newer pandas versions.
            _now_iso = pd.Timestamp(_now_line).isoformat()
            fig.add_shape(
                type="line",
                xref="x", yref="paper",
                x0=_now_iso, x1=_now_iso, y0=0, y1=1,
                line=dict(color="grey", dash="dot"),
            )
            fig.add_annotation(
                x=_now_iso, y=1,
                xref="x", yref="paper",
                text="Now",
                showarrow=False,
                xanchor="left", yanchor="top",
                xshift=4,
                font=dict(color="grey", size=11),
            )
            fig.update_layout(
                height=450,
                xaxis_title="Date",
                yaxis_title="Price (€/MWh)",
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=40, r=20, t=40, b=40),
                hovermode="x unified",
                **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Data quality warning
            if "lag_data_quality" in _omie_fc.columns:
                _low_q = _omie_fc[_omie_fc["lag_data_quality"] < 80]
                if not _low_q.empty:
                    st.warning(
                        f"⚠️ {len(_low_q)} forecast hours have < 80% lag coverage "
                        f"(first: {_low_q['datetime'].iloc[0]})"
                    )

            # KPI metrics
            _omie_sum_df = omie_load_hourly_summary()
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Avg 7-day forecast",
                          f"{_omie_fc['point_forecast'].mean():.1f} €/MWh")
            with k2:
                st.metric("Peak forecast",
                          f"{_omie_fc['point_forecast'].max():.1f} €/MWh")
            with k3:
                st.metric("Min forecast",
                          f"{_omie_fc['point_forecast'].min():.1f} €/MWh")
            with k4:
                if not _omie_sum_df.empty and "mae_mean" in _omie_sum_df.columns:
                    st.metric("Model avg MAE",
                              f"{_omie_sum_df['mae_mean'].mean():.2f} €/MWh")
                else:
                    st.metric("Horizon", "7 days")

            # Daily summary
            st.subheader("Daily Summary")
            _fc_copy = _omie_fc.copy()
            _fc_copy["date"] = pd.to_datetime(_fc_copy["datetime"]).dt.date
            _daily = _fc_copy.groupby("date").agg(
                avg_forecast=("point_forecast", "mean"),
                min_forecast=("point_forecast", "min"),
                max_forecast=("point_forecast", "max"),
                avg_lower=("lower_80ci", "mean"),
                avg_upper=("upper_80ci", "mean"),
            ).round(2).reset_index()
            st.dataframe(_daily, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # OMIE sub-tab 2 — Model Diagnostics
    # ─────────────────────────────────────────────────────────────────
    with st_o2:
        st.header("OMIE Model Diagnostics")

        _omie_sum_df  = omie_load_hourly_summary()
        _omie_bundles = omie_load_models()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("MAE by Hour")
            if _omie_sum_df.empty:
                st.info("No evaluation results yet. Run 04_evaluate_models.py in OMIEForecast.")
            else:
                _bar_colors = [
                    "crimson" if h in [10, 11, 12, 19, 20, 21] else "steelblue"
                    for h in _omie_sum_df["hour"]
                ]
                _fig_mae = go.Figure(go.Bar(
                    x=[f"H{int(h):02d}" for h in _omie_sum_df["hour"]],
                    y=_omie_sum_df["mae_mean"],
                    marker_color=_bar_colors,
                ))
                _fig_mae.update_layout(
                    height=350, xaxis_title="Hour", yaxis_title="MAE (€/MWh)",
                    showlegend=False, margin=dict(l=40, r=10, t=30, b=40),
                    **_PLOTLY_LAYOUT,
                )
                st.plotly_chart(_fig_mae, use_container_width=True)

        with col_b:
            st.subheader("Performance Summary")
            if not _omie_sum_df.empty:
                st.dataframe(_omie_sum_df.round(2), use_container_width=True, height=350)

        # Feature importance heatmap (toggle)
        if omie_show_fi and _omie_bundles:
            st.subheader("Feature Importance Heatmap (|LASSO coefficient|)")
            _all_feats: set[str] = set()
            for _b in _omie_bundles.values():
                _all_feats.update(_b.get("selected_features", []))
            _feats_sorted = sorted(_all_feats)
            if _feats_sorted:
                _z = [
                    [abs(_omie_bundles[h].get("coef", {}).get(f, 0.0)) for h in range(24)]
                    for f in _feats_sorted
                ]
                _fig_hm = go.Figure(go.Heatmap(
                    z=_z,
                    x=[f"H{h:02d}" for h in range(24)],
                    y=_feats_sorted,
                    colorscale="YlOrRd",
                    hoverongaps=False,
                ))
                _fig_hm.update_layout(
                    height=max(400, len(_feats_sorted) * 18),
                    margin=dict(l=180, r=20, t=40, b=40),
                    xaxis_title="Hour",
                    paper_bgcolor=WHITE,
                )
                st.plotly_chart(_fig_hm, use_container_width=True)

        # Side-by-side coefficient comparison
        if _omie_bundles:
            st.subheader("Coefficient Comparison — two hours")
            _avail_hours = sorted(_omie_bundles.keys())
            _cc1, _cc2 = st.columns(2)
            with _cc1:
                _ha = st.selectbox(
                    "Hour A", _avail_hours,
                    index=_avail_hours.index(10) if 10 in _avail_hours else 0,
                    key="omie_coef_ha",
                )
            with _cc2:
                _hb = st.selectbox(
                    "Hour B", _avail_hours,
                    index=_avail_hours.index(20) if 20 in _avail_hours else 1,
                    key="omie_coef_hb",
                )

            def _coef_chart(h: int, label: str) -> go.Figure:
                _b   = _omie_bundles[h]
                coef = _b.get("coef", {})
                sel  = {f: c for f, c in coef.items() if abs(c) > 1e-8}
                if not sel:
                    return go.Figure()
                _fl = sorted(sel, key=lambda f: abs(sel[f]), reverse=True)[:20]
                _fc_colors = ["crimson" if sel[f] < 0 else "steelblue" for f in _fl]
                _fig = go.Figure(go.Bar(
                    x=[sel[f] for f in _fl], y=_fl,
                    orientation="h", marker_color=_fc_colors,
                ))
                _fig.update_layout(
                    title=f"H{h:02d} — {label}",
                    height=450,
                    margin=dict(l=160, r=10, t=40, b=30),
                    xaxis_title="Coefficient",
                    paper_bgcolor=WHITE,
                )
                return _fig

            _ca, _cb = st.columns(2)
            with _ca:
                st.plotly_chart(_coef_chart(_ha, "Hour A"), use_container_width=True)
            with _cb:
                st.plotly_chart(_coef_chart(_hb, "Hour B"), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # OMIE sub-tab 3 — Historical Analysis
    # ─────────────────────────────────────────────────────────────────
    with st_o3:
        st.header("Historical OMIE Price Analysis")

        _omie_full = omie_load_master_full()
        if _omie_full.empty:
            st.info("No master dataset. Run scripts/02_build_features.py in OMIEForecast.")
        else:
            _omie_full.index = pd.to_datetime(_omie_full.index)
            _t0_h = (pd.Timestamp(omie_date_range[0])
                     if omie_date_range[0] else _omie_full.index.min())
            _t1_h = (pd.Timestamp(omie_date_range[1])
                     if omie_date_range[1] else _omie_full.index.max())
            _mslice = _omie_full[
                (_omie_full.index >= _t0_h) & (_omie_full.index <= _t1_h)
            ].copy()

            # ── Price heatmap: X = hour, Y = date ──
            st.subheader("Hourly Price Heatmap (Date × Hour)")
            _mslice["_date"] = _mslice.index.normalize()
            _mslice["_hour"] = _mslice.index.hour
            _pivot = _mslice.pivot_table(
                index="_date", columns="_hour", values="price_es", aggfunc="mean"
            )
            if not _pivot.empty:
                _fig_heatmap = px.imshow(
                    _pivot.values,
                    x=[f"H{h:02d}" for h in _pivot.columns],
                    y=[str(d.date()) for d in _pivot.index],
                    color_continuous_scale="RdYlGn_r",
                    labels={"color": "€/MWh"},
                    aspect="auto",
                    title="OMIE Price Heatmap — Hour vs Date",
                )
                _fig_heatmap.update_layout(
                    height=max(450, len(_pivot) * 3),
                    xaxis_title="Hour of Day",
                    yaxis_title="Date",
                    coloraxis_colorbar=dict(title="€/MWh"),
                    margin=dict(l=80, r=20, t=50, b=40),
                    paper_bgcolor=WHITE,
                )
                st.plotly_chart(_fig_heatmap, use_container_width=True)

            # ── Correlation table ──
            st.subheader("Correlation of Fundamentals with Spot Price")
            _fund_cols = [
                c for c in (omie_cfg.FUNDAMENTAL_FEATURES + omie_cfg.ENTSOE_FEATURES)
                if c in _mslice.columns
            ]
            if _fund_cols:
                _corr_rows: list[dict] = []
                for _h in (omie_hour_sel if omie_hour_sel else [0, 6, 12, 18]):
                    _sub = _mslice[_mslice["_hour"] == _h][["price_es"] + _fund_cols].dropna()
                    if len(_sub) > 10:
                        _corrs = _sub.corr()["price_es"].drop("price_es")
                        for _feat, _val in _corrs.items():
                            _corr_rows.append({
                                "hour": f"H{_h:02d}",
                                "feature": _feat,
                                "correlation": round(float(_val), 3),
                            })
                if _corr_rows:
                    _corr_df = pd.DataFrame(_corr_rows).pivot(
                        index="feature", columns="hour", values="correlation"
                    )
                    st.dataframe(
                        _corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                        use_container_width=True,
                    )
            else:
                st.info("Fundamental features not found in master dataset.")


# ===================================================================
# ═══ TAB 2 — OMIP FUTURES ══════════════════════════════════════════
# ===================================================================
with tab_omip:
    st_p1, st_p2, st_p3 = st.tabs([
        "OMIP Forecasts", "Fundamentals", "Model Performance"
    ])

    # ─────────────────────────────────────────────────────────────────
    # OMIP sub-tab 1 — Price Forecasts
    # ─────────────────────────────────────────────────────────────────
    with st_p1:
        st.header("OMIP Price Forecasts")

        if _omip_master.empty:
            st.warning("No master dataset available. Run the OMIP pipeline first.")
        else:
            _omip_filt = _omip_master.copy()
            if omip_date_range:
                _omip_filt = _omip_filt.loc[
                    str(omip_date_range[0]):str(omip_date_range[1])
                ]

            for _contract in omip_contracts:
                _bundle = omip_load_model_bundle(_contract)
                _target = _bundle["target_col"] if _bundle else "omip_yr1"

                if _target not in _omip_filt.columns:
                    st.info(f"No price data for {_contract} (column: {_target})")
                    continue

                _fig = go.Figure()
                _fig.add_trace(go.Scatter(
                    x=_omip_filt.index,
                    y=_omip_filt[_target],
                    name="Historical",
                    line=dict(color=ORANGE, width=2),
                ))

                if not _omip_fc.empty:
                    _fc_c = _omip_fc[_omip_fc["contract"] == _contract]
                    if not _fc_c.empty:
                        _avail_h = _fc_c["horizon_days"].unique()
                        _best_h  = min(_avail_h, key=lambda h: abs(h - omip_horizon))
                        _fc_row  = _fc_c[_fc_c["horizon_days"] == _best_h]
                        if not _fc_row.empty:
                            _row = _fc_row.iloc[0]
                            _fd  = pd.Timestamp(_row["forecast_date"])
                            _hd  = _fd + pd.Timedelta(days=int(_row["horizon_days"]))
                            # CI shading
                            _fig.add_trace(go.Scatter(
                                x=[_fd, _hd, _hd, _fd],
                                y=[_row["current_price"], _row["upper_80ci"],
                                   _row["lower_80ci"], _row["current_price"]],
                                fill="toself",
                                fillcolor="rgba(232,97,26,0.15)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="90% CI",
                            ))
                            # Forecast line
                            _fig.add_trace(go.Scatter(
                                x=[_fd, _hd],
                                y=[_row["current_price"], _row["point_forecast"]],
                                name="Forecast",
                                line=dict(color=DARK_TEXT, width=2, dash="dash"),
                            ))

                _hz_lbl = int(omip_horizon)
                if not _omip_fc.empty:
                    _fc_h = _omip_fc[_omip_fc["contract"] == _contract]
                    if not _fc_h.empty:
                        _hz_lbl = int(min(
                            _fc_h["horizon_days"].unique(),
                            key=lambda h: abs(h - omip_horizon),
                        ))

                _fig.update_layout(
                    title=f"{_contract} — Historical + Forecast ({_hz_lbl}d)",
                    xaxis_title="Date",
                    yaxis_title="EUR/MWh",
                    height=400,
                    **_PLOTLY_LAYOUT,
                )
                st.plotly_chart(_fig, use_container_width=True)

            # Forecast table
            if not _omip_fc.empty:
                st.subheader("Forecast Details")
                _tbl_rows = []
                for _c in omip_contracts:
                    _fc_c = _omip_fc[_omip_fc["contract"] == _c]
                    if _fc_c.empty:
                        continue
                    _bh = min(_fc_c["horizon_days"].unique(),
                              key=lambda h: abs(h - omip_horizon))
                    _tbl_rows.append(_fc_c[_fc_c["horizon_days"] == _bh])
                if _tbl_rows:
                    _display_fc = pd.concat(_tbl_rows)

                    def _color_signal(val: str) -> str:
                        if "Opportunity" in str(val):
                            return "background-color: #d4edda; color: #155724"
                        if "Wait" in str(val):
                            return f"background-color: {ORANGE_PALE}; color: {ORANGE}"
                        return ""

                    if "signal" in _display_fc.columns:
                        st.dataframe(
                            _display_fc.style.map(_color_signal, subset=["signal"]),
                            use_container_width=True,
                        )
                    else:
                        st.dataframe(_display_fc, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # OMIP sub-tab 2 — Fundamentals
    # ─────────────────────────────────────────────────────────────────
    with st_p2:
        st.header("Fundamental Drivers")

        if _omip_master.empty:
            st.warning("No data loaded.")
        else:
            _omip_filt2 = _omip_master.copy()
            if omip_date_range:
                _omip_filt2 = _omip_filt2.loc[
                    str(omip_date_range[0]):str(omip_date_range[1])
                ]

            _fundamentals = {
                "TTF Gas (EUR/MWh)": "ttf_gas",
                "EUA CO2 (EUR/t)":   "eua_co2",
                "API2 Coal":          "api2_coal",
                "Hydro Anomaly":      "hydro_anomaly",
            }
            _fcols = st.columns(2)
            for _i, (_title, _col) in enumerate(_fundamentals.items()):
                with _fcols[_i % 2]:
                    if _col in _omip_filt2.columns and _omip_filt2[_col].notna().any():
                        _figf = px.line(
                            _omip_filt2, y=_col, title=_title,
                            labels={_col: _title, "date": "Date"},
                        )
                        _figf.update_traces(line_color=ORANGE)
                        _figf.update_layout(height=300, **_PLOTLY_LAYOUT)
                        st.plotly_chart(_figf, use_container_width=True)
                    else:
                        st.info(f"No data for {_title}")

            # News sentiment
            st.subheader("News Sentiment")
            if ("news_sentiment" in _omip_filt2.columns
                    and _omip_filt2["news_sentiment"].notna().any()):
                _nd = _omip_filt2[_omip_filt2["news_sentiment"].notna()]
                _fig_sent = go.Figure()
                _fig_sent.add_trace(go.Bar(
                    x=_nd.index,
                    y=_nd["news_sentiment"],
                    marker_color=[
                        "#34A853" if v > 0.05 else ("#EA4335" if v < -0.05 else "#AAAAAA")
                        for v in _nd["news_sentiment"]
                    ],
                    name="Weekly Sentiment",
                ))
                if "news_sentiment_ma4w" in _nd.columns:
                    _fig_sent.add_trace(go.Scatter(
                        x=_nd.index, y=_nd["news_sentiment_ma4w"],
                        name="4-Week MA", line=dict(color=ORANGE, width=2),
                    ))
                _fig_sent.add_hline(y=0, line_dash="dash", line_color="#AAAAAA")
                _fig_sent.update_layout(
                    title="Energy News Sentiment (VADER compound score)",
                    yaxis_title="Sentiment (−1 bearish → +1 bullish)",
                    height=350,
                    **_PLOTLY_LAYOUT,
                )
                st.plotly_chart(_fig_sent, use_container_width=True)

                # Latest headlines
                _arts_path = _SCRIPTS_DIR / "data" / "raw" / "news_articles_latest.csv"
                if _arts_path.exists():
                    _arts = pd.read_csv(_arts_path)
                    _arts = _arts.sort_values("compound", key=abs, ascending=False).head(10)
                    st.markdown("**Top Recent Headlines (by sentiment strength):**")
                    for _, _art_row in _arts.iterrows():
                        _score = _art_row["compound"]
                        _icon  = "🟢" if _score > 0.05 else ("🔴" if _score < -0.05 else "⚪")
                        st.markdown(
                            f"{_icon} **{_score:+.2f}** — "
                            f"[{_art_row['title']}]({_art_row['url']}) "
                            f"*({_art_row['source']})*"
                        )
            else:
                st.info("No news sentiment data yet. Run 01b_collect_news.py.")

            # Correlation table
            st.subheader("Correlation: Fundamentals vs OMIP Contracts")
            _fund_c = ["ttf_gas", "eua_co2", "api2_coal", "hydro_anomaly",
                       "res_penetration", "eurusd", "german_cal_futures", "news_sentiment"]
            _omip_c = [
                c for c in _omip_filt2.columns
                if c.startswith("omip_") and "lag" not in c
            ]
            _af = [c for c in _fund_c if c in _omip_filt2.columns]
            _ao = [c for c in _omip_c if c in _omip_filt2.columns]
            if _af and _ao:
                _corr_m = _omip_filt2[_af + _ao].corr()
                _fig_corr = px.imshow(
                    _corr_m.loc[_af, _ao],
                    text_auto=".2f",
                    color_continuous_scale=[[0, "#1A73E8"], [0.5, WHITE], [1, ORANGE]],
                    zmin=-1, zmax=1,
                    title="Correlation Matrix",
                )
                _fig_corr.update_layout(height=400, **_PLOTLY_LAYOUT)
                st.plotly_chart(_fig_corr, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # OMIP sub-tab 3 — Model Performance
    # ─────────────────────────────────────────────────────────────────
    with st_p3:
        st.header("OMIP Model Performance")

        if _omip_wf.empty:
            st.warning("No walk-forward results available. Run OMIP training first.")
        else:
            st.subheader("Walk-Forward MAE per Fold")
            _wff = _omip_wf[_omip_wf["contract"].isin(omip_contracts)]
            _hz_lbl2 = ""
            if "horizon_days" in _wff.columns:
                _avail_hf = sorted(_wff["horizon_days"].unique())
                if _avail_hf:
                    _bh2  = min(_avail_hf, key=lambda h: abs(h - omip_horizon))
                    _wff  = _wff[_wff["horizon_days"] == _bh2]
                    _hz_lbl2 = f"{_bh2}d"
            if not _wff.empty:
                _fig_wf = px.line(
                    _wff, x="fold", y="mae", color="contract", markers=True,
                    title=f"MAE across Walk-Forward Folds ({_hz_lbl2})",
                )
                _fig_wf.update_layout(height=400, **_PLOTLY_LAYOUT)
                st.plotly_chart(_fig_wf, use_container_width=True)

                _avg_mae = (
                    _wff.groupby("contract")["mae"]
                    .agg(["mean", "min", "max", "count"])
                    .rename(columns={
                        "mean": "Avg MAE", "min": "Best Fold",
                        "max": "Worst Fold", "count": "Folds",
                    })
                    .round(2)
                    .sort_values("Avg MAE")
                )
                st.markdown(f"**Average MAE per Position ({_hz_lbl2})**")
                st.dataframe(_avg_mae, use_container_width=True)

        # Feature importance
        st.subheader("Feature Importance (XGBoost)")
        for _c in omip_contracts:
            _b = omip_load_model_bundle(_c)
            if _b is None:
                continue
            _xgb   = _b.get("xgb")
            _feats = _b.get("feature_cols", [])
            if _xgb is None or not _feats:
                continue
            _imp = pd.DataFrame({
                "feature": _feats,
                "importance": _xgb.feature_importances_,
            }).sort_values("importance", ascending=False).head(15)
            _lykw = {**_PLOTLY_LAYOUT}
            _lykw["yaxis"] = {**_lykw.get("yaxis", {}), "autorange": "reversed"}
            _fig_imp = px.bar(
                _imp, x="importance", y="feature", orientation="h",
                title=f"Feature Importance — {_c}",
                color_discrete_sequence=[ORANGE],
            )
            _fig_imp.update_layout(height=400, **_lykw)
            st.plotly_chart(_fig_imp, use_container_width=True)

        # Actual vs Predicted
        st.subheader("Actual vs Predicted")
        if not _omip_master.empty:
            for _c in omip_contracts:
                _b = omip_load_model_bundle(_c)
                if _b is None:
                    continue
                _tc    = _b["target_col"]
                _fcols = _b["feature_cols"]
                _ridge = _b.get("ridge")
                _xgbm  = _b.get("xgb")
                if _ridge is None or _xgbm is None or _tc not in _omip_master.columns:
                    continue
                _tmp = _omip_master.copy()
                for _f in _fcols:
                    if _f not in _tmp.columns:
                        _tmp[_f] = 0.0
                _sub = _tmp.dropna(subset=[_tc])
                if _sub.empty:
                    continue
                _X   = _sub[_fcols].ffill().fillna(0.0)
                _y   = _sub[_tc]
                _rp  = _ridge.predict(_X)
                _xp  = _xgbm.predict(_X)
                _ens = _b["ridge_weight"] * _rp + _b["xgb_weight"] * (_rp + _xp)
                _scat = pd.DataFrame({"Actual": _y.values, "Predicted": _ens})
                _fig_sc = px.scatter(
                    _scat, x="Actual", y="Predicted",
                    title=f"Actual vs Predicted — {_c}",
                    trendline="ols",
                    color_discrete_sequence=[ORANGE],
                )
                _mn, _mx = min(_y.min(), _ens.min()), max(_y.max(), _ens.max())
                _fig_sc.add_trace(go.Scatter(
                    x=[_mn, _mx], y=[_mn, _mx],
                    mode="lines",
                    line=dict(dash="dash", color="#AAAAAA"),
                    name="Perfect fit",
                ))
                _fig_sc.update_layout(height=450, **_PLOTLY_LAYOUT)
                st.plotly_chart(_fig_sc, use_container_width=True)


# ===================================================================
# Footer
# ===================================================================
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;" class="footer-text">'
    f"Nossa Energia &mdash; Iberian Power Dashboard &mdash; "
    f"OMIE Spot + OMIP Futures &mdash; Built with Streamlit + Plotly</div>",
    unsafe_allow_html=True,
)
