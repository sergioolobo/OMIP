"""
03_cointegration_check.py — Stationarity & Cointegration Diagnostics
=====================================================================
Runs ADF, Johansen, and Engle-Granger tests on the key price series.
If cointegration is found, computes an error-correction term (ECT) and
appends it to the master dataset.

Inputs:  data/processed/master_dataset.csv
Outputs: data/processed/cointegration_report.txt
         data/processed/master_dataset.csv  (updated with ect_term)
         outputs/charts/correlation_heatmap.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("cointegration_check")
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
# ADF stationarity test
# ===================================================================

def run_adf_tests(df: pd.DataFrame, series_names: list[str]) -> list[str]:
    """Run Augmented Dickey-Fuller tests on each named series.

    Returns a list of report lines.
    """
    lines = ["ADF STATIONARITY TESTS", "=" * 60]
    for name in series_names:
        if name not in df.columns or df[name].dropna().shape[0] < 30:
            lines.append(f"  {name}: SKIPPED (insufficient data)")
            continue
        series = df[name].dropna()
        result = adfuller(series, autolag="AIC")
        stat, pval, _, nobs, crit, _ = result
        stationary = "STATIONARY" if pval < 0.05 else "NON-STATIONARY"
        lines.append(f"  {name}:")
        lines.append(f"    ADF stat = {stat:.4f},  p-value = {pval:.4f}  ->  {stationary}")
        lines.append(f"    Critical values: {crit}")
        lines.append(f"    Observations: {nobs}")
        lines.append("")
    return lines


# ===================================================================
# Johansen cointegration test
# ===================================================================

def run_johansen_test(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Run Johansen cointegration test on specified columns.

    Returns report lines.
    """
    lines = ["", "JOHANSEN COINTEGRATION TEST  (Power–Gas–CO2 triplet)", "=" * 60]
    data = df[cols].dropna()
    if data.shape[0] < 50:
        lines.append("  SKIPPED — fewer than 50 complete observations.")
        return lines

    try:
        result = coint_johansen(data, det_order=0, k_ar_diff=2)
        lines.append(f"  Variables: {cols}")
        lines.append(f"  Observations: {data.shape[0]}")
        lines.append("")
        lines.append("  Trace statistic:")
        for i, (trace, cv) in enumerate(zip(result.lr1, result.cvt)):
            sig = "***" if trace > cv[2] else ("**" if trace > cv[1] else ("*" if trace > cv[0] else ""))
            lines.append(f"    r <= {i}: trace = {trace:.2f},  90%/95%/99% = {cv[0]:.2f}/{cv[1]:.2f}/{cv[2]:.2f}  {sig}")
        lines.append("")
        lines.append("  Max-eigenvalue statistic:")
        for i, (maxeig, cv) in enumerate(zip(result.lr2, result.cvm)):
            sig = "***" if maxeig > cv[2] else ("**" if maxeig > cv[1] else ("*" if maxeig > cv[0] else ""))
            lines.append(f"    r <= {i}: max-eig = {maxeig:.2f},  90%/95%/99% = {cv[0]:.2f}/{cv[1]:.2f}/{cv[2]:.2f}  {sig}")
    except Exception as exc:
        lines.append(f"  ERROR: {exc}")

    return lines


# ===================================================================
# Engle-Granger pairwise cointegration
# ===================================================================

def run_engle_granger(
    df: pd.DataFrame,
    target: str,
    regressors: list[str],
) -> tuple[list[str], pd.Series | None]:
    """Run Engle-Granger cointegration between *target* and each regressor.

    Returns report lines AND an ECT Series (or None).
    """
    lines = ["", "ENGLE-GRANGER PAIRWISE COINTEGRATION", "=" * 60]
    best_ect: pd.Series | None = None
    best_pval = 1.0

    for reg in regressors:
        if target not in df.columns or reg not in df.columns:
            lines.append(f"  {target} ~ {reg}: SKIPPED (column missing)")
            continue
        pair = df[[target, reg]].dropna()
        if pair.shape[0] < 50:
            lines.append(f"  {target} ~ {reg}: SKIPPED (insufficient data)")
            continue

        try:
            stat, pval, crit_vals = coint(pair[target], pair[reg])
            cointegrated = pval < 0.05
            label = "COINTEGRATED at 5%" if cointegrated else "NOT cointegrated"
            lines.append(f"  {target} ~ {reg}:")
            lines.append(f"    EG stat = {stat:.4f},  p-value = {pval:.4f}  ->  {label}")
            lines.append(f"    Critical values (1%/5%/10%): {crit_vals}")
            lines.append("")

            if cointegrated:
                logger.info("Power and %s ARE cointegrated at 5%% level — using ECT as feature", reg)
                lines.append(f"    -> Power and {reg} ARE cointegrated at 5% level — using ECT as feature")

                # Compute ECT: residuals from OLS(target ~ reg)
                from numpy.polynomial.polynomial import polyfit
                coeffs = np.polyfit(pair[reg], pair[target], 1)
                fitted = coeffs[0] * df[reg] + coeffs[1]
                ect = df[target] - fitted

                if pval < best_pval:
                    best_pval = pval
                    best_ect = ect
        except Exception as exc:
            lines.append(f"  {target} ~ {reg}: ERROR — {exc}")

    return lines, best_ect


# ===================================================================
# Correlation heatmap
# ===================================================================

def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Generate and save a correlation heatmap of all numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        logger.warning("Not enough numeric columns for heatmap.")
        return

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(min(24, numeric.shape[1]), min(20, numeric.shape[1])))
    sns.heatmap(
        corr,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Correlation heatmap saved to %s", path)


# ===================================================================
# Main
# ===================================================================

def run_cointegration_check() -> None:
    """Execute all diagnostics and update master dataset with ECT."""
    logger.info("=" * 60)
    logger.info("COINTEGRATION & STATIONARITY DIAGNOSTICS")
    logger.info("=" * 60)

    # Load master dataset
    df = pd.read_csv(config.MASTER_DATASET, index_col="date", parse_dates=True)
    logger.info("Loaded master dataset: %d rows × %d cols", *df.shape)

    report_lines: list[str] = []

    # --- ADF tests ---
    adf_cols = ["omip_yr1", "ttf_gas", "api2_coal", "eua_co2", "omie_spot"]
    report_lines.extend(run_adf_tests(df, adf_cols))

    # --- Johansen test on power-gas-CO2 triplet ---
    johansen_cols = ["omip_yr1", "ttf_gas", "eua_co2"]
    report_lines.extend(run_johansen_test(df, johansen_cols))

    # --- Engle-Granger pairwise ---
    eg_lines, best_ect = run_engle_granger(
        df, target="omip_yr1", regressors=["ttf_gas", "api2_coal", "eua_co2"]
    )
    report_lines.extend(eg_lines)

    # --- Update master dataset with ECT ---
    if best_ect is not None:
        df["ect_term"] = best_ect
        df.to_csv(config.MASTER_DATASET)
        report_lines.append("")
        report_lines.append("ECT term computed and added to master_dataset.csv")
        logger.info("ECT term added to master dataset.")
    else:
        report_lines.append("")
        report_lines.append("No cointegration found — ECT term remains NaN.")
        logger.info("No cointegration found at 5%% significance.")

    # --- Save report ---
    report_text = "\n".join(report_lines)
    config.COINTEGRATION_REPORT.write_text(report_text, encoding="utf-8")
    logger.info("Report saved to %s", config.COINTEGRATION_REPORT)
    # Replace Unicode arrows with ASCII for Windows console compatibility
    print(report_text.replace("\u2192", "->").encode("ascii", errors="replace").decode("ascii"))

    # --- Correlation heatmap ---
    plot_correlation_heatmap(df, config.CHARTS_DIR / "correlation_heatmap.png")

    logger.info("=" * 60)
    logger.info("COINTEGRATION CHECK COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_cointegration_check()
