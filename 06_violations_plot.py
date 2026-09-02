# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:11:10 2026

@author: user
"""

# =============================================================================
# Script 06 — Violations Plot
# Tail Risk in Pakistani Financial Markets — VaR and ES in Python
# Ahmer | GitHub: ahmer-econ | 2026
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from arch import arch_model

# Set working directory
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR_ES_Python")
print("Working directory set to:", os.getcwd())

# -----------------------------------------------------------------------------
# 1. Load clean returns
# -----------------------------------------------------------------------------
returns = pd.read_csv("data_clean/returns.csv", parse_dates=["Date"], index_col="Date")
print("Returns loaded — Shape:", returns.shape)

assets = ["KSE100", "Gold"]
alpha  = 0.99
window = 250

# =============================================================================
# ROLLING VAR FUNCTIONS
# =============================================================================

def rolling_hs_var(r_window, alpha):
    return np.percentile(r_window, (1 - alpha) * 100)

def rolling_param_var(r_window, alpha):
    mu    = r_window.mean()
    sigma = r_window.std()
    z     = stats.norm.ppf(1 - alpha)
    return mu + z * sigma

def rolling_garch_fhs_var(r_window, alpha):
    r_pct = r_window * 100
    try:
        gm  = arch_model(r_pct, vol="Garch", p=1, q=1,
                         dist="normal", rescale=False)
        res = gm.fit(disp="off")
        cond_vol  = res.conditional_volatility.values
        std_resid = r_pct / cond_vol
        forecast  = res.forecast(horizon=1, reindex=False)
        sigma_h1  = np.sqrt(forecast.variance.values[-1, 0])
        var_std   = np.percentile(std_resid, (1 - alpha) * 100)
        return (sigma_h1 * var_std) / 100
    except Exception:
        return rolling_hs_var(r_window, alpha)

# =============================================================================
# ROLLING BACKTEST — STORE VAR SERIES
# =============================================================================

for asset in assets:
    print(f"\nComputing rolling VaR for {asset}...")
    r     = returns[asset].values
    dates = returns.index[window:]

    var_hs    = []
    var_param = []
    var_gfhs  = []

    for t in range(window, len(r)):
        r_window = r[t - window: t]
        var_hs.append(rolling_hs_var(r_window, alpha))
        var_param.append(rolling_param_var(r_window, alpha))
        var_gfhs.append(rolling_garch_fhs_var(r_window, alpha))

    actual   = r[window:]
    var_hs   = np.array(var_hs)
    var_param = np.array(var_param)
    var_gfhs = np.array(var_gfhs)

    # Violation flags
    viol_hs    = actual < var_hs
    viol_param = actual < var_param
    viol_gfhs  = actual < var_gfhs

    print(f"  HS violations:    {viol_hs.sum()}")
    print(f"  Param violations: {viol_param.sum()}")
    print(f"  GARCH-FHS violations: {viol_gfhs.sum()}")

    # =========================================================================
    # PLOT — three panels stacked vertically
    # =========================================================================
    label    = "KSE-100" if asset == "KSE100" else "COMEX Gold"
    color_r  = "steelblue" if asset == "KSE100" else "goldenrod"

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"{label} — Rolling 99% VaR vs Actual Returns (2015–2024)\n"
        f"Window = {window} days | Red dots = VaR violations",
        fontsize=13, y=1.01
    )

    methods = [
        ("Historical Simulation",   var_hs,    viol_hs),
        ("Parametric (Normal)",      var_param, viol_param),
        ("GARCH-FHS",               var_gfhs,  viol_gfhs),
    ]

    for ax, (method_label, var_series, viol_mask) in zip(axes, methods):
        # Actual returns
        ax.plot(dates, actual, color=color_r, linewidth=0.5,
                alpha=0.7, label="Actual return")

        # VaR line
        ax.plot(dates, var_series, color="black", linewidth=1.0,
                linestyle="--", label=f"99% VaR ({method_label})")

        # Violation dots
        ax.scatter(dates[viol_mask], actual[viol_mask],
                   color="red", s=12, zorder=5,
                   label=f"Violations (n={viol_mask.sum()})")

        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.set_ylabel("Log Return")
        ax.set_title(method_label, fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    fname = f"outputs/figures/{asset.lower()}_violations_99.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

# -----------------------------------------------------------------------------
# Confirm outputs
# -----------------------------------------------------------------------------
print("\nFiles in outputs/figures/:")
print(os.listdir("outputs/figures"))