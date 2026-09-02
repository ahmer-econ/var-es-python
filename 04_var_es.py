# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:06:27 2026

@author: user
"""

# =============================================================================
# Script 04 — VaR and ES Estimation
# Tail Risk in Pakistani Financial Markets — VaR and ES in Python
# Ahmer | GitHub: ahmer-econ | 2026
# =============================================================================

import os
import numpy as np
import pandas as pd
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

assets   = ["KSE100", "Gold"]
alphas   = [0.95, 0.99]   # confidence levels
results  = []             # will collect one row per asset-level combination

# =============================================================================
# FUNCTIONS
# =============================================================================

def hs_var_es(r, alpha):
    """Historical Simulation VaR and ES."""
    var = np.percentile(r, (1 - alpha) * 100)
    es  = r[r <= var].mean()
    return var, es


def parametric_var_es(r, alpha):
    """Parametric (Normal) VaR and ES."""
    mu    = r.mean()
    sigma = r.std()
    z     = stats.norm.ppf(1 - alpha)
    var   = mu + z * sigma
    # Analytical ES for normal distribution
    es    = mu - sigma * stats.norm.pdf(z) / (1 - alpha)
    return var, es


def garch_fhs_var_es(r, alpha):
    """
    GARCH(1,1) Filtered Historical Simulation VaR and ES.
    1. Fit GARCH(1,1) to the return series.
    2. Extract standardised residuals.
    3. Apply HS to standardised residuals.
    4. Scale by one-step-ahead conditional volatility forecast.
    """
    # Scale returns to percentage for arch library stability
    r_pct = r * 100

    # Fit GARCH(1,1) with normal innovations
    gm = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = gm.fit(disp="off")

    # Conditional volatilities (in percentage scale)
    cond_vol = res.conditional_volatility

    # Standardised residuals
    std_resid = r_pct.values / cond_vol.values

    # One-step-ahead volatility forecast (last forecast, in percentage scale)
    forecast  = res.forecast(horizon=1, reindex=False)
    sigma_h1  = np.sqrt(forecast.variance.values[-1, 0])   # percentage scale

    # HS on standardised residuals
    var_std = np.percentile(std_resid, (1 - alpha) * 100)
    es_std  = std_resid[std_resid <= var_std].mean()

    # Scale back to return scale (divide by 100)
    var = (sigma_h1 * var_std) / 100
    es  = (sigma_h1 * es_std)  / 100

    # Print GARCH summary for this asset
    print(f"\n  GARCH(1,1) parameters:")
    print(f"  omega={res.params['omega']:.6f}  "
          f"alpha[1]={res.params['alpha[1]']:.4f}  "
          f"beta[1]={res.params['beta[1]']:.4f}  "
          f"persistence={res.params['alpha[1]']+res.params['beta[1]']:.4f}")
    print(f"  One-step-ahead sigma forecast: {sigma_h1/100:.6f} (return scale)")

    return var, es, res


# =============================================================================
# ESTIMATION LOOP
# =============================================================================

garch_results = {}   # store fitted GARCH objects for backtesting script

for asset in assets:
    r = returns[asset].values
    print(f"\n{'='*60}")
    print(f"Asset: {asset}")
    print(f"{'='*60}")

    # --- Fit GARCH once per asset (used across both alpha levels) ---
    print("\nFitting GARCH(1,1)...")
    r_pct = returns[asset] * 100
    gm    = arch_model(r_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    garch_res = gm.fit(disp="off")
    garch_results[asset] = garch_res

    cond_vol  = garch_res.conditional_volatility.values
    std_resid = r_pct.values / cond_vol

    forecast  = garch_res.forecast(horizon=1, reindex=False)
    sigma_h1  = np.sqrt(forecast.variance.values[-1, 0])

    print(f"  omega={garch_res.params['omega']:.6f}  "
          f"alpha[1]={garch_res.params['alpha[1]']:.4f}  "
          f"beta[1]={garch_res.params['beta[1]']:.4f}  "
          f"persistence={garch_res.params['alpha[1]']+garch_res.params['beta[1]']:.4f}")
    print(f"  One-step-ahead sigma forecast: {sigma_h1/100:.6f} (return scale)")

    for alpha in alphas:
        # Historical Simulation
        hs_v, hs_e = hs_var_es(r, alpha)

        # Parametric
        p_v, p_e = parametric_var_es(r, alpha)

        # GARCH-FHS
        var_std = np.percentile(std_resid, (1 - alpha) * 100)
        es_std  = std_resid[std_resid <= var_std].mean()
        gfhs_v  = (sigma_h1 * var_std) / 100
        gfhs_e  = (sigma_h1 * es_std)  / 100

        results.append({
            "Asset"          : asset,
            "Confidence"     : f"{int(alpha*100)}%",
            "HS_VaR"         : round(hs_v, 6),
            "HS_ES"          : round(hs_e, 6),
            "Param_VaR"      : round(p_v,  6),
            "Param_ES"       : round(p_e,  6),
            "GARCH_FHS_VaR"  : round(gfhs_v, 6),
            "GARCH_FHS_ES"   : round(gfhs_e, 6),
        })

# =============================================================================
# RESULTS TABLE
# =============================================================================
df_results = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("VaR and ES Estimation Results")
print("=" * 70)
print(df_results.to_string(index=False))
print("=" * 70)

# Save results
df_results.to_csv("outputs/tables/var_es_results.csv", index=False)
print("\nResults saved to outputs/tables/var_es_results.csv")

# Save standardised residuals and conditional volatility for backtesting
for asset in assets:
    r_pct     = returns[asset] * 100
    cond_vol  = garch_results[asset].conditional_volatility
    std_resid = r_pct / cond_vol

    pd.DataFrame({
        "Date"      : returns.index,
        "Return"    : returns[asset].values,
        "CondVol"   : (cond_vol.values / 100),
        "StdResid"  : std_resid.values
    }).to_csv(f"data_clean/garch_{asset.lower()}.csv", index=False)

print("\nGARCH residuals saved to data_clean/")
print("Files in data_clean/:", os.listdir("data_clean"))
print("Files in outputs/tables/:", os.listdir("outputs/tables"))