# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:07:29 2026

@author: user
"""

# =============================================================================
# Script 05 — Backtesting: Kupiec POF and Christoffersen Independence Tests
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

assets  = ["KSE100", "Gold"]
alphas  = [0.95, 0.99]
window  = 250
results = []

# =============================================================================
# FUNCTIONS
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


def kupiec_pof(violations, n, alpha):
    """
    Kupiec (1995) Proportion of Failures test.
    H0: violation rate = (1 - alpha)
    Test statistic is chi-squared with 1 degree of freedom.
    """
    p     = 1 - alpha
    p_hat = violations / n

    if violations == 0:
        lr = 2 * n * np.log(1 / (1 - p))
    elif violations == n:
        lr = 2 * n * np.log(1 / p)
    else:
        lr = 2 * (
            violations * np.log(p_hat / p) +
            (n - violations) * np.log((1 - p_hat) / (1 - p))
        )

    pval = 1 - stats.chi2.cdf(lr, df=1)
    return round(lr, 4), round(pval, 4)


def christoffersen_ind(hits):
    """
    Christoffersen (1998) Independence test.
    Tests whether violations are clustered or independent.
    H0: violations are independently distributed.
    Test statistic is chi-squared with 1 degree of freedom.
    """
    hits = np.array(hits, dtype=int)

    # Transition counts
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    n0 = n00 + n01   # total non-violation days
    n1 = n10 + n11   # total violation days

    # If no transitions exist, test is undefined
    if n0 == 0 or n1 == 0:
        return np.nan, np.nan

    # Unconditional violation probability
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Conditional probabilities
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0

    eps = 1e-10  # prevent log(0)

    # Log-likelihood under H0 (independence — single pi)
    ll0 = (
        (n00 + n10) * np.log(1 - pi + eps) +
        (n01 + n11) * np.log(pi + eps)
    )

    # Log-likelihood under H1 (Markov — separate pi01 and pi11)
    ll1 = (
        n00 * np.log(1 - pi01 + eps) +
        n01 * np.log(pi01 + eps) +
        n10 * np.log(1 - pi11 + eps) +
        n11 * np.log(pi11 + eps)
    )

    lr   = 2 * (ll1 - ll0)
    lr   = max(lr, 0.0)     # numerical safeguard — LR must be non-negative
    pval = 1 - stats.chi2.cdf(lr, df=1)

    return round(lr, 4), round(pval, 4)


# =============================================================================
# ROLLING BACKTEST LOOP
# =============================================================================

for asset in assets:
    r      = returns[asset].values
    n_test = len(r) - window
    print(f"\n{'='*60}")
    print(f"Asset: {asset} | Window: {window} | Test obs: {n_test}")
    print(f"{'='*60}")

    for alpha in alphas:
        hits_hs    = []
        hits_param = []
        hits_gfhs  = []

        for t in range(window, len(r)):
            r_window = r[t - window: t]
            r_actual = r[t]

            var_hs    = rolling_hs_var(r_window, alpha)
            var_param = rolling_param_var(r_window, alpha)
            var_gfhs  = rolling_garch_fhs_var(r_window, alpha)

            hits_hs.append(1    if r_actual < var_hs    else 0)
            hits_param.append(1 if r_actual < var_param else 0)
            hits_gfhs.append(1  if r_actual < var_gfhs  else 0)

        n = len(hits_hs)

        for method, hits in [("HS",         hits_hs),
                              ("Parametric", hits_param),
                              ("GARCH-FHS",  hits_gfhs)]:

            v      = sum(hits)
            v_rate = round(v / n, 4)
            exp_v  = round((1 - alpha) * n)

            lr_k, pval_k = kupiec_pof(v, n, alpha)
            lr_c, pval_c = christoffersen_ind(hits)

            results.append({
                "Asset"      : asset,
                "Confidence" : f"{int(alpha*100)}%",
                "Method"     : method,
                "N_test"     : n,
                "Violations" : v,
                "Expected"   : exp_v,
                "Viol_Rate"  : v_rate,
                "Kupiec_LR"  : lr_k,
                "Kupiec_p"   : pval_k,
                "Christ_LR"  : lr_c,
                "Christ_p"   : pval_c,
            })

            print(f"  {method} | {int(alpha*100)}% | "
                  f"Violations: {v}/{n} ({v_rate:.1%}) | "
                  f"Expected: {exp_v} | "
                  f"Kupiec p={pval_k} | Christ p={pval_c}")

# =============================================================================
# RESULTS TABLE
# =============================================================================
df_bt = pd.DataFrame(results)

print("\n")
print("=" * 95)
print("Backtesting Results — Kupiec POF and Christoffersen Independence Tests")
print("=" * 95)
print(df_bt.to_string(index=False))
print("=" * 95)

df_bt.to_csv("outputs/tables/backtest_results.csv", index=False)
print("\nBacktest results saved to outputs/tables/backtest_results.csv")
print("Files in outputs/tables/:", os.listdir("outputs/tables"))