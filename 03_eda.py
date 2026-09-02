# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:02:15 2026

@author: user
"""

# =============================================================================
# Script 03 — Exploratory Data Analysis
# Tail Risk in Pakistani Financial Markets — VaR and ES in Python
# Ahmer | GitHub: ahmer-econ | 2026
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# Set working directory
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR_ES_Python")
print("Working directory set to:", os.getcwd())

# -----------------------------------------------------------------------------
# 1. Load clean returns
# -----------------------------------------------------------------------------
returns = pd.read_csv("data_clean/returns.csv", parse_dates=["Date"], index_col="Date")
print("Returns loaded — Shape:", returns.shape)
print(returns.head())

# -----------------------------------------------------------------------------
# 2. Skewness and Kurtosis table
# -----------------------------------------------------------------------------
print("\n--- Higher Moments ---")
moments = pd.DataFrame({
    "Mean"    : returns.mean(),
    "Std Dev" : returns.std(),
    "Skewness": returns.skew(),
    "Kurtosis": returns.kurt(),   # excess kurtosis (normal = 0)
    "Min"     : returns.min(),
    "Max"     : returns.max()
})
print(moments.round(6))

# -----------------------------------------------------------------------------
# 3. Plot 1 — KSE-100 returns over time
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(returns.index, returns["KSE100"], color="steelblue", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("KSE-100 Daily Log Returns (2015–2024)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("outputs/figures/kse100_returns.png", dpi=150)
plt.show()
print("Saved: outputs/figures/kse100_returns.png")

# -----------------------------------------------------------------------------
# 4. Plot 2 — Gold returns over time
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(returns.index, returns["Gold"], color="goldenrod", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("COMEX Gold Daily Log Returns (2015–2024)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("outputs/figures/gold_returns.png", dpi=150)
plt.show()
print("Saved: outputs/figures/gold_returns.png")

# -----------------------------------------------------------------------------
# 5. Plot 3 — Return distributions with normal overlay
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, color, label in zip(
    axes,
    ["KSE100", "Gold"],
    ["steelblue", "goldenrod"],
    ["KSE-100", "COMEX Gold"]
):
    data = returns[col].dropna()
    sns.histplot(data, bins=80, stat="density", color=color,
                 alpha=0.5, ax=ax, label="Empirical")

    # Normal overlay
    x = np.linspace(data.min(), data.max(), 300)
    ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()),
            color="red", linewidth=1.5, label="Normal fit")

    ax.set_title(f"{label} Return Distribution", fontsize=12)
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()

plt.suptitle("Empirical vs Normal Distribution (2015–2024)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/figures/return_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/figures/return_distributions.png")

# -----------------------------------------------------------------------------
# 6. Plot 4 — Autocorrelation of squared returns (ARCH effects)
# -----------------------------------------------------------------------------
from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, label in zip(
    axes,
    ["KSE100", "Gold"],
    ["KSE-100", "COMEX Gold"]
):
    sq_returns = returns[col] ** 2
    # Manual ACF using pandas
    acf_vals = [sq_returns.autocorr(lag=i) for i in range(1, 31)]
    ax.bar(range(1, 31), acf_vals, color="steelblue" if col == "KSE100" else "goldenrod")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1.96 / np.sqrt(len(sq_returns)), color="red",
               linewidth=1, linestyle="--", label="95% CI")
    ax.axhline(-1.96 / np.sqrt(len(sq_returns)), color="red",
               linewidth=1, linestyle="--")
    ax.set_title(f"{label} — ACF of Squared Returns", fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()

plt.suptitle("ARCH Effects — Autocorrelation of Squared Returns", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/figures/arch_effects.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: outputs/figures/arch_effects.png")

# -----------------------------------------------------------------------------
# 7. Confirm all figures saved
# -----------------------------------------------------------------------------
print("\nFiles in outputs/figures/:")
print(os.listdir("outputs/figures"))