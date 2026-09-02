# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:46:20 2026

@author: user
"""
# =============================================================================
# Script 02 — Data Cleaning and Log Returns
# Tail Risk in Pakistani Financial Markets — VaR and ES in Python
# Ahmer | GitHub: ahmer-econ | 2026
# =============================================================================

import os
import numpy as np
import pandas as pd

# Set working directory
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR_ES_Python")
print("Working directory set to:", os.getcwd())

# -----------------------------------------------------------------------------
# 1. Reload raw CSVs from data_raw\
# -----------------------------------------------------------------------------
kse_raw = pd.read_csv("data_raw/kse100_raw.csv", parse_dates=["Date"], index_col="Date")
gold_raw = pd.read_csv("data_raw/gold_raw.csv", parse_dates=["Date"], index_col="Date")

print("\n--- Raw KSE-100 ---")
print("Shape:", kse_raw.shape)
print("Dtype:", kse_raw.dtypes)
print(kse_raw.head())

print("\n--- Raw Gold ---")
print("Shape:", gold_raw.shape)
print("Dtype:", gold_raw.dtypes)
print(gold_raw.head())

# -----------------------------------------------------------------------------
# 2. Fix KSE-100 — strip commas, convert Close to float
# -----------------------------------------------------------------------------
kse_raw["KSE100"] = kse_raw["KSE100"].astype(str).str.replace(",", "", regex=False).astype(float)

print("\n--- KSE-100 after comma fix ---")
print("Dtype:", kse_raw.dtypes)
print(kse_raw.head())

# -----------------------------------------------------------------------------
# 3. Fix Gold — flatten MultiIndex columns if present
# -----------------------------------------------------------------------------
if isinstance(gold_raw.columns, pd.MultiIndex):
    gold_raw.columns = ["_".join(col).strip() for col in gold_raw.columns]
    print("\nGold columns after flatten:", gold_raw.columns.tolist())

# Rename to clean single column name
gold_raw.columns = ["Gold"]

print("\n--- Gold after column fix ---")
print("Dtype:", gold_raw.dtypes)
print(gold_raw.head())

# -----------------------------------------------------------------------------
# 4. Inner join on common dates
# -----------------------------------------------------------------------------
prices = kse_raw.join(gold_raw, how="inner")
prices = prices.sort_index()

print("\n--- Combined prices after inner join ---")
print("Shape:", prices.shape)
print("Date range:", prices.index.min(), "to", prices.index.max())
print("Missing values:\n", prices.isnull().sum())
print(prices.head())

# -----------------------------------------------------------------------------
# 5. Compute daily log returns
# -----------------------------------------------------------------------------
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

print("\n--- Log Returns ---")
print("Shape:", returns.shape)
print("Date range:", returns.index.min(), "to", returns.index.max())
print("Missing values:\n", returns.isnull().sum())
print("\nDescriptive Statistics:")
print(returns.describe())

# -----------------------------------------------------------------------------
# 6. Save clean returns to data_clean\
# -----------------------------------------------------------------------------
returns.to_csv("data_clean/returns.csv")
print("\nClean returns saved to data_clean/returns.csv")

print("\nFiles in data_clean/:")
print(os.listdir("data_clean"))
