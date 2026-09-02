# =============================================================================
# Script 01 — Data Download and Inspection
# Tail Risk in Pakistani Financial Markets — VaR and ES in Python
# Ahmer | GitHub: ahmer-econ | 2026
# =============================================================================

import os
import pandas as pd
import yfinance as yf

# Set working directory
os.chdir(r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR_ES_Python")
print("Working directory set to:", os.getcwd())

# -----------------------------------------------------------------------------
# 1. Load KSE-100 from Kaggle CSV
# -----------------------------------------------------------------------------
print("\n--- Loading KSE-100 from CSV ---")

kse_raw = pd.read_csv(
    r"D:\Documents\APPLIED ECONOMETRICS WORK\VaR_ES_Pakistan Model\data_raw\KSE100-20years.csv",
    parse_dates=["Date"]
)

# Sort ascending (CSV is newest-first)
kse_raw = kse_raw.sort_values("Date").reset_index(drop=True)

print("Shape:", kse_raw.shape)
print("\nFirst 5 rows:")
print(kse_raw.head())
print("\nLast 5 rows:")
print(kse_raw.tail())
print("\nDate range:", kse_raw["Date"].min(), "to", kse_raw["Date"].max())

# -----------------------------------------------------------------------------
# 2. Download COMEX Gold from Yahoo Finance
# -----------------------------------------------------------------------------
print("\n--- Downloading COMEX Gold (GC=F) ---")

gold_raw = yf.download("GC=F", start="2015-01-01", end="2024-12-31", auto_adjust=True)

print("Shape:", gold_raw.shape)
print("\nFirst 5 rows:")
print(gold_raw.head())
print("\nLast 5 rows:")
print(gold_raw.tail())

# -----------------------------------------------------------------------------
# 3. Extract closing prices and align to common sample
# -----------------------------------------------------------------------------

# KSE-100: keep Date and Close only, set Date as index
kse_close = kse_raw[["Date", "Close"]].copy()
kse_close = kse_close.rename(columns={"Close": "KSE100"})
kse_close = kse_close.set_index("Date")

# Gold: keep Close only, rename
gold_close = gold_raw[["Close"]].copy()
gold_close.columns = ["Gold"]

# Trim KSE-100 to match Gold sample start (2015-01-01)
kse_close = kse_close[kse_close.index >= "2015-01-01"]

print("\n--- After trimming to common sample ---")
print("KSE-100 shape:", kse_close.shape)
print("KSE-100 date range:", kse_close.index.min(), "to", kse_close.index.max())
print("\nGold shape:", gold_close.shape)
print("Gold date range:", gold_close.index.min(), "to", gold_close.index.max())

# -----------------------------------------------------------------------------
# 4. Save raw closing prices to data_raw\
# -----------------------------------------------------------------------------
kse_close.to_csv("data_raw/kse100_raw.csv")
gold_close.to_csv("data_raw/gold_raw.csv")
print("\nRaw files saved to data_raw/")

# -----------------------------------------------------------------------------
# 5. Confirm saved files
# -----------------------------------------------------------------------------
print("\nFiles in data_raw/:")
print(os.listdir("data_raw"))