# Tail Risk in Pakistani Financial Markets — VaR and ES in Python

Value at Risk (VaR) and Expected Shortfall (ES) estimation for the KSE-100 equity index and COMEX Gold futures using three methods — Historical Simulation, Parametric Normal, and GARCH(1,1) Filtered Historical Simulation — with formal backtesting via the Kupiec POF and Christoffersen Independence tests.

**Ahmer | GitHub: ahmer-econ | 2026**

---

## Assets and Sample

| Asset | Ticker | Source | Sample |
|---|---|---|---|
| KSE-100 Index | — | Kaggle historical dataset | Jan 2015 – Aug 2024 |
| COMEX Gold Futures | GC=F | Yahoo Finance (yfinance) | Jan 2015 – Aug 2024 |

2,288 daily log return observations after inner join on common trading dates.

---

## Methods

**Historical Simulation (HS)**
VaR estimated as the empirical percentile of the return distribution. No distributional assumption. ES is the average of all returns below the VaR threshold.

**Parametric (Normal)**
VaR estimated as mean minus z-score times standard deviation, assuming normally distributed returns. ES derived analytically from the normal density.

**GARCH(1,1) Filtered Historical Simulation (GARCH-FHS)**
GARCH(1,1) fitted to extract conditional volatility. Standardised residuals computed. Historical Simulation applied to standardised residuals and scaled by the one-step-ahead volatility forecast. Adapts to current volatility conditions while retaining the empirical tail distribution.

Confidence levels: **95% and 99%**
Rolling backtest window: **250 trading days**

---

## Key Results

### GARCH(1,1) Parameter Estimates

| Parameter | KSE-100 | COMEX Gold |
|---|---|---|
| ω (omega) | 0.000713 | 0.000139 |
| α (alpha) | 0.1352 | 0.0340 |
| β (beta) | 0.8088 | 0.9509 |
| Persistence (α+β) | 0.9440 | 0.9850 |

### VaR and ES Estimates (full sample)

| Asset | Level | Method | VaR | ES |
|---|---|---|---|---|
| KSE-100 | 99% | Historical Simulation | −3.24% | −4.34% |
| KSE-100 | 99% | Parametric | −2.53% | −2.90% |
| KSE-100 | 99% | GARCH-FHS | −2.71% | −3.23% |
| Gold | 99% | Historical Simulation | −2.53% | −3.41% |
| Gold | 99% | Parametric | −2.20% | −2.53% |
| Gold | 99% | GARCH-FHS | −2.65% | −3.77% |

### Backtesting Summary (2,038 out-of-sample observations)

| Asset | Method | Level | Violations | Expected | Kupiec p | Christ. p | Pass? |
|---|---|---|---|---|---|---|---|
| KSE-100 | HS | 95% | 119 | 102 | 0.090 | 0.000 | ❌ |
| KSE-100 | Parametric | 95% | 106 | 102 | 0.679 | 0.001 | ❌ |
| KSE-100 | GARCH-FHS | 95% | 119 | 102 | 0.090 | 0.000 | ❌ |
| KSE-100 | HS | 99% | 34 | 20 | 0.006 | 0.019 | ❌ |
| KSE-100 | Parametric | 99% | 37 | 20 | 0.001 | 0.179 | ❌ |
| KSE-100 | GARCH-FHS | 99% | 34 | 20 | 0.006 | 0.019 | ❌ |
| Gold | HS | 95% | 109 | 102 | 0.475 | 0.710 | ✅ |
| Gold | Parametric | 95% | 107 | 102 | 0.607 | 0.867 | ✅ |
| Gold | GARCH-FHS | 95% | 109 | 102 | 0.475 | 0.710 | ✅ |
| Gold | HS | 99% | 28 | 20 | 0.108 | 0.400 | ✅ |
| Gold | Parametric | 99% | 37 | 20 | 0.001 | 0.703 | ❌ |
| Gold | GARCH-FHS | 99% | 28 | 20 | 0.108 | 0.400 | ✅ |

---

## Main Findings

- For COMEX Gold, Historical Simulation and GARCH-FHS pass all Kupiec and Christoffersen tests at both confidence levels. Violations arrive at the correct frequency and independently over time.
- For KSE-100, all methods fail backtesting. Violations cluster during crisis periods (2017, 2020) rather than arriving randomly — a structural feature of Pakistan's frontier equity market, not a failure of the methods.
- The Parametric Normal method is the worst performer at 99% for both assets, generating the most violations and failing the Kupiec test. The normality assumption cannot capture fat-tailed return distributions (excess kurtosis: KSE-100 = 4.25, Gold = 10.48).
- GARCH-FHS matches HS in violation counts while incorporating a time-varying volatility forecast, making it more suitable for real-time risk management.

---
## Requirements

pandas
numpy
scipy
matplotlib
seaborn
yfinance
arch
statsmodels


Install with:

```bash
pip install yfinance arch statsmodels seaborn
```

pandas, numpy, scipy, and matplotlib are included in the Anaconda base distribution.

---

## How to Reproduce

1. Clone the repository
2. Install dependencies (see above)
3. Run scripts in order: `01_data.py` through `06_violations_plot.py`
4. Each script sets its working directory at the top — update the path if your folder location differs
5. All outputs are saved automatically to `outputs/figures/` and `outputs/tables/`

---

## Related Work

This project replicates the methodology of an earlier R-based implementation of the same analysis. The Python version demonstrates equivalent results using pandas, NumPy, SciPy, and the arch library in place of R's rugarch and PerformanceAnalytics packages.

---

## References

- Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management* (Revised ed.). Princeton University Press.
- Sheppard, K. (2024). arch: ARCH and other tools for financial econometrics. https://github.com/bashtage/arch

## Repository Structure
