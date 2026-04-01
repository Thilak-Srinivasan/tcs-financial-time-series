# Financial Time Series & Portfolio Forecasting : TCS (2015-2025)

> **Applied Statistical Methods | MATH F432** | November 2025 \
> Thilak S · Under Prof. Sumanta Pasari \
> Department of Mathematics, BITS Pilani 

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-orange)](https://www.statsmodels.org/)
[![scipy](https://img.shields.io/badge/scipy-1.11-blue)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ez7CSYgPsVoVq-BXQyVm3YWgSKs_D4Cl?usp=sharing)
[![Report](https://img.shields.io/badge/Report-PDF-red?logo=adobe)](https://drive.google.com/file/d/1ml-DV5bJA8EZulhCa91VB0FBTXETR5tu/view?usp=share_link)

---

## Overview

This project presents a rigorous **econometric and statistical analysis** of Tata Consultancy Services (TCS) stock prices from **2015 to 2025** on the National Stock Exchange (NSE). Conducted as part of the Applied Statistical Methods course at BITS Pilani, the study investigates the stochastic nature of TCS stock returns, identifies the best-fit probability distribution, and evaluates the forecasting capability of classical linear time-series models.

The central finding — that TCS stock follows a **non-stationary random walk** (ARIMA(0,1,0)) with **heavy-tailed Student-t returns** (ν̂ = 3.606) — provides both a statistical and an economic interpretation consistent with the **Weak-Form Efficient Market Hypothesis**.

---

## Key Results at a Glance

| Metric | Value |
|---|---|
| Best-fit return distribution | Student-t (ν̂ = **3.606**) |
| Best-fit price distribution | Weibull (k̂ > 1) |
| ADF test — price level | Non-stationary (p = 0.357) |
| ADF test — first difference | Stationary, I(1) (p < 0.001) |
| Optimal ARIMA specification | **ARIMA(0, 1, 0)** — Random Walk |
| SARIMA AIC (best seasonal fit) | 26,819.8 (spurious — invertibility issue) |
| Residual kurtosis | 445.54 (extreme fat tails) |
| SIP recommendation | Monthly SIP — 12.44% CAGR |

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results & Discussion](#results--discussion)
- [Conclusions & Investor Recommendations](#conclusions--investor-recommendations)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [References](#references)

---

## Background

### Why TCS?

Tata Consultancy Services (TCS) is one of India's largest and most liquid IT sector companies, listed on the NSE and a constituent of the Nifty 50. With a market history spanning over two decades, TCS provides a rich dataset for exploring the statistical properties of large-cap Indian equity — particularly useful for examining market efficiency in an emerging economy context.

### Research Questions

1. What is the best-fit probability distribution for TCS closing prices and log-returns?
2. Is the TCS price series stationary, or does it follow a random walk?
3. Can ARIMA or SARIMA models meaningfully forecast future prices?
4. What practical investment strategies does the evidence support?

---

## Project Structure

```
tcs-financial-time-series/
│
├── data/
│   └── TCS_NSE_2015_2025.csv                  # Raw historical closing price data
│
├── notebooks/
│   └── TCS_Analysis.ipynb                     # Full analysis notebook (Google Colab)
│
├── outputs/
│   ├── figures/
│   │   ├── 01_intraday_volatility_profile.png
│   │   ├── 02_tcs_closing_price_2015_2025.png
│   │   ├── 03_distribution_closing_prices.png
│   │   ├── 04_moving_averages_50_200_day.png
│   │   ├── 05_stl_decomposition.png
│   │   ├── 06_rolling_mean_std_stationarity.png
│   │   ├── 07_acf_pacf_differenced_series.png
│   │   ├── 08_arima_forecast_zoomed.png
│   │   └── 09_sarima_forecast_zoomed.png
│   └── model_results/
│       ├── adf_test_results.csv
│       └── model_comparison_aic_bic.csv
│
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Preprocessing

Raw TCS closing price CSV files across multiple years were consolidated, standardized, and cleaned. Key preprocessing steps:

- **Header standardization** — reconciled `ClosePrice` vs. `Close Price` across different download formats
- **Numeric parsing** — stripped comma separators from price/volume columns
- **Date parsing & sorting** — enforced chronological ordering for valid rolling statistics
- **Missing value handling** — addressed non-trading days to maintain time-series continuity
- **Stock split adjustment** — the May 2018 2:1 stock split creates a structural break; addressed through log-return transformation rather than price-level analysis

### 2. Return Transformation

Prices are modelled as a Geometric Brownian Motion process:

```
dPt = μ·Pt·dt + σ·Pt·dWt
```

Raw closing prices were transformed into **log-returns**:

```
Rt = ln(Pt / Pt-1)
```

Log-returns are additive across time, approximately stationary, and provide a valid domain for distribution fitting and ARIMA modeling.

The intraday volatility profile below illustrates how price variance concentrates near market open and close — reinforcing the importance of return-based modelling over raw price levels.

![Intraday Volatility Profile](outputs/figures/01_intraday_volatility_profile.png)
*Figure 1: Intraday volatility profile of TCS — elevated variance at open (09:15–10:15 IST) and close (14:30–15:30 IST)*

### 3. Price Evolution & Stock Split Impact

The full 10-year closing price history shows TCS appreciating from ~₹1,750 to a peak above ₹4,500. The sharp discontinuity in mid-2018 reflects the **2:1 stock split** (May 2018) — a deterministic structural break, not a market crash.

![TCS Closing Price 2015–2025](outputs/figures/02_tcs_closing_price_2015_2025.png)
*Figure 2: TCS closing price (2015–2025). The steep drop in 2018 corresponds to the 2:1 stock split, not a market event. COVID-19 trough visible in early 2020.*

### 4. Distribution Fitting (MLE + AIC)

**Closing prices** — candidates tested via Maximum Likelihood Estimation:
- Normal → rejected (no lower bound; poor tail fit)
- Exponential → rejected (tail severely underestimated)
- Lognormal → reasonable but inadequate curvature fit
- **Weibull → best fit** by AIC/BIC (k̂ > 1, stretched right-skew)

The bimodal character of the empirical price histogram (peaks near ₹2,500 and ₹3,300) reflects the pre- and post-COVID price regimes rather than a true mixture distribution.

![Distribution of TCS Closing Prices](outputs/figures/03_distribution_closing_prices.png)
*Figure 3: Empirical distribution of TCS closing prices with Weibull KDE overlay. Bimodality reflects distinct price regimes across the decade rather than a mixture distribution.*

**Log-returns** — empirical return density exhibits:
- High central peak
- Symmetric heavy tails (leptokurtosis)
- **Student-t (ν̂ = 3.606) → best fit** — significantly heavier than Gaussian (ν → ∞), confirming elevated tail-risk

### 5. Trend Analysis — Moving Averages

The 50-day and 200-day Simple Moving Averages (SMAs) confirm the long-term upward trend. Classic **Golden Cross** (50-day crossing above 200-day) and **Death Cross** signals are visible around 2020–2021, coinciding with the COVID recovery rally.

![50-Day and 200-Day Moving Averages](outputs/figures/04_moving_averages_50_200_day.png)
*Figure 4: TCS close price with 50-day (orange) and 200-day (red) SMAs. The 2018 split discontinuity is visible; post-2020 recovery shows a sustained Golden Cross.*

### 6. Trend & Seasonality Decomposition

- **Multiplicative STL decomposition** applied (additive rejected — seasonal variance scales with trend level)
- Trend component: clear upward trajectory from ~₹2,000 → ₹4,000+
- Seasonal component: repetitive envelope oscillating around 1.00, consistent with quarterly earnings cycles and institutional rebalancing
- Residuals: heteroscedastic, with the 2018 split clearly visible as a concentrated outlier cluster

![STL Decomposition](outputs/figures/05_stl_decomposition.png)
*Figure 5: Multiplicative STL decomposition of TCS closing prices into trend, seasonal, and residual components. The residual spike at index ~750 corresponds to the 2018 stock split.*

### 7. Stationarity Testing — ADF

The Augmented Dickey-Fuller (ADF) test was applied at two levels:

| Series | ADF Stat | p-value | 1% CV | 5% CV | Result |
|---|---|---|---|---|---|
| Close Price (Level) | -1.847 | 0.357 | -3.433 | -2.863 | **Non-Stationary** |
| Close Price (1st Diff) | -49.356 | < 0.001 | -3.433 | -2.863 | **Stationary I(1)** |

The series is **Integrated of Order 1** — one round of differencing is required before any linear model can be applied. The rolling mean and standard deviation plot below confirms the time-varying mean (µt) and heteroscedasticity visually.

![Rolling Mean and Std — Stationarity Check](outputs/figures/06_rolling_mean_std_stationarity.png)
*Figure 6: Rolling mean (cyan) tracks the price trend closely, confirming a time-dependent mean µt. Rolling std (purple) spikes at the 2018 split — evidence of heteroscedasticity throughout the series.*

### 8. ACF / PACF Analysis

Both ACF and PACF of the differenced series showed **no statistically significant autocorrelation** at any positive lag — all values within 95% confidence bands. No cut-off or tail-off pattern characteristic of AR or MA structure. The series behaves as **white noise** post-differencing.

![ACF and PACF of Differenced Series](outputs/figures/07_acf_pacf_differenced_series.png)
*Figure 7: ACF (left) and PACF (right) of the first-differenced TCS series. All lags beyond zero fall within the 95% confidence interval (shaded band) — consistent with a white noise process and ARIMA(0,1,0).*

### 9. Time Series Forecasting Models

#### AR(2) and MA(2)
Both models produced statistically insignificant lag coefficients (p > 0.05), confirming no linear predictability in TCS returns.

#### ARIMA(0, 1, 0) — Random Walk
Optimal specification by AIC/BIC. Forecast trajectory is flat — conditional expectation Ŷ(t+1) = Y(t). Visually and statistically confirms **Weak-Form Efficient Market Hypothesis** for TCS.

![ARIMA Forecast — Zoomed on Tail](outputs/figures/08_arima_forecast_zoomed.png)
*Figure 8: ARIMA(0,1,0) 30-day forecast (dashed blue) on the last 100 observed trading days. The flat trajectory at ~₹3,600 is the hallmark of a random walk — the best linear forecast is simply the last observed price.*

#### ARMA(2, 2) — Over-parameterized
Near-cancellation of AR and MA coefficients (ϕ₁ ≈ −θ₁ = −0.55, 0.53) indicates parameter redundancy. Higher AIC (27,008) than simple random walk.

#### SARIMA(1,1,1)×(1,1,1)₃₀
Seasonal period s = 30 trading days. Achieved lower AIC (26,819.8) but:
- All coefficients statistically insignificant (p > 0.05)
- Seasonal MA coefficient Θ₃₀ ≈ −1.0 → **invertibility boundary failure**
- Concluded as spurious seasonal fit — random walk remains the superior model

![SARIMA Forecast — Zoomed on Tail](outputs/figures/09_sarima_forecast_zoomed.png)
*Figure 9: SARIMA(1,1,1)×(1,1,1)₃₀ 30-day forecast (blue dotted) on the last 150 observed trading days. Despite a marginally lower AIC, the near-flat trajectory and invertibility failure of Θ₃₀ ≈ −1.0 confirm this as a spurious fit.*

#### LSTM (Advanced — conceptual inclusion)
LSTM's capacity to model long-range sequential dependencies and non-linear volatility clustering makes it theoretically superior to linear ARIMA/SARIMA for volatile financial series. Discussed as a benchmark framing for the linear model limitations.

---

## Results & Discussion

### Model Comparison

| Model | Log-Likelihood | AIC | BIC | Verdict |
|---|---|---|---|---|
| AutoReg(2) | -13492.63 | 26993.26 | 27016.54 | Lowest AIC (best linear fit) |
| MA(2) | -13502.67 | 27013.33 | 27036.61 | Poor fit |
| ARMA(2,2) | -13498.43 | 27008.85 | 27043.77 | Over-parameterized |
| ARIMA(1,1,1) | -13502.42 | 27010.84 | 27028.30 | Redundant parameters |
| **ARIMA(0,1,0)** | — | — | — | **Preferred by parsimony** |
| SARIMA(1,1,1)×(1,1,1)₃₀ | — | 26819.8 | — | Spurious — invertibility issue |

Despite AutoReg(2) having the lowest raw AIC, its AR coefficients are individually insignificant (p > 0.05), meaning the marginal AIC improvement comes from the error variance term alone — not genuine predictive signal. ARIMA(0,1,0) is therefore adopted per the principle of parsimony.

### Three Core Econometric Findings

**Non-Stationarity & Random Walk** — TCS price is I(1). First-differenced returns behave as white noise. No linear signal extractable from historical prices.

**Heavy-Tailed Risk Profile** — Student-t with ν̂ = 3.606 implies extreme market events (crashes, rallies) occur far more frequently than a Gaussian model would predict. Standard deviation-based risk measures will systematically understate downside risk.

**Forecasting Efficacy** — ARIMA and SARIMA models serve as tools to *characterize* the random walk and *test* for seasonality — not to generate actionable price forecasts. Short-term (1–5 day) directional prediction is not statistically supported.

---

## Conclusions & Investor Recommendations

### 1. Adopt Disciplined, Systematic Investing
Since daily returns are essentially random, market timing fails statistically. A monthly SIP (Systematic Investment Plan) matches daily/weekly SIP returns (~12.44% CAGR) while minimizing administrative overhead and transaction costs.

### 2. Utilize Tail-Risk Quantifiers
Do **not** use normal distribution assumptions for risk. Use the Student-t parameters (ν̂ = 3.606) to compute:
- **Value-at-Risk (VaR)** at 95% and 99% confidence
- **Expected Shortfall (CVaR)** for true tail exposure

### 3. Limit Forecasting Utility
ARIMA/SARIMA are valid for market efficiency testing and short-range tactical horizon framing (1–5 days). Long-range price forecasting requires non-linear models capable of capturing volatility clustering — GARCH, Stochastic Volatility, or deep learning (LSTM) approaches.

---

## How to Run

### Option 1 — Google Colab (Recommended)

Click the badge at the top or open directly:  
[https://colab.research.google.com/drive/1ez7CSYgPsVoVq-BXQyVm3YWgSKs_D4Cl?usp=sharing](https://colab.research.google.com/drive/1ez7CSYgPsVoVq-BXQyVm3YWgSKs_D4Cl?usp=sharing)

All dependencies are pre-installed in the Colab environment.

### Option 2 — Local Setup

```bash
# Clone the repository
git clone https://github.com/Thilak-Srinivasan/tcs-financial-time-series.git
cd tcs-financial-time-series

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/TCS_Analysis.ipynb
```

---

## Dependencies

```txt
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
statsmodels>=0.14
scikit-learn>=1.3
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## References

1. **Hyndman, R. J., & Athanasopoulos, G.** (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts. https://otexts.com/fpp2/
2. **Quigley, M.** (2009). *Extreme Value Theory for Stock Returns*. Diplomarbeit, University of Muenster. https://www.uni-muenster.de/Stochastik/paulsen/Abschlussarbeiten/Diplomarbeiten/Quigley.pdf
3. **Nau, R.** (2020). *ARIMA(0,1,0) model with constant (random walk with drift)*. Duke University. https://people.duke.edu/~rnau/411arim.htm
4. **Investopedia.** *Autoregressive Integrated Moving Average (ARIMA)*. https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp
5. **Wikipedia.** *Autoregressive Integrated Moving Average*. https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
6. **GeeksforGeeks.** *SARIMA — Seasonal Autoregressive Integrated Moving Average*. https://www.geeksforgeeks.org/machine-learning/sarima-seasonal-autoregressive-integrated-moving-average/
7. **Phosgene89.** *ARIMA and SARIMAX Models*. https://phosgene89.github.io/sarima.html
8. **Williams, J.** (2023). *Stock Forecasting with the SARIMA Model*. Medium. https://medium.com/@juliawilliams_79854/stock-forecasting-with-the-sarima-model-edad16d37445
9. **Fiveable.** *Ljung-Box Test and White Noise Processes*. https://fiveable.me/intro-time-series/unit-4/ljung-box-test-white-noise-processes/study-guide/OIbpshoxfpjQSmGM
10. **NIST/SEMATECH.** *e-Handbook of Statistical Methods — Box-Ljung Test for Autocorrelation*. https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4481.htm
11. **Fiveable.** *Evaluating Forecast Accuracy: MAE, RMSE, MAPE*. https://fiveable.me/intro-time-series/unit-8/evaluating-forecast-accuracy-mae-rmse-mape/study-guide/ijqkb0CAqRaHLBFi
12. **Unofficed.** *Understanding the Results of Augmented Dickey-Fuller (ADF) Test on Stock Market*. https://unofficed.com/courses/risk-management/lessons/understanding-the-results-of-augmented-dickey-fuller-adf-test-on-stock-market/
13. **Zilliz AI.** *How do you choose parameters for an ARIMA model?* https://zilliz.com/ai-faq/how-do-you-choose-parameters-for-an-arima-model
14. **Hyndman, R. J., & Athanasopoulos, G.** (2018). *Stationarity and Differencing*. Forecasting: Principles and Practice. https://otexts.com/fpp2/stationarity.html
15. **Meghan.** *Stock Price Prediction using ARIMA and SARIMA*. GitHub Repository. https://github.com/M3GHAN/stock-price-prediction-ARIMA-SARIMA

---

> *"The closing values of TCS are modeled best using a Weibull distribution, while the log-return data are described by a heavy-tailed Student-t distribution. This result supports recent developments in financial theory that emphasize return-based modeling in contrast to raw level price modeling."*  
> — ASM G3 Report, November 2025

---

**Course:** MATH F432 - Applied Statistical Methods \
**Supervisor:** Prof. Sumanta Pasari | Associate Professor | Department of Mathematics, BITS Pilani \
**Period:** October–December 2025 \
**Contact:** f20220771@pilani.bits-pilani.ac.in
