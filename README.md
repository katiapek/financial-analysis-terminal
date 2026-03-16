# Markets Manners

Institutional-grade positioning, seasonality, and regime analysis for futures & options traders.

## What This Is

An automated data pipeline and report generation system that:

- Ingests weekly CFTC Commitment of Traders (COT) positioning data
- Computes regime classifications based on positioning, seasonality, volatility, and CTA flow models
- Generates a weekly analytical report for commodity and equity index futures markets
- Produces publication-quality charts for social media distribution

## Markets Covered

| Market | Complex | Key Analysis |
|--------|---------|-------------|
| NQ (E-mini Nasdaq 100) | Equity Index | TFF positioning, CTA levels, weekly options regimes |
| GC (Gold) | Precious Metals | Commercial hedging, macro regime, seasonal patterns |
| CL (WTI Crude Oil) | Energy | Producer positioning, OPEC cycles, seasonal + regime |
| ZC (Corn) | Agriculture | Planting/harvest seasonality, USDA cycle regimes |

## Tech Stack

Python · pandas · matplotlib · yfinance · CFTC COT API · CME CVOL API

## Author

[@marketsmanners](https://x.com/marketsmanners) — 6 years in capital markets (futures derivatives, options on futures)

## Disclaimer

This project is for educational and informational purposes only. Nothing produced by this system constitutes investment advice or a recommendation to buy or sell any securities. Trading futures and options involves substantial risk of loss. Past performance is not indicative of future results.
