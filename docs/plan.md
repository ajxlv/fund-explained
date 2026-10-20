# Fund Explained: one-page plan

**Pitch:** pick any Indian mutual fund and get a plain-English report card: what ₹10,000 would have become, how bad its worst fall was, how risky it is compared with the Nifty, which funds it secretly behaves like, and whether it acts like its official category. **It explains and never recommends.**

## Scope (frozen: anything else goes on a "Later" list)
1. **Report card:** ₹10,000 invested 1, 3 and 5 years ago, CAGR, worst fall (max drawdown) and how long it took to recover.
2. **Risk vs Nifty:** volatility, beta ("moves 0.95× as much as the Nifty"), and correlation, with a plain-English risk label.
3. **Lookalikes (ML):** the 5 funds whose daily returns move most like this one, for example "moves 98% the same as X".
4. **Label check (ML):** a classifier predicts each fund's category from its behaviour. Funds whose behaviour confidently matches a *different* category get flagged ("labelled Flexi Cap, behaves like Large Cap").
5. **Glossary:** one-line explanations of NAV, SIP, CAGR, drawdown, beta, volatility, direct plan and so on, shown as tooltips everywhere.

**Later (not now):** expense ratios and fee impact, SIP calculator, portfolio upload, LLM chat.

## Data
- **mfapi.in** gives free NAV history plus `fund_house` and `scheme_category` for every Indian fund. [Certain, checked on 19 Oct 2026]
- **Universe:** open-ended equity schemes, Direct plan, Growth option, with at least 5 years of history. That's probably 300–500 funds. [Guessing on the count]
- **Nifty proxy:** a Nifty 50 index fund's NAV, so we never need to touch NSE.
- **Network:** mfapi is blocked in my environment and possibly in cloud Claude Code. So **one fetch script runs on your laptop**, and the cleaned NAV data (Parquet, about 5–10 MB) is committed to the repo. Everything else runs offline from that file.

## ML, honestly evaluated
- **Lookalikes:** correlation of daily returns over the last 3 years, giving the nearest neighbours. Check: index funds tracking the same index should be each other's top match. That's a built-in sanity test.
- **Clustering:** k-means on standardised behaviour features, compared with the official categories using the adjusted Rand index. Report the number even if it's low, because it *is* the finding.
- **Label check:** logistic regression and a random forest predict category from 8 features. Report 5-fold stratified CV accuracy and the confusion matrix, and only flag a fund when it's misclassified with probability ≥ 0.7.

## Stack
Python 3.12, pandas, numpy, scikit-learn, Streamlit (simplest UI, free hosting on Streamlit Community Cloud), Plotly for charts, pytest, ruff, and GitHub Actions CI.

## Timeline (2 weeks)
| Days | Work |
|---|---|
| 1 | Laptop: fetch data, commit Parquet |
| 2–4 | Metrics module + tests (report card, risk) |
| 5–7 | Lookalikes + clustering + label check + evaluation report |
| 8–10 | Streamlit app: search, report card, lookalikes, label check, glossary |
| 11–12 | Deploy, README with screenshots, write-up of findings |
| 13–14 | Buffer, plus your own read-through of the code |

## Resume bullets (fill in real numbers)
- Built **Fund Explained**, a Streamlit app that turns NAV history for **N** Indian mutual funds into beginner-friendly report cards (growth, worst fall, risk vs Nifty), using pandas and a tested metrics library.
- Used return-correlation nearest neighbours and k-means clustering to find lookalike funds, and a category classifier (**X%** CV accuracy) to flag **M** funds whose behaviour doesn't match their official category.
