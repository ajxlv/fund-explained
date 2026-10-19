# Overnight build review

## The one thing to know first

**Everything was built and tested on synthetic demo data.** `api.mfapi.in` was blocked from the build environment (proxy `403 Forbidden` on every request, including `scripts/fetch_navs.py` after its 4 retries). The app, tests and reports are wired to switch to real data automatically once `data/schemes.csv` and `data/navs.parquet` exist. `docs/MORNING_STEPS.md` has the exact PowerShell commands. Until then, **every number in EVALUATION.md and the README is a demo number and says nothing about real funds.**

## What was built

| Part | Files |
|---|---|
| Plan + fetch script (verbatim from the brief) | `docs/plan.md`, `scripts/fetch_navs.py` |
| Synthetic demo data (60 "Demo …" funds, fixed seed) | `scripts/make_demo_data.py`, `data/demo/` |
| Data layer: real-or-demo loader, trading-day calendar, ≤5-day forward-fill, returns | `fundx/data.py` |
| Metrics: growth of ₹10k, CAGR, drawdown + recovery, volatility, beta, correlation, risk label | `fundx/metrics.py` |
| Nifty 50 proxy picker | `fundx/benchmark.py` |
| Lookalikes (3-year return correlation, ≥600 shared days) | `fundx/lookalikes.py` |
| 8 behaviour features | `fundx/features.py` |
| k-means + PCA, category classifier (LR + RF, stratified CV, baseline), flags, report generator | `fundx/ml.py` |
| Report card data and plain-English wording | `fundx/report.py` |
| Glossary, ₹ formatting, fuzzy search | `fundx/glossary.py`, `fundx/fmt.py`, `fundx/search.py` |
| Streamlit app: report card page + explore page, theme, CSS, Plotly charts | `app/`, `.streamlit/config.toml` |
| CI (ruff + pytest on push and PR) | `.github/workflows/ci.yml` |
| Docs | `README.md`, `docs/DECISIONS.md`, `DATA_SUMMARY.md`, `EVALUATION.md`, `DEPLOY.md`, `MORNING_STEPS.md`, `LATER.md`, this file |

## Tests

**135 tests: 134 passed, 1 skipped** (`pytest`), and `ruff check .` is clean (ruff pinned to 0.17.1).

- The skip is the real-data sanity test ("each Nifty 50 index fund's top lookalike is another one, corr ≥ 0.99, for ≥ 90% of them"). It skips with a clear reason on demo data. I checked that it runs and passes when data sits in `data/` by temporarily copying the demo files there, and a separate demo-only test covers the same logic on the 3 synthetic index funds.
- The metric tests use hand-built series, as asked: √2 − 1 for a 2-year doubling, the 100→120→90→125 drawdown path, beta 1 and 2, and None for too-short history.
- `tests/test_app.py` runs the real Streamlit app headless (`AppTest`): the landing page, several fund pages and the filters. It caught a real crash during the build.
- `tests/test_wording.py` scans every code and doc file outside `tests/` for "buy", "sell", "best fund" and "you should". Those words appear only in `tests/wording.py`, which is the guard itself.
- I also checked the app in Chromium (Playwright) at 1280 px and 390 px, in light and dark mode: no exceptions and no horizontal scroll.

## Key numbers (demo data, so they mean nothing about real funds)

- 60 funds, 1,512 trading days (31 Dec 2020 to 16 Oct 2026). One fund has a 12-day gap.
- Lookalike sanity: 100% (3 of 3 demo index funds, corr ≈ 0.9998).
- k-means: best k = 3 by silhouette (0.37), ARI 0.24. At k = 6 (number of categories), ARI 0.40.
- Classifier (5-fold stratified CV): random forest 82.5% accuracy / 0.84 macro-F1, logistic regression 70.2% / 0.71, majority baseline 17.5%. The 3 index funds were left out (their merged class was too small to cross-validate).
- 3 funds flagged as "behaves like another category" (Flexi Cap and Mid Cap swapping places, which the generator makes plausible because their loadings overlap).

## Decisions worth a look (full list in `docs/DECISIONS.md`)

1. **Returns come from the *unfilled* NAVs.** Forward-filling a missing day creates a fake 0% return followed by a two-day jump. In the demo data that pulled two index funds' correlation from 0.9997 to 0.986, enough to fail the sanity check for the wrong reason. The forward-filled matrix (gaps ≤ 5 days) is used only for NAV-level numbers and charts.
2. **Trading days = dates on which ≥ 50% of the live funds published a NAV**, so one fund's stray holiday NAV can't punch a hole in everyone else's returns.
3. **Merged "Other equity" class still < 8 funds → left out of the classifier** rather than cross-validating a 3-member class. The app tells those funds that the check couldn't run.
4. **Flags come from the model with the better CV macro-F1**, never pointing at "Other equity".
5. **The evaluation report on the Explore page is generated live**, so it always matches the data the app has loaded.
6. **ruff is pinned** because its newer default rule set differs from older versions.

## Six interview questions (with answers)

**1. Why correlation for lookalikes and not k-means distance?**
Lookalikes answer "which funds move *together* day to day?". Correlation of daily returns measures exactly that, across ~750 shared days, and it ignores scale: a fund that moves 1.2× the Nifty every day still correlates ~1 with it. k-means distance works on 8 summary features (CAGR, volatility, …). Two funds can have similar summaries while moving on different days, for example two sector funds with similar volatility in unrelated sectors. Clusters also give only a coarse group, while correlation gives a ranked top 5 per fund. The built-in check backs this up: index funds tracking the same index should be each other's top match, and with correlation they are.

**2. Why stratified cross-validation?**
The classes are small and uneven (on real data roughly 8 to 40 funds per category). Plain K-fold can leave a small category almost or entirely out of a test fold, which makes accuracy and macro-F1 swing with the random split. Stratified folds keep each category's share the same in every fold, so each fold is a fair, smaller copy of the dataset. All reported numbers are out-of-fold: every fund is scored by a model that never saw it.

**3. What's survivorship bias here?**
The universe is funds that exist *today* and have ≥ 5 years of history. Funds that were closed or merged away, often because they did badly, aren't in the data at all. Averages such as "median 5-year CAGR of Small Cap funds" are therefore computed only over survivors and look better than what an investor in the category actually got. It also slightly skews the risk labels, because those are relative to the surviving funds.

**4. Why is beta computed on daily returns?**
Beta is a regression slope of fund returns on market returns, and daily returns give ~750 paired points over 3 years instead of ~36 monthly ones, so the estimate is much less noisy. Returns, not NAV levels, are what you regress: two trending price series correlate highly even when they're unrelated (spurious regression). The trade-off is that daily data can understate beta for funds holding less-liquid stocks, whose prices react with a lag. That's a known caveat for small caps.

**5. Why report a majority-class baseline and macro-F1, not just accuracy?**
With uneven classes, "always guess the biggest category" already gets some accuracy for free. The baseline shows how much the model actually adds (here, on demo data, 82.5% vs 17.5%). Macro-F1 averages F1 over categories equally, so a model can't look good by doing well only on the big categories and ignoring small ones.

**6. Why flag only at ≥ 0.7 out-of-fold probability, and why the careful wording?**
Every model mislabels some funds just from noise. Requiring a confident vote for a *different* category, from a model that never trained on that fund, keeps flags to the cases where behaviour is clearly different. The wording is careful because SEBI categories are rules about *holdings* (for example, Large Cap = mostly the top 100 stocks). A Flexi Cap fund that happens to hold mostly large companies will *behave* like Large Cap while being correctly labelled. So the app says "check its portfolio", not "mislabelled".

## Blocked

- **Real data fetch:** blocked by the build environment's network policy (mfapi.in → 403). Not retried beyond the script's own 4 attempts. Fix: `docs/MORNING_STEPS.md` on your laptop.

## Left unfinished, needs real data or you

- Run the fetch, then `python -m fundx.data --summary`, `python -m fundx.ml --report` and `pytest` (the real-data sanity test will run for the first time). If that test fails on real data, the most likely cause is a bad NAV print in one index fund (a one-day spike). Check the failing pairs in EVALUATION.md before changing anything.
- Replace the demo numbers in the README "Key findings" table, and fill in the N / X% / M in the resume bullets in `docs/plan.md`.
- Retake `docs/screenshot.png` on real data. The current one shows the demo banner on purpose.
- Deploy (steps in `docs/DEPLOY.md`). Not done, as instructed.
- First load on real data (~400 to 500 funds) will take roughly 10 to 20 seconds while features and models are computed. After that it's cached. This is an estimate: I couldn't time it on real data.
