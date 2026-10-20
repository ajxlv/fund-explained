# Decisions log

Choices made during the overnight build without being able to ask. Each one takes the simplest honest option.

## Data

1. **The build uses demo data.** `api.mfapi.in` was blocked in the build environment: the egress proxy returned `403 Forbidden` on every request, and `scripts/fetch_navs.py` failed after its 4 retries. Following the plan, `scripts/make_demo_data.py` generates a clearly synthetic dataset in `data/demo/` instead. No real-looking data was invented. `docs/MORNING_STEPS.md` covers the real fetch.
2. **Demo dataset design.** 57 active funds across 5 equity categories, plus 3 "Demo Nifty 50 Index Fund" series, all from "Demo AMC" with codes starting at 900001 (real AMFI codes are lower). Fund names end in "- Direct Plan - Growth" so they match the real naming that `looks_direct_growth` expects. Each daily return is a market factor, a size factor and (for sector funds) a banking or tech factor, plus noise, on 1,512 weekday "trading days" ending on 2026-10-16. The seed and end date are fixed so the data is reproducible. To exercise the gap handling, about 0.5% of days are dropped at random, one fund has a 12-day gap and one has a 4-day gap. Six funds launch later but still have at least 5 years of history.
3. **The demo data is committed** (`data/demo/`, about 0.8 MB). CI and a Streamlit Cloud deploy can then run without the network. Real data, once fetched, goes in `data/` and takes priority automatically.
4. **Ruff exception for `scripts/fetch_navs.py`.** The script was copied verbatim as instructed. The current ruff defaults flag its `# noqa: BLE001` as unused (RUF100), so `pyproject.toml` ignores RUF100 for that one file rather than editing it.

## Alignment and gaps

5. **Trading calendar = the union of all dates in the dataset.** There is no external holiday calendar, so a "trading day" is any date on which at least one fund has a NAV.
6. **Forward-fill only interior gaps of 5 trading days or fewer** (`fundx.data.ffill_short_gaps`). Longer gaps stay fully NaN. Plain `ffill(limit=5)` would partly fill a long gap, so it isn't used. Nothing is filled before a fund's first NAV or after its last.
7. **Daily returns are NaN when either day's NAV is missing.** After a long gap, the return that spans the gap is dropped rather than spread across the missing days.

## Metrics

8. **Recovery time is measured from the trough**, in calendar days until the NAV first gets back to the previous peak. The app shows it in months (days / 30.44).
9. **The drawdown window** starts at the same point as growth/CAGR: the last NAV on or before (latest date − 5 years). With less history than that, it returns None.
10. **Beta, correlation and tracking error need at least 20 paired daily returns** (`min_obs`) and use only dates where both the fund and the benchmark have a return.
11. **29 February anniversaries** follow the pandas `DateOffset` convention: 29 Feb 2020 + 1 year = 28 Feb 2021.
12. **The Nifty 50 proxy** matches `nifty\s*50` not followed by a digit, so "Nifty 500" is excluded. The rule also requires "index" in the category, Direct + Growth in the name, and none of next/equal/value/momentum/quality/low vol. Longest history means the earliest `first_date`, with the NAV count as the tie-break. The Direct+Growth check is a small copy of the one in `scripts/fetch_navs.py`, which must not be edited or imported as a package.
13. **Indian rupee formatting** (`fundx.fmt.fmt_inr`) lives in the package, not the app, so it can be tested without Streamlit.
14. **Returns come from the unfilled NAVs; the forward-filled matrix is only for NAV-level numbers and charts.** A forward-filled day gives a fake 0% return followed by a two-day jump, which distorted the correlations noticeably: two demo index funds dropped from 0.9997 to 0.986. Missing days therefore drop their returns instead.
15. **Trading days exclude stray dates.** A date counts as a trading day only if at least half of the funds alive on that date published a NAV. Otherwise one fund's holiday NAV would leave a hole in every other fund's returns.

## ML

16. **Funds with any missing feature are left out of the ML** (clustering and classifier). Missing values are not imputed. The report gives the count used.
17. **The k for "number of distinct equity categories"** is the number of distinct `scheme_category` values in the universe, index funds included, computed before merging small categories.
18. **When the merged "Other equity" class still has fewer than 8 funds, its funds are left out of the classifier.** Five-fold stratified CV can't evaluate a class that small. In the demo data this applies to the 3 index funds, and the app tells the user the check couldn't run for those funds.
19. **Flags use the model with the higher cross-validated macro-F1**, and that choice is logged in the report. "Behaves like Other equity" is never flagged because it isn't a meaningful statement.
20. **Model settings:** logistic regression is `StandardScaler` + `LogisticRegression(max_iter=5000)`, multinomial by default in scikit-learn 1.9. The random forest has 500 trees. KMeans uses `n_init=10`. Everything uses `random_state=42`.

## App

21. **Report-card logic and wording live in `fundx/report.py`, not in the app**, so the sentences (including the "never recommend" rule) are unit-tested. `tests/test_app.py` runs the real app headless with Streamlit's `AppTest` (landing page, several fund pages, filters), so CI catches app crashes too.
22. **Two pages via `st.navigation`**: "Report card" and "Explore all funds". The Explore page renders the evaluation report **live** from the data the app is using (same text as `docs/EVALUATION.md`), so it can never be out of sync.
23. **Dark mode** uses Streamlit's `[theme.light]` / `[theme.dark]` config sections, which follow the system setting. The custom CSS follows the same setting via `prefers-color-scheme`. The hamburger menu is hidden, so the two can't disagree. Plotly colours pick light or dark steps via `st.context.theme`.
24. **Chart colours were checked with the dataviz palette validator.** The fund line is indigo (#4F46E5 light, #7C7FF2 dark). The Nifty line is slate (#64748B), deliberately low in colour and dashed: it's a reference line, and the validator's "reads grey" warning is intended. Separation between the two passes in both modes. The legend shows each line's end value, so the lines aren't told apart by colour alone. Clusters use the validated 8-hue categorical order plus marker shapes; any clusters past 8 are grey.
25. **Fuzzy search uses stdlib `difflib`** (no new dependency). The search matches every query word by prefix or a ≥0.6 similarity, in any order.
26. **Example funds** on the landing page are well-known schemes when present (Parag Parikh Flexi Cap, HDFC Flexi Cap, ICICI Pru Bluechip/Large Cap, …). AUM isn't in the data, so "large" can't be measured. Otherwise the app falls back to the oldest Flexi Cap fund, the oldest Large Cap fund and the Nifty proxy.
27. **ruff is pinned (0.17.1) in `requirements-dev.txt`.** Recent ruff versions changed the default rule set (ISC, C4 and RUF rules now fire), so CI and laptops must agree on the version.
28. **Windows-safe output.** Dates are formatted without `strftime("%-d")`, which raises on Windows. `--report` and `--summary` print a one-line ASCII confirmation rather than the whole Unicode report, so they can't hit a `cp1252` console encoding error.
29. **The real-data path was exercised before handing over** by copying the demo files into `data/` temporarily: all tests passed, the real-data sanity test ran instead of skipping, and the report had no demo header. The copies were deleted afterwards.
