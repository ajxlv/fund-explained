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
