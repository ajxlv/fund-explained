"""Generate a clearly SYNTHETIC demo dataset, used only when real mfapi.in data is unavailable.

    python scripts/make_demo_data.py

Writes data/demo/schemes.csv and data/demo/navs.parquet in the same format as
scripts/fetch_navs.py. Every fund is called "Demo ..." from "Demo AMC" so it can never be mistaken
for a real scheme.

Model: daily return = beta * market + size_loading * size + sector_loading * sector + noise,
with one loading set per category, 6 years of weekday "trading days" and a fixed seed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261019
END_DATE = "2026-10-16"  # fixed so the dataset is identical on every machine
N_DAYS = 6 * 252

# category -> (number of funds, beta range, size-loading range, noise-sd range)
CATEGORIES = {
    "Equity Scheme - Large Cap Fund": (12, (0.88, 1.00), (-0.10, 0.10), (0.002, 0.004)),
    "Equity Scheme - Mid Cap Fund": (12, (0.95, 1.10), (0.40, 0.70), (0.003, 0.005)),
    "Equity Scheme - Small Cap Fund": (11, (0.90, 1.10), (0.90, 1.30), (0.004, 0.007)),
    "Equity Scheme - Flexi Cap Fund": (12, (0.85, 1.05), (0.00, 0.50), (0.003, 0.005)),
    "Equity Scheme - Sectoral/ Thematic": (10, (0.60, 0.90), (-0.10, 0.30), (0.004, 0.006)),
}
SHORT_NAMES = {
    "Equity Scheme - Large Cap Fund": "Large Cap",
    "Equity Scheme - Mid Cap Fund": "Mid Cap",
    "Equity Scheme - Small Cap Fund": "Small Cap",
    "Equity Scheme - Flexi Cap Fund": "Flexi Cap",
    "Equity Scheme - Sectoral/ Thematic": "Sector",
}
INDEX_CATEGORY = "Other Scheme - Index Funds"


def simulate_factors(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    # Fat-tailed market (Student-t, df=4) scaled to ~17% annual volatility, ~13% annual drift.
    market = 0.0005 + 0.011 * rng.standard_t(4, n) / np.sqrt(2.0)
    market[300:340] += -0.004  # one sharp sell-off so drawdowns and recoveries are visible
    return {
        "market": market,
        "size": 0.0001 + 0.006 * rng.standard_normal(n),
        "banking": 0.008 * rng.standard_normal(n),
        "tech": 0.009 * rng.standard_normal(n),
    }


def make_funds(rng: np.random.Generator) -> list[dict]:
    funds = []
    code = 900001
    for cat, (count, beta_r, size_r, noise_r) in CATEGORIES.items():
        for j in range(count):
            fund = {
                "scheme_code": code,
                "scheme_name": f"Demo {SHORT_NAMES[cat]} Fund {chr(65 + j)} - Direct Plan - Growth",
                "scheme_category": cat,
                "beta": rng.uniform(*beta_r),
                "size": rng.uniform(*size_r),
                "sector": None,
                "sector_loading": 0.0,
                "noise": rng.uniform(*noise_r),
                "alpha": rng.normal(0.0, 0.00008),
            }
            if cat == "Equity Scheme - Sectoral/ Thematic":
                fund["sector"] = "banking" if j % 2 == 0 else "tech"
                fund["sector_loading"] = rng.uniform(0.8, 1.2)
            funds.append(fund)
            code += 1
    for j in range(3):
        funds.append({
            "scheme_code": code,
            "scheme_name": f"Demo Nifty 50 Index Fund {chr(65 + j)} - Direct Plan - Growth",
            "scheme_category": INDEX_CATEGORY,
            "beta": 1.0, "size": 0.0, "sector": None, "sector_loading": 0.0,
            "noise": 0.0002,  # tiny tracking error
            "alpha": -0.000008 * (j + 1),  # a small expense drag
        })
        code += 1
    return funds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS)
    factors = simulate_factors(rng, N_DAYS)
    funds = make_funds(rng)

    # A few funds launch later (still >= 5 years of history), mimicking newer schemes.
    late_start = {int(i): int(rng.integers(60, 200)) for i in rng.choice(57, size=6, replace=False)}

    rows, navs = [], []
    for i, f in enumerate(funds):
        r = (f["alpha"] + f["beta"] * factors["market"] + f["size"] * factors["size"]
             + f["noise"] * rng.standard_normal(N_DAYS))
        if f["sector"]:
            r = r + f["sector_loading"] * factors[f["sector"]]
        nav = rng.uniform(10, 100) * np.cumprod(1 + r)
        s = pd.Series(np.round(nav, 4), index=dates)
        s = s.iloc[late_start.get(i, 0):]

        # Missing publication days: drop ~0.5% of days at random. Missing data stays missing.
        drop = rng.random(len(s)) < 0.005
        drop[0] = drop[-1] = False
        s = s[~drop]
        if i == 5:  # one fund with a long 12-trading-day gap (exercises the gap report)
            s = s.drop(s.index[400:412])
        if i == 20:  # one fund with a short 4-trading-day gap (allowed to be forward-filled)
            s = s.drop(s.index[700:704])

        rows.append({
            "scheme_code": f["scheme_code"],
            "scheme_name": f["scheme_name"],
            "fund_house": "Demo AMC",
            "scheme_category": f["scheme_category"],
            "scheme_type": "Open Ended Schemes",
            "first_date": s.index[0].date(),
            "last_date": s.index[-1].date(),
        })
        navs.append(pd.DataFrame({"scheme_code": f["scheme_code"], "date": s.index, "nav": s.values}))

    pd.DataFrame(rows).to_csv(out / "schemes.csv", index=False)
    pd.concat(navs).to_parquet(out / "navs.parquet", index=False, compression="zstd")
    print(f"Wrote {len(rows)} synthetic funds to {out}")


if __name__ == "__main__":
    main()
