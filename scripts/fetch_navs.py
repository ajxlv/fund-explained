"""Fetch NAV history for Indian equity mutual funds (Direct, Growth) from mfapi.in.

Run on your laptop (stdlib + pandas + pyarrow):
    pip install pandas pyarrow
    python fetch_navs.py --out data

Writes:
    data/schemes.csv   one row per selected fund: code, name, fund house, category, first/last date
    data/navs.parquet  long table: scheme_code, date, nav
Resumable: already-fetched schemes are cached in data/raw/<code>.json.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.mfapi.in/mf"
UA = {"User-Agent": "fund-explained/0.1 (student project)"}


def get(url: str):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - retry any network error
            if attempt == 3:
                raise
            print(f"retry {url}: {e}")
            time.sleep(2 ** attempt)


def looks_direct_growth(name: str) -> bool:
    n = name.lower()
    return "direct" in n and "growth" in n and not any(
        w in n for w in ("idcw", "dividend", "bonus", "payout", "reinvest")
    )


NON_EQUITY_WORDS = ("gilt", "sdl", "bond", "g-sec", "gsec", "debt", "liquid", "gold", "silver",
                    "crisil", "psu", "money market", "overnight", "treasury", "corporate")


def is_equity_like(category: str, name: str) -> bool:
    """Equity schemes, plus equity index funds (index funds are filed under 'Other Scheme')."""
    if category.startswith("Equity Scheme"):
        return True
    if "Index" in category:
        return not any(w in name.lower() for w in NON_EQUITY_WORDS)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-years", type=float, default=5.0)
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    all_schemes = get(API)
    # Pre-filter by name to skip obvious debt/FMP schemes before fetching (saves most of the requests).
    skip = NON_EQUITY_WORDS + ("fixed maturity", "fmp", "interval", "duration", "credit risk",
                               "banking and psu", "floater", "arbitrage", "savings", "income")
    candidates = [s for s in all_schemes if looks_direct_growth(s["schemeName"])
                  and not any(w in s["schemeName"].lower() for w in skip)]
    print(f"{len(all_schemes)} schemes, {len(candidates)} look like Direct-Growth")

    rows, navs = [], []
    for i, s in enumerate(candidates, 1):
        code = s["schemeCode"]
        cache = raw / f"{code}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
        else:
            d = get(f"{API}/{code}")
            cache.write_text(json.dumps(d))
            time.sleep(0.3)  # be polite
        meta, data = d.get("meta", {}), d.get("data", [])
        cat = meta.get("scheme_category") or ""
        if not data or not is_equity_like(cat, meta.get("scheme_name", s["schemeName"])):
            continue
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna().sort_values("date")
        span_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        if span_years < args.min_years or df["date"].iloc[-1] < pd.Timestamp.today() - pd.Timedelta(days=30):
            continue
        rows.append({
            "scheme_code": code,
            "scheme_name": meta.get("scheme_name", s["schemeName"]),
            "fund_house": meta.get("fund_house"),
            "scheme_category": cat,
            "scheme_type": meta.get("scheme_type"),
            "first_date": df["date"].iloc[0].date(),
            "last_date": df["date"].iloc[-1].date(),
        })
        df["scheme_code"] = code
        navs.append(df[["scheme_code", "date", "nav"]])
        if i % 100 == 0:
            print(f"{i}/{len(candidates)} checked, {len(rows)} kept")

    pd.DataFrame(rows).to_csv(out / "schemes.csv", index=False)
    pd.concat(navs).to_parquet(out / "navs.parquet", index=False, compression="zstd")
    print(f"Kept {len(rows)} funds. Category counts:")
    print(pd.DataFrame(rows)["scheme_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
