"""Load schemes and NAVs, build the aligned daily NAV matrix (dates x funds) and daily returns.

Real data lives in data/ (from scripts/fetch_navs.py). If it is absent we fall back to the
synthetic data in data/demo/ and say so loudly everywhere (`Dataset.is_demo`).

    python -m fundx.data --summary    # writes docs/DATA_SUMMARY.md
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = REPO_ROOT / "data"
DEMO_DIR = REPO_ROOT / "data" / "demo"
MAX_FILL_GAP = 5  # trading days; longer gaps stay missing
MIN_SHARE_PUBLISHING = 0.5  # a date is a trading day if >= half of the live funds have a NAV
LONG_GAP = 10  # trading days; reported in DATA_SUMMARY.md


def _has_files(d: Path) -> bool:
    return (d / "schemes.csv").exists() and (d / "navs.parquet").exists()


def resolve_data_dir() -> tuple[Path, bool]:
    """Return (directory, is_demo). Real data in data/ wins; data/demo/ is the fallback."""
    if _has_files(REAL_DIR):
        return REAL_DIR, False
    if _has_files(DEMO_DIR):
        return DEMO_DIR, True
    raise FileNotFoundError(
        "No data found. Run `python scripts/fetch_navs.py --out data` (real) or "
        "`python scripts/make_demo_data.py` (synthetic demo)."
    )


def load_schemes(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "schemes.csv", parse_dates=["first_date", "last_date"])
    df["scheme_code"] = df["scheme_code"].astype(int)
    return df.sort_values("scheme_name").reset_index(drop=True)


def load_navs(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "navs.parquet")
    df["scheme_code"] = df["scheme_code"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    return df


def ffill_short_gaps(df: pd.DataFrame, max_gap: int = MAX_FILL_GAP) -> pd.DataFrame:
    """Forward-fill interior runs of missing values that are at most `max_gap` rows long.

    Longer gaps, and anything before a fund's first NAV or after its last one, stay NaN.
    (Plain `ffill(limit=...)` would wrongly fill the first few days of a long gap.)
    """
    def fill(s: pd.Series) -> pd.Series:
        na = s.isna()
        if not na.any():
            return s
        run_id = (~na).cumsum()
        run_len = na.groupby(run_id).transform("sum")
        interior = s.ffill().notna() & s.bfill().notna()
        fillable = na & interior & (run_len <= max_gap)
        return s.where(~fillable, s.ffill())

    return df.apply(fill)


def trading_days(navs: pd.DataFrame, min_share: float = MIN_SHARE_PUBLISHING) -> pd.DatetimeIndex:
    """Dates on which at least `min_share` of the funds alive that day published a NAV.

    This drops stray dates (e.g. one fund reporting a NAV on a holiday), which would otherwise
    leave every other fund with a hole in its returns.
    """
    per_fund = navs.groupby("scheme_code")["date"].agg(["min", "max"])
    dates = pd.DatetimeIndex(sorted(navs["date"].unique()))
    started = np.searchsorted(np.sort(per_fund["min"].to_numpy()), dates.to_numpy(), side="right")
    ended = np.searchsorted(np.sort(per_fund["max"].to_numpy()), dates.to_numpy(), side="left")
    alive = started - ended
    published = navs.groupby("date")["scheme_code"].nunique().reindex(dates).to_numpy()
    return dates[published >= min_share * alive]


def raw_matrix(navs: pd.DataFrame) -> pd.DataFrame:
    """Dates x scheme_code NAV matrix on the trading-day calendar, gaps left as NaN."""
    wide = navs.pivot_table(index="date", columns="scheme_code", values="nav", aggfunc="last")
    return wide.reindex(trading_days(navs))


def nav_matrix(navs: pd.DataFrame, max_gap: int = MAX_FILL_GAP) -> pd.DataFrame:
    """Aligned NAV matrix with interior gaps of up to `max_gap` trading days forward-filled."""
    return ffill_short_gaps(raw_matrix(navs), max_gap)


def daily_returns(matrix: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. A return is NaN whenever either day's NAV is missing.

    Pass the *unfilled* matrix: a forward-filled day would show a fake 0% return followed by a
    two-day jump, which distorts volatility, beta and correlations.
    """
    return matrix.pct_change(fill_method=None)


@dataclass
class Dataset:
    schemes: pd.DataFrame
    navs: pd.DataFrame
    matrix: pd.DataFrame  # aligned NAVs, short gaps forward-filled (for NAV-level metrics, charts)
    returns: pd.DataFrame  # daily returns from the unfilled NAVs (for risk, beta, correlations)
    data_dir: Path
    is_demo: bool

    @property
    def as_of(self) -> pd.Timestamp:
        return self.matrix.index.max()


def load_dataset() -> Dataset:
    data_dir, is_demo = resolve_data_dir()
    schemes = load_schemes(data_dir)
    navs = load_navs(data_dir)
    navs = navs[navs["scheme_code"].isin(schemes["scheme_code"])]
    raw = raw_matrix(navs)
    matrix = ffill_short_gaps(raw)
    return Dataset(schemes, navs, matrix, daily_returns(raw), data_dir, is_demo)


def longest_gaps(navs: pd.DataFrame) -> pd.Series:
    """Longest run of missing trading days per fund, between its own first and last NAV.

    Measured on the `trading_days` calendar (stray non-trading dates are ignored).
    """
    calendar = trading_days(navs)
    pos = pd.Series(range(len(calendar)), index=calendar)
    out = {}
    for code, g in navs.groupby("scheme_code"):
        p = pos.reindex(g["date"].sort_values()).dropna().to_numpy().astype(int)
        out[code] = int((p[1:] - p[:-1] - 1).max()) if len(p) > 1 else 0
    return pd.Series(out, name="longest_gap_days")


def write_summary(path: Path = REPO_ROOT / "docs" / "DATA_SUMMARY.md") -> str:
    ds = load_dataset()
    s = ds.schemes
    gaps = longest_gaps(ds.navs)
    long_gaps = gaps[gaps > LONG_GAP].sort_values(ascending=False)
    kind = ("**DEMO DATA, NOT REAL.** Synthetic funds from `scripts/make_demo_data.py`."
            if ds.is_demo else "**Real data** from mfapi.in (AMFI NAVs), via `scripts/fetch_navs.py`.")
    lines = [
        "# Data summary",
        "",
        "_Generated by `python -m fundx.data --summary`. Do not edit by hand._",
        "",
        kind,
        "",
        f"- Source directory: `{ds.data_dir.relative_to(REPO_ROOT)}/`",
        f"- Funds: **{len(s)}**",
        f"- NAV rows: {len(ds.navs):,}",
        f"- Date range: {ds.navs['date'].min().date()} to {ds.navs['date'].max().date()}",
        f"- Trading days in the calendar: {len(ds.matrix)}",
        "",
        "## Funds per category",
        "",
        "| Category | Funds |",
        "|---|---:|",
    ]
    for cat, n in s["scheme_category"].value_counts().items():
        lines.append(f"| {cat} | {n} |")
    lines += [
        "",
        f"## Funds with gaps of more than {LONG_GAP} trading days",
        "",
        (f"Gaps of up to {MAX_FILL_GAP} trading days are forward-filled when aligning dates; "
         "longer gaps stay missing."),
        "",
    ]
    if long_gaps.empty:
        lines.append("None.")
    else:
        names = s.set_index("scheme_code")["scheme_name"]
        lines += ["| Code | Fund | Longest gap (trading days) |", "|---|---|---:|"]
        for code, g in long_gaps.items():
            lines.append(f"| {code} | {names[code]} | {g} |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="write docs/DATA_SUMMARY.md")
    args = ap.parse_args()
    if args.summary:
        write_summary()
        print("Wrote docs/DATA_SUMMARY.md")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
