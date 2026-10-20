"""Report-card and risk metrics on a single fund's NAV / daily-return series.

Conventions:
- NAVs are trading-day NAVs as given (index = dates).
- CAGR uses exact calendar years between the two dates actually used.
- Volatility-type numbers are annualised with 252 trading days per year.
- Too little history returns None rather than raising.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(s: pd.Series) -> pd.Series:
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def calendar_years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Exact calendar years: whole years, plus the fraction of the following year."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    whole = end.year - start.year
    if start + pd.DateOffset(years=whole) > end:
        whole -= 1
    anniversary = start + pd.DateOffset(years=whole)
    next_anniversary = start + pd.DateOffset(years=whole + 1)
    return whole + (end - anniversary).days / (next_anniversary - anniversary).days


def window_start(nav: pd.Series, years: float) -> pd.Timestamp | None:
    """Date of the last NAV on or before (last date - years), or None if history is too short."""
    s = _clean(nav)
    if len(s) < 2:
        return None
    target = s.index[-1] - pd.DateOffset(years=years)
    before = s.index[s.index <= target]
    return before[-1] if len(before) else None


def growth_of_10k(nav: pd.Series, years: float) -> float | None:
    """What Rs 10,000 invested `years` ago would be worth at the latest NAV."""
    s = _clean(nav)
    start = window_start(s, years)
    if start is None:
        return None
    return 10_000 * s.iloc[-1] / s[start]


def cagr(nav: pd.Series, years: float) -> float | None:
    """Compound annual growth rate over the last `years` (exact calendar years)."""
    s = _clean(nav)
    start = window_start(s, years)
    if start is None:
        return None
    span = calendar_years(start, s.index[-1])
    return (s.iloc[-1] / s[start]) ** (1 / span) - 1


@dataclass
class Drawdown:
    drawdown: float  # e.g. -0.25 for a 25% fall from the peak
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None  # first date NAV >= peak again; None if not yet
    recovery_days: int | None  # calendar days from trough to recovery


def max_drawdown(nav: pd.Series, years: float | None = 5) -> Drawdown | None:
    """Worst peak-to-trough fall within the last `years` (whole history if None)."""
    s = _clean(nav)
    if years is not None:
        start = window_start(s, years)
        if start is None:
            return None
        s = s[s.index >= start]
    if len(s) < 2:
        return None
    dd = s / s.cummax() - 1
    if dd.min() >= 0:
        return Drawdown(0.0, None, None, None, None)
    trough = dd.idxmin()
    peak = s[:trough].idxmax()
    later = s[s.index > trough]
    recovered = later[later >= s[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return Drawdown(
        drawdown=float(dd.min()),
        peak_date=peak,
        trough_date=trough,
        recovery_date=recovery,
        recovery_days=(recovery - trough).days if recovery is not None else None,
    )


def last_years(x: pd.Series | pd.DataFrame, years: float | None):
    """Rows after (last date - years). Works on returns, so N years of returns ~ N years of NAVs."""
    x = x.dropna()
    if years is None or x.empty:
        return x
    cutoff = x.index[-1] - pd.DateOffset(years=years)
    return x[x.index > cutoff]


def volatility(returns: pd.Series) -> float | None:
    """Annualised volatility: std of daily returns x sqrt(252)."""
    r = returns.dropna()
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def downside_volatility(returns: pd.Series) -> float | None:
    """Annualised downside deviation: only falls count (rises are treated as 0)."""
    r = returns.dropna()
    if len(r) < 2:
        return None
    return float(np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS))


def _paired(fund: pd.Series, bench: pd.Series, years: float | None, min_obs: int):
    df = pd.concat([fund.rename("f"), bench.rename("b")], axis=1, join="inner").dropna()
    df = last_years(df, years)
    return df if len(df) >= min_obs else None


def beta_vs_benchmark(fund_returns: pd.Series, bench_returns: pd.Series,
                      years: float | None = 3, min_obs: int = 20) -> float | None:
    """cov(fund, bench) / var(bench) on daily returns over the same dates (last `years`)."""
    df = _paired(fund_returns, bench_returns, years, min_obs)
    if df is None or df["b"].var() == 0:
        return None
    return float(df["f"].cov(df["b"]) / df["b"].var())


def correlation_vs_benchmark(fund_returns: pd.Series, bench_returns: pd.Series,
                             years: float | None = 3, min_obs: int = 20) -> float | None:
    """Pearson correlation of daily returns over the same dates (last `years`)."""
    df = _paired(fund_returns, bench_returns, years, min_obs)
    if df is None or df["f"].std() == 0 or df["b"].std() == 0:
        return None
    return float(df["f"].corr(df["b"]))


def tracking_error(fund_returns: pd.Series, bench_returns: pd.Series,
                   years: float | None = 3, min_obs: int = 20) -> float | None:
    """Annualised std of the daily return difference (fund - bench)."""
    df = _paired(fund_returns, bench_returns, years, min_obs)
    if df is None:
        return None
    return float((df["f"] - df["b"]).std(ddof=1) * np.sqrt(TRADING_DAYS))


RISK_LABELS = ("Low", "Moderate", "High", "Very High")


def risk_label(vol: float | None, all_vols) -> str | None:
    """Relative risk label from volatility quartiles across all funds in the dataset."""
    vols = pd.Series(all_vols, dtype=float).dropna()
    if vol is None or np.isnan(vol) or vols.empty:
        return None
    q1, q2, q3 = np.quantile(vols, [0.25, 0.5, 0.75])
    if vol <= q1:
        return RISK_LABELS[0]
    if vol <= q2:
        return RISK_LABELS[1]
    if vol <= q3:
        return RISK_LABELS[2]
    return RISK_LABELS[3]


def risk_percentile(vol: float | None, all_vols) -> float | None:
    """Share of funds in the dataset (in %) with lower volatility than `vol`."""
    vols = pd.Series(all_vols, dtype=float).dropna()
    if vol is None or np.isnan(vol) or vols.empty:
        return None
    return float((vols < vol).mean() * 100)
