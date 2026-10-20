"""Lookalike funds: Pearson correlation of daily returns over the last 3 years."""
import pandas as pd

from fundx.benchmark import nifty50_index_funds

WINDOW_YEARS = 3
MIN_OVERLAP = 600  # trading days both funds must share
TOP_K = 5
SANITY_CORR = 0.99


def correlation_matrix(returns: pd.DataFrame, years: float = WINDOW_YEARS,
                       min_overlap: int = MIN_OVERLAP) -> pd.DataFrame:
    """Pairwise return correlations; NaN where two funds share fewer than `min_overlap` days."""
    cutoff = returns.index.max() - pd.DateOffset(years=years)
    return returns[returns.index > cutoff].corr(min_periods=min_overlap)


def top_lookalikes(corr: pd.DataFrame, code: int, k: int = TOP_K) -> pd.Series:
    """The k funds whose returns correlate most with `code` (itself excluded), highest first."""
    if code not in corr:
        return pd.Series(dtype=float)
    return corr[code].drop(code).dropna().sort_values(ascending=False).head(k)


def nifty50_sanity(corr: pd.DataFrame, schemes: pd.DataFrame,
                   threshold: float = SANITY_CORR) -> pd.DataFrame:
    """For each Nifty 50 index fund: is its top lookalike another one, with corr >= threshold?"""
    idx_codes = set(nifty50_index_funds(schemes)["scheme_code"]) & set(corr.columns)
    rows = []
    for code in sorted(idx_codes):
        top = top_lookalikes(corr, code, k=1)
        top_code = int(top.index[0]) if len(top) else None
        top_corr = float(top.iloc[0]) if len(top) else None
        rows.append({
            "scheme_code": code,
            "top_lookalike": top_code,
            "corr": top_corr,
            "passed": top_code in idx_codes and top_corr is not None and top_corr >= threshold,
        })
    return pd.DataFrame(rows, columns=["scheme_code", "top_lookalike", "corr", "passed"])
