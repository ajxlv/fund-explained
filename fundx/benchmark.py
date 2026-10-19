"""Pick the Nifty 50 proxy: a plain Nifty 50 index fund (Direct, Growth) with the longest history."""
import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

_NIFTY50 = re.compile(r"nifty\s*50(?!\d)", re.IGNORECASE)  # "Nifty 50", not "Nifty 500"
_EXCLUDE = ("next", "equal", "value", "momentum", "quality", "low vol")
_NOT_GROWTH = ("idcw", "dividend", "bonus", "payout", "reinvest")


def is_nifty50_index_fund(name: str, category: str) -> bool:
    n = name.lower()
    return (
        "index" in str(category).lower()
        and bool(_NIFTY50.search(name))
        and not any(w in n for w in _EXCLUDE)
        and "direct" in n and "growth" in n
        and not any(w in n for w in _NOT_GROWTH)
    )


def nifty50_index_funds(schemes: pd.DataFrame) -> pd.DataFrame:
    mask = [is_nifty50_index_fund(n, c)
            for n, c in zip(schemes["scheme_name"], schemes["scheme_category"])]
    return schemes[mask]


def pick_benchmark(schemes: pd.DataFrame, matrix: pd.DataFrame) -> int:
    """scheme_code of the Nifty 50 index fund with the longest NAV history."""
    cands = nifty50_index_funds(schemes)
    if cands.empty:
        raise ValueError("No plain Nifty 50 index fund (Direct, Growth) found in the data.")
    counts = matrix[cands["scheme_code"]].notna().sum()
    ranked = cands.assign(n_navs=cands["scheme_code"].map(counts)).sort_values(
        ["first_date", "n_navs", "scheme_code"], ascending=[True, False, True])
    chosen = ranked.iloc[0]
    log.info("Nifty 50 proxy: %s (%s), history from %s",
             chosen["scheme_name"], chosen["scheme_code"], chosen["first_date"].date())
    return int(chosen["scheme_code"])
