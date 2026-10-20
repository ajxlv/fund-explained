import numpy as np
import pandas as pd

from fundx.lookalikes import correlation_matrix, nifty50_sanity, top_lookalikes

IDX = "Other Scheme - Index Funds"


def make_returns(n=800, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    mkt = rng.normal(0, 0.01, n)
    return pd.DataFrame({
        1: mkt + rng.normal(0, 0.0002, n),   # index fund
        2: mkt + rng.normal(0, 0.0002, n),   # another index fund
        3: 0.9 * mkt + rng.normal(0, 0.004, n),
        4: rng.normal(0, 0.01, n),           # unrelated
    }, index=idx)


def test_top_lookalikes_excludes_self_and_sorts():
    corr = correlation_matrix(make_returns(), years=10, min_overlap=600)
    top = top_lookalikes(corr, 1, k=3)
    assert 1 not in top.index
    assert list(top.index) == [2, 3, 4]
    assert top.iloc[0] > 0.99
    assert top.is_monotonic_decreasing


def test_min_overlap_is_enforced():
    r = make_returns()
    r.iloc[:300, 2] = np.nan  # fund 3 now shares only 500 days with the others
    corr = correlation_matrix(r, years=10, min_overlap=600)
    assert corr[1].isna()[3]
    assert 3 not in top_lookalikes(corr, 1).index


def test_window_uses_last_three_years_only():
    r = make_returns(n=1500)
    r.iloc[:-756, 3] = r.iloc[:-756, 0]  # fund 4 copied fund 1 long ago, unrelated since
    corr = correlation_matrix(r, years=3, min_overlap=600)
    assert abs(corr.at[1, 4]) < 0.2


def test_unknown_code_gives_empty_result():
    corr = correlation_matrix(make_returns(), years=10, min_overlap=600)
    assert top_lookalikes(corr, 999).empty


def test_nifty50_sanity_on_synthetic_index_funds():
    corr = correlation_matrix(make_returns(), years=10, min_overlap=600)
    schemes = pd.DataFrame({
        "scheme_code": [1, 2, 3, 4],
        "scheme_name": ["A Nifty 50 Index Fund - Direct - Growth",
                        "B Nifty 50 Index Fund - Direct - Growth",
                        "C Large Cap - Direct - Growth", "D Gold-ish - Direct - Growth"],
        "scheme_category": [IDX, IDX, "Equity Scheme - Large Cap Fund", "Other"],
    })
    res = nifty50_sanity(corr, schemes)
    assert len(res) == 2 and res["passed"].all()
