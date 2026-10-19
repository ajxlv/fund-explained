import pandas as pd
import pytest

from fundx.benchmark import is_nifty50_index_fund, pick_benchmark

IDX = "Other Scheme - Index Funds"


@pytest.mark.parametrize("name, category, expected", [
    ("UTI Nifty 50 Index Fund - Direct Plan - Growth Option", IDX, True),
    ("HDFC NIFTY50 Index Fund - Direct Plan - Growth", IDX, True),
    ("UTI Nifty Next 50 Index Fund - Direct Plan - Growth", IDX, False),
    ("Motilal Oswal Nifty 500 Index Fund - Direct Plan - Growth", IDX, False),
    ("DSP Nifty 50 Equal Weight Index Fund - Direct - Growth", IDX, False),
    ("UTI Nifty 50 Value 20 Index Fund - Direct - Growth", IDX, False),
    ("Some Nifty 50 Low Volatility Index Fund - Direct - Growth", IDX, False),
    ("UTI Nifty 50 Index Fund - Regular Plan - Growth", IDX, False),
    ("UTI Nifty 50 Index Fund - Direct Plan - IDCW", IDX, False),
    ("Demo Large Cap Fund A - Direct Plan - Growth", "Equity Scheme - Large Cap Fund", False),
])
def test_is_nifty50_index_fund(name, category, expected):
    assert is_nifty50_index_fund(name, category) is expected


def test_pick_benchmark_prefers_longest_history():
    schemes = pd.DataFrame({
        "scheme_code": [1, 2, 3],
        "scheme_name": ["A Nifty 50 Index Fund - Direct - Growth",
                        "B Nifty 50 Index Fund - Direct - Growth",
                        "C Nifty Next 50 Index Fund - Direct - Growth"],
        "scheme_category": [IDX] * 3,
        "first_date": pd.to_datetime(["2015-01-01", "2013-01-01", "2010-01-01"]),
    })
    matrix = pd.DataFrame({1: [1.0, 1.0], 2: [1.0, 1.0], 3: [1.0, 1.0]})
    assert pick_benchmark(schemes, matrix) == 2


def test_pick_benchmark_raises_without_candidates():
    schemes = pd.DataFrame({"scheme_code": [1], "scheme_name": ["X Flexi Cap - Direct - Growth"],
                            "scheme_category": ["Equity Scheme - Flexi Cap Fund"],
                            "first_date": pd.to_datetime(["2015-01-01"])})
    with pytest.raises(ValueError):
        pick_benchmark(schemes, pd.DataFrame({1: [1.0]}))
