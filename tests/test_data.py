import numpy as np
import pandas as pd

from fundx import data


def test_ffill_fills_short_interior_gaps_only():
    s = [1.0, np.nan, np.nan, 4.0] + [np.nan] * 7 + [12.0]
    df = pd.DataFrame({"a": s})
    out = data.ffill_short_gaps(df, max_gap=5)["a"].tolist()
    assert out[:4] == [1.0, 1.0, 1.0, 4.0]  # 2-day gap filled
    assert all(np.isnan(out[4:11]))  # 7-day gap left entirely missing (not partly filled)
    assert out[11] == 12.0


def test_ffill_never_fills_before_first_or_after_last_value():
    df = pd.DataFrame({"a": [np.nan, np.nan, 5.0, np.nan, 6.0, np.nan, np.nan]})
    out = data.ffill_short_gaps(df, max_gap=5)["a"]
    assert out.iloc[:2].isna().all()
    assert out.iloc[3] == 5.0
    assert out.iloc[5:].isna().all()


def test_ffill_exactly_max_gap_is_filled():
    df = pd.DataFrame({"a": [1.0] + [np.nan] * 5 + [2.0]})
    assert data.ffill_short_gaps(df, max_gap=5)["a"].notna().all()


def test_daily_returns_nan_across_missing_days():
    m = pd.DataFrame({"a": [100.0, 110.0, np.nan, 121.0]})
    r = data.daily_returns(m)["a"]
    assert np.isclose(r.iloc[1], 0.10)
    assert r.iloc[2:].isna().all()  # no return is invented across the gap


def test_longest_gaps_counts_missing_calendar_days():
    dates = pd.bdate_range("2024-01-01", periods=20)
    a = pd.DataFrame({"scheme_code": 1, "date": dates, "nav": 1.0})
    b = pd.DataFrame({"scheme_code": 2, "date": dates.delete(range(5, 17)), "nav": 1.0})
    gaps = data.longest_gaps(pd.concat([a, b]))
    assert gaps[1] == 0
    assert gaps[2] == 12


def test_load_dataset_shapes():
    ds = data.load_dataset()
    assert len(ds.schemes) > 0
    assert set(ds.matrix.columns) == set(ds.schemes["scheme_code"])
    assert ds.matrix.index.is_monotonic_increasing
    assert ds.returns.shape == ds.matrix.shape
    if ds.is_demo:
        assert ds.schemes["scheme_name"].str.startswith("Demo ").all()
