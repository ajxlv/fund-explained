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


def _long(codes_dates):
    return pd.concat([pd.DataFrame({"scheme_code": c, "date": d, "nav": np.arange(1.0, len(d) + 1)})
                      for c, d in codes_dates.items()])


def test_trading_days_drop_stray_dates():
    dates = pd.bdate_range("2024-01-01", periods=10)
    stray = pd.DatetimeIndex([pd.Timestamp("2024-01-06")])  # a Saturday, only fund 3 has it
    navs = _long({1: dates, 2: dates, 3: dates.append(stray).sort_values()})
    cal = data.trading_days(navs)
    assert pd.Timestamp("2024-01-06") not in cal
    assert len(cal) == 10


def test_trading_days_respect_fund_lifetimes():
    # Early dates where only the one live fund publishes are still trading days.
    early = pd.bdate_range("2020-01-01", periods=5)
    late = pd.bdate_range("2020-01-08", periods=5)
    navs = _long({1: early.append(late), 2: late, 3: late})
    assert len(data.trading_days(navs)) == 10


def test_returns_never_span_filled_days():
    dates = pd.bdate_range("2024-01-01", periods=10)
    navs = _long({1: dates, 2: dates, 3: dates.delete(4)})
    raw = data.raw_matrix(navs)
    filled = data.ffill_short_gaps(raw)
    assert filled[3].notna().all()  # aligned NAV matrix is filled...
    r = data.daily_returns(raw)[3]
    assert r.iloc[4:6].isna().all()  # ...but no 0% / two-day-jump returns are made up


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
