"""Metric tests on tiny hand-built series where the right answer is obvious."""
import math

import numpy as np
import pandas as pd
import pytest

from fundx import metrics as m


def series(values, dates):
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


# --- growth / CAGR -------------------------------------------------------------------------------

def test_cagr_doubling_over_exactly_two_years():
    nav = series([10, 12, 20], ["2020-01-01", "2021-01-01", "2022-01-01"])
    assert m.cagr(nav, 2) == pytest.approx(math.sqrt(2) - 1)


def test_calendar_years_is_exact_across_leap_years():
    assert m.calendar_years(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-01-01")) == 2.0
    # Anniversaries of 29 Feb fall on 28 Feb in non-leap years (pandas DateOffset convention).
    assert m.calendar_years(pd.Timestamp("2020-02-29"), pd.Timestamp("2021-02-28")) == 1.0
    assert m.calendar_years(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-07-02")) == \
        pytest.approx(182 / 365)


def test_growth_of_10k_uses_last_nav_on_or_before_target():
    # The target is 2021-01-04 (a Monday with no NAV), so the start is Friday 2021-01-01's NAV.
    nav = series([50, 100, 999, 150],
                 ["2020-12-31", "2021-01-01", "2021-01-05", "2022-01-04"])
    assert m.growth_of_10k(nav, 1) == pytest.approx(15_000)


def test_growth_and_cagr_too_short_history_return_none():
    nav = series([10, 11, 12], ["2024-01-01", "2024-06-01", "2024-12-01"])
    assert m.growth_of_10k(nav, 1) is None
    assert m.cagr(nav, 3) is None
    assert m.max_drawdown(nav, years=5) is None
    assert m.growth_of_10k(series([10], ["2024-01-01"]), 1) is None
    assert m.cagr(pd.Series(dtype=float), 1) is None


def test_growth_ignores_missing_navs():
    nav = series([10, np.nan, 20], ["2020-01-01", "2020-06-01", "2021-01-01"])
    assert m.growth_of_10k(nav, 1) == pytest.approx(20_000)


# --- drawdown ------------------------------------------------------------------------------------

def test_max_drawdown_known_path():
    dates = ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]
    dd = m.max_drawdown(series([100, 120, 90, 125], dates), years=None)
    assert dd.drawdown == pytest.approx(-0.25)
    assert dd.peak_date == pd.Timestamp("2021-02-01")
    assert dd.trough_date == pd.Timestamp("2021-03-01")
    assert dd.recovery_date == pd.Timestamp("2021-04-01")
    assert dd.recovery_days == 31


def test_max_drawdown_not_yet_recovered():
    dates = ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]
    dd = m.max_drawdown(series([100, 120, 90, 110], dates), years=None)
    assert dd.drawdown == pytest.approx(-0.25)
    assert dd.recovery_date is None and dd.recovery_days is None


def test_max_drawdown_recovery_at_exactly_the_peak():
    dates = ["2021-01-01", "2021-02-01", "2021-03-01"]
    dd = m.max_drawdown(series([120, 60, 120], dates), years=None)
    assert dd.drawdown == pytest.approx(-0.5)
    assert dd.recovery_date == pd.Timestamp("2021-03-01")


def test_max_drawdown_only_rises():
    dd = m.max_drawdown(series([1, 2, 3], ["2021-01-01", "2021-01-02", "2021-01-03"]), years=None)
    assert dd.drawdown == 0.0 and dd.peak_date is None


def test_max_drawdown_window_excludes_older_falls():
    # A crash 6 years ago must not appear in the 5-year window.
    nav = series([100, 50, 60, 70, 65, 80],
                 ["2015-01-01", "2015-06-01", "2019-01-01", "2020-01-01", "2021-01-01",
                  "2024-01-01"])
    dd = m.max_drawdown(nav, years=5)
    assert dd.drawdown == pytest.approx(65 / 70 - 1)


# --- volatility, beta, correlation ---------------------------------------------------------------

@pytest.fixture
def bench():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=300)
    return pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)


def test_volatility_annualises_with_252():
    r = pd.Series([0.01, -0.01] * 50)
    assert m.volatility(r) == pytest.approx(r.std() * math.sqrt(252))
    assert m.volatility(pd.Series([0.01])) is None


def test_identical_fund_has_beta_1_corr_1_te_0(bench):
    assert m.beta_vs_benchmark(bench, bench) == pytest.approx(1.0)
    assert m.correlation_vs_benchmark(bench, bench) == pytest.approx(1.0)
    assert m.tracking_error(bench, bench) == pytest.approx(0.0)


def test_double_returns_have_beta_2(bench):
    assert m.beta_vs_benchmark(2 * bench, bench) == pytest.approx(2.0)
    assert m.correlation_vs_benchmark(2 * bench, bench) == pytest.approx(1.0)


def test_beta_uses_only_common_dates(bench):
    fund = 2 * bench
    fund.iloc[::3] = np.nan  # missing days in the fund
    noisy_bench = bench.copy()
    noisy_bench.iloc[::3] = 99.0  # junk on dates the fund doesn't have; must be ignored
    assert m.beta_vs_benchmark(fund, noisy_bench) == pytest.approx(2.0)


def test_beta_too_few_points_is_none(bench):
    assert m.beta_vs_benchmark(bench.iloc[:5], bench.iloc[:5]) is None
    assert m.correlation_vs_benchmark(bench.iloc[:5], bench.iloc[:5]) is None


def test_downside_volatility_ignores_rises():
    up = pd.Series([0.01, 0.02, 0.03])
    assert m.downside_volatility(up) == 0.0
    mixed = pd.Series([0.05, -0.02, 0.05, -0.02])
    assert m.downside_volatility(mixed) == pytest.approx(math.sqrt(0.0002) * math.sqrt(252))


# --- risk label ----------------------------------------------------------------------------------

def test_risk_label_quartiles():
    vols = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
    assert m.risk_label(0.10, vols) == "Low"
    assert m.risk_label(0.15, vols) == "Moderate"
    assert m.risk_label(0.19, vols) == "High"
    assert m.risk_label(0.24, vols) == "Very High"
    assert m.risk_label(None, vols) is None


def test_risk_percentile():
    vols = [0.1, 0.2, 0.3, 0.4]
    assert m.risk_percentile(0.3, vols) == 50.0
    assert m.risk_percentile(0.1, vols) == 0.0
