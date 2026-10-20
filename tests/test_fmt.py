import math

import pytest

from fundx.fmt import fmt_inr, fmt_pct


@pytest.mark.parametrize("x, expected", [
    (0, "₹0"),
    (999, "₹999"),
    (1000, "₹1,000"),
    (10000, "₹10,000"),
    (123456, "₹1,23,456"),
    (1234567, "₹12,34,567"),
    (12345678, "₹1,23,45,678"),
    (123456789.4, "₹12,34,56,789"),
    (-123456, "-₹1,23,456"),
    (15234.6, "₹15,235"),
])
def test_fmt_inr_indian_grouping(x, expected):
    assert fmt_inr(x) == expected


def test_fmt_inr_decimals_and_symbol():
    assert fmt_inr(1234567.891, decimals=2) == "₹12,34,567.89"
    assert fmt_inr(123456, symbol=False) == "1,23,456"


def test_fmt_inr_missing():
    assert fmt_inr(None) == "—"
    assert fmt_inr(math.nan) == "—"


def test_fmt_pct():
    assert fmt_pct(0.1234) == "12.3%"
    assert fmt_pct(-0.25, 0) == "-25%"
    assert fmt_pct(0.05, signed=True) == "+5.0%"
    assert fmt_pct(None) == "—"
