"""Tests for the pure helpers in scripts/fetch_navs.py (no network)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_navs", Path(__file__).resolve().parents[1] / "scripts" / "fetch_navs.py"
)
fetch_navs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_navs)


@pytest.mark.parametrize("name, expected", [
    ("Parag Parikh Flexi Cap Fund - Direct Plan - Growth", True),
    ("HDFC Flexi Cap Fund - Growth Option - Direct Plan", True),
    ("UTI Nifty 50 Index Fund - Direct Plan - Growth Option", True),
    ("Parag Parikh Flexi Cap Fund - Regular Plan - Growth", False),
    ("SBI Large Cap Fund - Direct Plan - IDCW", False),
    ("Axis Midcap Fund - Direct Plan - Dividend", False),
    ("ICICI Prudential Value Fund - Direct Plan - Growth - Bonus", False),
    ("Some Fund - Direct Plan - Growth - Payout", False),
    ("Some Fund - Direct Plan - Growth - Reinvestment", False),
    ("Some Fund - Direct Plan", False),
])
def test_looks_direct_growth(name, expected):
    assert fetch_navs.looks_direct_growth(name) is expected


@pytest.mark.parametrize("category, name, expected", [
    ("Equity Scheme - Large Cap Fund", "Any Large Cap Fund - Direct - Growth", True),
    ("Equity Scheme - Sectoral/ Thematic", "Any Banking Fund", True),
    ("Other Scheme - Index Funds", "UTI Nifty 50 Index Fund", True),
    ("Other Scheme - Index Funds", "Nifty Next 50 Index Fund", True),
    ("Other Scheme - Index Funds", "Bharat Bond Index Fund", False),
    ("Other Scheme - Index Funds", "CRISIL IBX Gilt Index - April 2026", False),
    ("Other Scheme - Index Funds", "Nifty SDL Plus G-Sec Index Fund", False),
    ("Other Scheme - Index Funds", "Gold Index Fund", False),
    ("Debt Scheme - Liquid Fund", "Any Liquid Fund", False),
    ("Hybrid Scheme - Aggressive Hybrid Fund", "Any Hybrid Fund", False),
    ("", "No category", False),
])
def test_is_equity_like(category, name, expected):
    assert fetch_navs.is_equity_like(category, name) is expected


def test_equity_scheme_ignores_name_words():
    # Equity schemes pass on category alone, even if the name contains a "debt-ish" word.
    assert fetch_navs.is_equity_like("Equity Scheme - Sectoral/ Thematic", "Any PSU Equity Fund")
