"""Smoke tests: the Streamlit app runs without exceptions (headless, via Streamlit's AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fundx.data import load_dataset
from fundx.ml import analyse

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TIMEOUT = 120


@pytest.fixture(scope="module")
def codes():
    ds = load_dataset()
    an = analyse(ds)
    flagged = an.classifier.oof.index[an.classifier.oof["flag"]]
    picks = [an.bench_code, int(ds.schemes["scheme_code"].iloc[0])]
    if len(flagged):
        picks.append(int(flagged[0]))
    return ds, picks


def test_landing_page_renders(codes):
    ds, _ = codes
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Understand any mutual fund in 30 seconds" in text
    assert ("Demo mode: synthetic data" in text) == ds.is_demo
    assert "Not investment advice" in text


def test_fund_pages_render(codes):
    _, picks = codes
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    for code in picks:
        at.selectbox(key="fund").set_value(code).run()
        assert not at.exception, at.exception
        text = " ".join(m.value for m in at.markdown).lower()
        assert "report card" in text
        for word in ("buy", "sell", "best fund", "you should"):
            assert word not in text


def test_filters_and_search_do_not_crash(codes):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.text_input(key="q").set_value("large cap").run()
    assert not at.exception
    opts = at.multiselect(key="cats").options
    at.multiselect(key="cats").set_value([opts[0]]).run()
    assert not at.exception
