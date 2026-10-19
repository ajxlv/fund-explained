"""End-to-end checks on whichever dataset is present (real data/ or the demo fallback)."""
import pytest

from fundx import ml
from fundx.benchmark import nifty50_index_funds
from fundx.data import load_dataset
from fundx.features import FEATURES
from tests.wording import banned_phrases


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


@pytest.fixture(scope="module")
def analysis(ds):
    return ml.analyse(ds)


def test_features_cover_every_fund(ds, analysis):
    assert list(analysis.features.columns) == list(FEATURES)
    assert set(analysis.features.index) == set(ds.schemes["scheme_code"])
    assert len(analysis.used) >= 0.8 * len(ds.schemes)


def test_benchmark_has_beta_one(analysis):
    f = analysis.features.loc[analysis.bench_code]
    assert f["beta_3y"] == pytest.approx(1.0)
    assert f["corr_nifty_3y"] == pytest.approx(1.0)


def test_report_mentions_data_kind(ds, analysis):
    text = ml.build_report(ds, analysis)
    assert text.startswith("# ⚠️ DEMO DATA, NOT REAL") == ds.is_demo
    assert banned_phrases(text) == []


def test_real_data_nifty50_lookalikes_match_each_other(ds, analysis):
    if ds.is_demo:
        pytest.skip("Only demo data is present. This sanity check needs real mfapi.in data in "
                    "data/ (run scripts/fetch_navs.py; see docs/MORNING_STEPS.md).")
    assert len(nifty50_index_funds(ds.schemes)) >= 2
    assert analysis.sanity["passed"].mean() >= 0.9


def test_demo_index_funds_are_each_others_lookalikes(ds, analysis):
    if not ds.is_demo:
        pytest.skip("Demo-only pipeline check; real data is covered by the test above.")
    assert len(analysis.sanity) == 3
    assert analysis.sanity["passed"].all()
