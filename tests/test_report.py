import pandas as pd
import pytest

from fundx import ml, report
from fundx.data import load_dataset
from fundx.metrics import Drawdown

BANNED = ("buy", "sell", "best fund", "you should")


def test_category_noun_articles():
    assert report.category_noun("Equity Scheme - Large Cap Fund") == "a Large Cap fund"
    assert report.category_noun("Other Scheme - Index Funds") == "an Index fund"
    assert report.category_noun("Equity Scheme - ELSS") == "an ELSS fund"


def test_beta_sentence_matches_plan_wording():
    s = report.beta_sentence(0.95)
    assert s.startswith("When the Nifty rises 1%, this fund has typically risen about 0.95%")
    assert "more than the market" in report.beta_sentence(1.2)
    assert "less than the market" in report.beta_sentence(0.7)
    assert "Not enough" in report.beta_sentence(None)


def test_correlation_sentence_bands():
    assert "almost always" in report.correlation_sentence(0.97)
    assert "loosely" in report.correlation_sentence(0.3)


def test_recovery_text():
    d = pd.Timestamp("2021-01-01")
    assert report.recovery_text(Drawdown(-0.2, d, d, d, 92)) == "Recovered in about 3 months"
    assert report.recovery_text(Drawdown(-0.2, d, d, d, 10)) == "Recovered in about 1 month"
    assert report.recovery_text(Drawdown(-0.2, d, d, None, None)) == "Not yet recovered"
    assert report.recovery_text(None) == "Not enough history."


def test_drawdown_story():
    dd = Drawdown(-0.25, pd.Timestamp("2021-02-01"), pd.Timestamp("2021-03-01"),
                  pd.Timestamp("2021-06-01"), 92)
    s = report.drawdown_story(dd)
    assert "fell 25.0%" in s and "1 Feb 2021" in s and "about 3 months after the low" in s


@pytest.fixture(scope="module")
def setup():
    ds = load_dataset()
    return ds, ml.analyse(ds)


def test_build_report_for_every_fund_and_wording_is_neutral(setup):
    ds, an = setup
    for code in ds.schemes["scheme_code"]:
        r = report.build_report(ds, an, code)
        assert set(r.growth) == {1, 3, 5}
        assert len(r.lookalikes) <= 5
        assert code not in set(r.lookalikes["scheme_code"])
        assert r.risk_label in ("Low", "Moderate", "High", "Very High")
        text = " ".join([r.label_text, report.beta_sentence(r.beta),
                         report.correlation_sentence(r.correlation),
                         report.drawdown_story(r.drawdown)]).lower()
        for w in BANNED:
            assert w not in text


def test_flagged_wording(setup):
    ds, an = setup
    flagged = an.classifier.oof.index[an.classifier.oof["flag"]]
    if len(flagged) == 0:
        pytest.skip("No funds flagged in this dataset.")
    r = report.build_report(ds, an, flagged[0])
    assert r.label_flagged
    assert r.label_text.startswith("Based on its past price movements, this fund has behaved "
                                   "more like a")
    assert "This doesn't mean it's mislabelled; check its portfolio." in r.label_text


def test_example_funds_prefers_well_known_then_falls_back():
    schemes = pd.DataFrame({
        "scheme_code": [1, 2, 3, 4],
        "scheme_name": ["Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
                        "Zeta Large Cap Fund - Direct Plan - Growth",
                        "Alpha Large Cap Fund - Direct Plan - Growth",
                        "UTI Nifty 50 Index Fund - Direct Plan - Growth"],
        "scheme_category": ["Equity Scheme - Flexi Cap Fund", "Equity Scheme - Large Cap Fund",
                            "Equity Scheme - Large Cap Fund", "Other Scheme - Index Funds"],
        "first_date": pd.to_datetime(["2013-01-01", "2010-01-01", "2014-01-01", "2013-01-01"]),
    })
    assert report.example_funds(schemes, bench_code=4) == [1, 4, 2]
    assert len(report.example_funds(schemes.iloc[1:3], bench_code=4)) == 3


def test_short_name():
    assert report.short_name("HDFC Flexi Cap Fund - Growth Option - Direct Plan") == \
        "HDFC Flexi Cap Fund"
    assert report.short_name("No Suffix Fund") == "No Suffix Fund"
