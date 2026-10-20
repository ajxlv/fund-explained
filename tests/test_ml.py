import numpy as np
import pandas as pd
import pytest

from fundx import ml
from fundx.features import FEATURES


def test_short_category():
    assert ml.short_category("Equity Scheme - Large Cap Fund") == "Large Cap"
    assert ml.short_category("Other Scheme - Index Funds") == "Index Funds"
    assert ml.short_category("Equity Scheme - Sectoral/ Thematic") == "Sectoral/ Thematic"


def test_merge_small_categories():
    cats = pd.Series(["A"] * 10 + ["B"] * 8 + ["C"] * 7 + ["D"] * 2)
    merged, small = ml.merge_small_categories(cats, min_size=8)
    assert small == ["C", "D"]
    assert (merged == ml.OTHER).sum() == 9
    assert set(merged) == {"A", "B", ml.OTHER}


def blobs(sizes, sep=4.0, seed=0):
    """Well-separated groups in the 8-feature space, one per category."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, (cat, n) in enumerate(sizes.items()):
        X.append(rng.normal(i * sep, 1.0, (n, len(FEATURES))))
        y += [cat] * n
    idx = pd.RangeIndex(sum(sizes.values()))
    return pd.DataFrame(np.vstack(X), index=idx, columns=list(FEATURES)), pd.Series(y, index=idx)


def test_clustering_recovers_separated_groups():
    X, y = blobs({"A": 15, "B": 15, "C": 15})
    res = ml.run_clustering(X, y)
    assert res.k_categories == 3
    assert res.ari_k_categories > 0.9
    assert set(res.silhouettes) == set(range(3, 13))
    assert res.pca.shape == (45, 2)


def test_classifier_beats_baseline_and_flags_carefully():
    X, y = blobs({"A": 20, "B": 15, "C": 12, "D": 3})
    res = ml.run_classifier(X, y)
    assert res.merged == ["D"]
    assert len(res.excluded) == 3  # "Other equity" (just D) is still < 8 funds: left out
    for s in res.scores:
        assert s.accuracy > res.baseline.accuracy
        assert s.confusion.to_numpy().sum() == 47
    assert res.baseline.accuracy == pytest.approx(20 / 47)
    oof = res.oof
    assert len(oof) == 47
    flagged = oof[oof["flag"]]
    assert (flagged["flag_prob"] >= ml.FLAG_PROB).all()
    assert (flagged["flag_category"] != flagged["target"]).all()
    assert (oof.loc[~oof["flag"], "flag_category"].isna()).all()


def test_classifier_flags_a_planted_misfit():
    X, y = blobs({"A": 20, "B": 20, "C": 20}, sep=6.0)
    y.iloc[0] = "B"  # a point sitting in the middle of A but labelled B
    res = ml.run_classifier(X, y)
    assert res.oof.at[0, "flag"]
    assert res.oof.at[0, "flag_category"] == "A"


def test_classifier_is_deterministic():
    X, y = blobs({"A": 12, "B": 12, "C": 12}, sep=1.5)
    a, b = ml.run_classifier(X, y), ml.run_classifier(X, y)
    assert [s.accuracy for s in a.scores] == [s.accuracy for s in b.scores]
