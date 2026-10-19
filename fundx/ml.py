"""Clustering, the category "label check" classifier, and the evaluation report.

    python -m fundx.ml --report    # writes docs/EVALUATION.md
"""
import argparse
import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fundx.benchmark import pick_benchmark
from fundx.data import REPO_ROOT, Dataset, load_dataset
from fundx.features import FEATURES, build_features
from fundx.lookalikes import (
    MIN_OVERLAP,
    SANITY_CORR,
    WINDOW_YEARS,
    correlation_matrix,
    nifty50_sanity,
)

log = logging.getLogger(__name__)

RANDOM_STATE = 42
K_RANGE = range(3, 13)
MIN_CLASS_SIZE = 8
OTHER = "Other equity"
FLAG_PROB = 0.7
N_FOLDS = 5


def short_category(cat: str) -> str:
    """'Equity Scheme - Large Cap Fund' -> 'Large Cap'."""
    s = str(cat)
    for prefix in ("Equity Scheme - ", "Other Scheme - "):
        s = s.removeprefix(prefix)
    return s.removesuffix(" Fund").strip()


def merge_small_categories(cats: pd.Series, min_size: int = MIN_CLASS_SIZE):
    """Categories with fewer than `min_size` funds become OTHER. Returns (merged, merged_names)."""
    counts = cats.value_counts()
    small = sorted(counts[counts < min_size].index)
    return cats.where(~cats.isin(small), OTHER), small


# --- clustering ----------------------------------------------------------------------------------

@dataclass
class ClusterResult:
    silhouettes: dict[int, float]
    best_k: int
    labels: pd.Series  # cluster per fund at best_k
    ari_best: float
    k_categories: int
    labels_k_categories: pd.Series
    ari_k_categories: float
    silhouette_k_categories: float
    pca: pd.DataFrame  # pc1, pc2 per fund
    pca_explained: tuple[float, float]


def run_clustering(X: pd.DataFrame, categories: pd.Series) -> ClusterResult:
    Z = StandardScaler().fit_transform(X)

    def fit(k):
        return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)

    ks = [k for k in K_RANGE if k < len(X)]
    labels_by_k = {k: fit(k) for k in ks}
    sil = {k: float(silhouette_score(Z, labels_by_k[k])) for k in ks}
    best_k = max(sil, key=sil.get)
    k_cat = int(categories.nunique())
    lab_cat = labels_by_k[k_cat] if k_cat in labels_by_k else fit(k_cat)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)
    return ClusterResult(
        silhouettes=sil,
        best_k=best_k,
        labels=pd.Series(labels_by_k[best_k], index=X.index, name="cluster"),
        ari_best=float(adjusted_rand_score(categories, labels_by_k[best_k])),
        k_categories=k_cat,
        labels_k_categories=pd.Series(lab_cat, index=X.index, name="cluster"),
        ari_k_categories=float(adjusted_rand_score(categories, lab_cat)),
        silhouette_k_categories=float(silhouette_score(Z, lab_cat)),
        pca=pd.DataFrame(coords, index=X.index, columns=["pc1", "pc2"]),
        pca_explained=tuple(float(v) for v in pca.explained_variance_ratio_),
    )


# --- label check (classifier) --------------------------------------------------------------------

@dataclass
class ModelScore:
    name: str
    accuracy: float
    macro_f1: float
    confusion: pd.DataFrame | None = None


@dataclass
class ClassifierResult:
    classes: list[str]
    merged: list[str]
    scores: list[ModelScore]
    baseline: ModelScore
    flag_model: str
    oof: pd.DataFrame  # per fund: target, predicted, p_predicted, p_own, flag, flag_category, ...
    class_counts: pd.Series = field(default_factory=pd.Series)
    excluded: pd.Index = field(default_factory=pd.Index)  # funds whose class is still too small


def models() -> dict:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        "Random forest": RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE),
    }


def run_classifier(X: pd.DataFrame, categories: pd.Series) -> ClassifierResult:
    y, merged = merge_small_categories(categories)
    if merged:
        log.info("Merged into %r (fewer than %d funds): %s", OTHER, MIN_CLASS_SIZE, merged)
    # If the merged bucket itself is still too small, CV can't evaluate it: leave those funds out.
    counts = y.value_counts()
    too_small = counts[counts < MIN_CLASS_SIZE].index
    excluded = y.index[y.isin(too_small)]
    if len(excluded):
        log.info("Left out of the classifier (class still < %d funds): %d funds in %s",
                 MIN_CLASS_SIZE, len(excluded), list(too_small))
    X, y, categories = X.drop(excluded), y.drop(excluded), categories.drop(excluded)
    classes = sorted(y.unique())
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def score(name, pred):
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)
        return ModelScore(name, float(accuracy_score(y, pred)),
                          float(f1_score(y, pred, average="macro", zero_division=0)), cm)

    scores, probas = [], {}
    for name, model in models().items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        probas[name] = pd.DataFrame(proba, index=X.index, columns=classes)
        scores.append(score(name, probas[name].idxmax(axis=1)))
    base_pred = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=cv)
    baseline = score("Majority-class baseline", base_pred)

    # Flags come from the model with the better cross-validated macro-F1.
    best = max(scores, key=lambda s: s.macro_f1).name
    p = probas[best]
    p_own = pd.Series([p.at[i, y[i]] for i in p.index], index=p.index)
    others = p.copy()
    for i in p.index:
        others.at[i, y[i]] = -1.0
    if OTHER in others:
        others[OTHER] = -1.0  # "behaves like Other equity" is not a meaningful statement
    flag_cat, flag_p = others.idxmax(axis=1), others.max(axis=1)
    oof = pd.DataFrame({
        "category": categories,
        "target": y,
        "predicted": p.idxmax(axis=1),
        "p_predicted": p.max(axis=1),
        "p_own": p_own,
        "flag": flag_p >= FLAG_PROB,
        "flag_category": flag_cat.where(flag_p >= FLAG_PROB),
        "flag_prob": flag_p.where(flag_p >= FLAG_PROB),
    })
    return ClassifierResult(classes, merged, scores, baseline, best, oof, y.value_counts(),
                            excluded)


# --- everything together -------------------------------------------------------------------------

@dataclass
class Analysis:
    bench_code: int
    features: pd.DataFrame  # all funds (may contain NaN)
    used: pd.Index  # funds with complete features (used for clustering / classifier)
    corr: pd.DataFrame
    sanity: pd.DataFrame
    clusters: ClusterResult
    classifier: ClassifierResult


def analyse(ds: Dataset) -> Analysis:
    bench = pick_benchmark(ds.schemes, ds.matrix)
    feats = build_features(ds.matrix, ds.returns, bench)
    complete = feats.dropna()
    cats = ds.schemes.set_index("scheme_code")["scheme_category"].reindex(complete.index)
    corr = correlation_matrix(ds.returns)
    return Analysis(
        bench_code=bench,
        features=feats,
        used=complete.index,
        corr=corr,
        sanity=nifty50_sanity(corr, ds.schemes),
        clusters=run_clustering(complete, cats),
        classifier=run_classifier(complete, cats),
    )


def cluster_summary(an: Analysis, schemes: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: size, most common category, and median behaviour."""
    names = schemes.set_index("scheme_code")["scheme_category"].map(short_category)
    df = an.features.loc[an.used].assign(cluster=an.clusters.labels, category=names)
    rows = []
    for c, g in df.groupby("cluster"):
        top = g["category"].value_counts()
        rows.append({
            "Cluster": int(c),
            "Funds": len(g),
            "Most common category": f"{top.index[0]} ({top.iloc[0] / len(g):.0%})",
            "Median 3y CAGR": g["cagr_3y"].median(),
            "Median volatility": g["vol_3y"].median(),
            "Median beta": g["beta_3y"].median(),
            "Median worst fall": g["max_dd_5y"].median(),
        })
    return pd.DataFrame(rows)


# --- report --------------------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):  # keeps ints as ints (iterrows would upcast)
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _agreement(ari: float) -> str:
    if ari < 0.2:
        return "weak"
    if ari < 0.5:
        return "moderate"
    return "strong"


def build_report(ds: Dataset, an: Analysis) -> str:
    s = ds.schemes.set_index("scheme_code")
    clf, cl = an.classifier, an.clusters
    out = []
    if ds.is_demo:
        out += [
            "# ⚠️ DEMO DATA, NOT REAL",
            "",
            ("> Every number below comes from **synthetic** funds made by "
            "`scripts/make_demo_data.py`. The categories in that data were *simulated* with "
            "different factor loadings, so good clustering or classification scores here only "
            "show that the pipeline works. **They say nothing about real Indian mutual funds.** "
            "See `docs/MORNING_STEPS.md` to regenerate this report on real data."),
            "",
        ]
    out += [
        "# Evaluation",
        "",
        (f"_Generated by `python -m fundx.ml --report` on {date.today().isoformat()}. "
        "Do not edit by hand._"),
        "",
        "## Data",
        "",
        f"- Data: {'**synthetic demo**' if ds.is_demo else '**real** (mfapi.in / AMFI NAVs)'}",
        f"- Funds: {len(ds.schemes)}; with all 8 features (used for ML): {len(an.used)}",
        (f"- NAV dates: {ds.matrix.index.min().date()} to {ds.as_of.date()} "
        f"({len(ds.matrix)} trading days)"),
        f"- Nifty 50 proxy: {s.at[an.bench_code, 'scheme_name']} (`{an.bench_code}`)",
        "",
        "## Features",
        "",
        "One row per fund, standardised (`StandardScaler`) before k-means and logistic regression:",
        "",
    ]
    out += [f"- `{k}`: {v}" for k, v in FEATURES.items()]

    # Lookalikes
    san = an.sanity
    out += ["", "## Lookalikes sanity check", "",
            (f"Lookalikes = Pearson correlation of daily returns over the last {WINDOW_YEARS} years "
            f"(at least {MIN_OVERLAP} shared trading days). Check: each Nifty 50 index fund's top "
            f"lookalike should be another Nifty 50 index fund with correlation ≥ {SANITY_CORR}."),
            ""]
    if san.empty:
        out.append("No Nifty 50 index funds found, so the check could not run.")
    else:
        rate = san["passed"].mean()
        out += [f"**Pass rate: {rate:.0%} ({int(san['passed'].sum())} of {len(san)}).**", ""]
        tbl = pd.DataFrame({
            "Index fund": [s.at[c, "scheme_name"] for c in san["scheme_code"]],
            "Top lookalike": [s.at[c, "scheme_name"] if pd.notna(c) else "—"
                              for c in san["top_lookalike"]],
            "Correlation": san["corr"].astype(float),
            "Passed": san["passed"].map({True: "yes", False: "no"}),
        })
        out.append(_md_table(tbl, "{:.4f}"))

    # Clustering
    sil = pd.DataFrame({"k": list(cl.silhouettes), "Silhouette": list(cl.silhouettes.values())})
    out += [
        "", "## Clustering (k-means)", "",
        (f"k-means on the standardised features for k = {min(cl.silhouettes)} to "
        f"{max(cl.silhouettes)}, with k chosen by silhouette score."), "",
        _md_table(sil), "",
        (f"- **Best k by silhouette: {cl.best_k}** (silhouette {cl.silhouettes[cl.best_k]:.3f}). "
        f"Adjusted Rand index vs official categories: **{cl.ari_best:.3f}** "
        f"({_agreement(cl.ari_best)} agreement)."),
        (f"- **k = number of distinct categories ({cl.k_categories})**: silhouette "
        f"{cl.silhouette_k_categories:.3f}, ARI **{cl.ari_k_categories:.3f}** "
        f"({_agreement(cl.ari_k_categories)} agreement)."),
        (f"- PCA (for the 2D scatter in the app) explains {cl.pca_explained[0]:.0%} + "
        f"{cl.pca_explained[1]:.0%} of the variance."),
        "",
        ("**What this means.** The adjusted Rand index (ARI) is 1 when clusters match the official "
        "categories exactly and about 0 when they agree no better than chance. Official categories "
        "are SEBI rules about *what a fund may hold* (for example, at least 80% in the top 100 "
        "stocks for Large Cap). Clusters are about *how the price moved*. A low ARI means "
        "funds in different categories often behave alike, which is the finding, not a bug."),
    ]
    if ds.is_demo:
        out += ["", ("On demo data the categories were simulated with different factor loadings, "
                "so the ARI mostly reflects how the generator was set up.")]

    # Classifier
    counts = clf.class_counts.sort_index()
    out += [
        "", "## Label check (category classifier)", "",
        (f"Target: `scheme_category`. Categories with fewer than {MIN_CLASS_SIZE} funds are merged "
        f"into \"{OTHER}\": {', '.join(clf.merged) if clf.merged else 'none'}."), "",
        "Class sizes: " + ", ".join(f"{short_category(k)} {v}" for k, v in counts.items()) + ".",
        "",
    ]
    if len(clf.excluded):
        out += [f"Even after merging, \"{OTHER}\" had fewer than {MIN_CLASS_SIZE} funds, so its "
                f"{len(clf.excluded)} fund(s) are **left out** of the classifier (too few to "
                "cross-validate): "
                + ", ".join(s.at[c, "scheme_name"] for c in clf.excluded) + ".", ""]
    out += [
        (f"{N_FOLDS}-fold **stratified** cross-validation (shuffle, random_state={RANDOM_STATE}). "
        "All numbers are out-of-fold."), "",
    ]
    tbl = pd.DataFrame([{"Model": m.name, "Accuracy": m.accuracy, "Macro-F1": m.macro_f1}
                        for m in [*clf.scores, clf.baseline]])
    out += [_md_table(tbl), ""]
    best = max(clf.scores, key=lambda m: m.macro_f1)
    out.append(f"Best model by macro-F1: **{best.name}**: accuracy {best.accuracy:.1%} vs "
               f"{clf.baseline.accuracy:.1%} for always guessing the largest class.")
    for m in clf.scores:
        cm = m.confusion.copy()
        cm.index = [short_category(c) for c in cm.index]
        cm.columns = [short_category(c) for c in cm.columns]
        out += ["", f"**Confusion matrix: {m.name}** (rows = official, columns = predicted)", "",
                _md_table(cm.reset_index(names="official \\ predicted"), "{:d}")]

    flagged = clf.oof[clf.oof["flag"]]
    out += [
        "", "## Funds that behave like another category", "",
        (f"A fund is flagged only when the out-of-fold probability ({clf.flag_model}) for a "
        f"*different* category is ≥ {FLAG_PROB}. This does **not** mean it is mislabelled. "
        "Category is about holdings, and this check only looks at past price behaviour."), "",
    ]
    if flagged.empty:
        out.append("No funds were flagged.")
    else:
        f = an.features.loc[flagged.index]
        tbl = pd.DataFrame({
            "Fund": [s.at[c, "scheme_name"] for c in flagged.index],
            "Official": flagged["category"].map(short_category).to_numpy(),
            "Behaves like": flagged["flag_category"].map(short_category).to_numpy(),
            "Prob.": flagged["flag_prob"].to_numpy(),
        })
        for col in FEATURES:
            tbl[col] = f[col].to_numpy()
        out += [f"**{len(flagged)} fund(s) flagged.**", "", _md_table(tbl, "{:.2f}")]

    out += [
        "", "## Limitations", "",
        ("- **Past behaviour only.** Every number describes history. None of it predicts future "
        "returns."),
        ("- **Survivorship bias.** Only funds alive today with at least 5 years of history are "
        "included. Funds that were closed or merged (often the poor performers) are missing, so "
        "averages look better than reality."),
        ("- **Direct plan, Growth option only.** Regular plans (higher fees) and IDCW options are "
        "excluded."),
        ("- **No expense ratios or holdings.** The label check sees prices, not portfolios, so a "
        "flag is a prompt to look at the portfolio, not a verdict."),
        ("- **One Nifty 50 index fund stands in for the Nifty 50.** Its small fees and tracking "
        "error are included."),
        (f"- **Small classes.** With few funds per category, CV estimates are noisy. Categories "
        f"under {MIN_CLASS_SIZE} funds are merged."),
    ]
    if ds.is_demo:
        out.append("- **Synthetic data.** This whole report uses demo data. See the top of the page.")
    return "\n".join(out) + "\n"


def write_report(path=REPO_ROOT / "docs" / "EVALUATION.md") -> str:
    ds = load_dataset()
    text = build_report(ds, analyse(ds))
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="write docs/EVALUATION.md")
    args = ap.parse_args()
    if args.report:
        print(write_report())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
