"""Everything the report card shows for one fund, plus the plain-English wording.

Kept free of Streamlit so the numbers and sentences can be tested.
"""
from dataclasses import dataclass

import pandas as pd

from fundx import metrics as m
from fundx.data import Dataset
from fundx.fmt import fmt_pct
from fundx.lookalikes import top_lookalikes
from fundx.ml import FLAG_PROB, OTHER, Analysis, short_category

DAYS_PER_MONTH = 30.44


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%-d %b %Y")


def category_name(cat: str) -> str:
    """'Equity Scheme - Large Cap Fund' -> 'Large Cap fund'; index funds -> 'Index fund'."""
    short = short_category(cat)
    return f"{'Index' if short == 'Index Funds' else short} fund"


def category_noun(cat: str) -> str:
    """With an article: 'a Large Cap fund', 'an Index fund'."""
    name = category_name(cat)
    return f"{'an' if name[:1].upper() in 'AEIOU' else 'a'} {name}"


def is_index_category(cat: str) -> bool:
    return "index" in str(cat).lower()


@dataclass
class FundReport:
    code: int
    name: str
    fund_house: str
    category: str
    as_of: pd.Timestamp
    first_date: pd.Timestamp
    growth: dict[int, float | None]
    cagr: dict[int, float | None]
    drawdown: m.Drawdown | None
    volatility: float | None
    risk_label: str | None
    risk_percentile: float | None
    beta: float | None
    correlation: float | None
    lookalikes: pd.DataFrame
    label_text: str
    label_flagged: bool
    is_benchmark: bool


def build_report(ds: Dataset, an: Analysis, code: int) -> FundReport:
    s = ds.schemes.set_index("scheme_code").loc[code]
    nav, r = ds.matrix[code], ds.returns[code]
    bench_r = ds.returns[an.bench_code]
    vol = m.volatility(m.last_years(r, 3))
    all_vols = an.features["vol_3y"]

    look = top_lookalikes(an.corr, code)
    info = ds.schemes.set_index("scheme_code")
    lookalikes = pd.DataFrame({
        "scheme_code": look.index.astype(int),
        "name": [info.at[c, "scheme_name"] for c in look.index],
        "category": [short_category(info.at[c, "scheme_category"]) for c in look.index],
        "corr": look.to_numpy(),
        "is_index": [is_index_category(info.at[c, "scheme_category"]) for c in look.index],
    })
    text, flagged = label_check_text(an, code, s["scheme_category"])
    return FundReport(
        code=code,
        name=s["scheme_name"],
        fund_house=s["fund_house"],
        category=s["scheme_category"],
        as_of=nav.dropna().index[-1],
        first_date=nav.dropna().index[0],
        growth={y: m.growth_of_10k(nav, y) for y in (1, 3, 5)},
        cagr={y: m.cagr(nav, y) for y in (1, 3, 5)},
        drawdown=m.max_drawdown(nav, 5),
        volatility=vol,
        risk_label=m.risk_label(vol, all_vols),
        risk_percentile=m.risk_percentile(vol, all_vols),
        beta=m.beta_vs_benchmark(r, bench_r, 3),
        correlation=m.correlation_vs_benchmark(r, bench_r, 3),
        lookalikes=lookalikes,
        label_text=text,
        label_flagged=flagged,
        is_benchmark=code == an.bench_code,
    )


# --- wording -------------------------------------------------------------------------------------

def beta_sentence(beta: float | None) -> str:
    if beta is None:
        return "Not enough shared history with the Nifty to measure this."
    if beta > 1.05:
        tail = "It has swung more than the market."
    elif beta < 0.95:
        tail = "It has swung less than the market."
    else:
        tail = "It has moved about as much as the market."
    return (f"When the Nifty rises 1%, this fund has typically risen about {beta:.2f}% "
            f"(and fallen about as much when the Nifty falls). {tail}")


def correlation_sentence(corr: float | None) -> str:
    if corr is None:
        return "Not enough shared history with the Nifty to measure this."
    if corr >= 0.95:
        how = "almost always moves in step with"
    elif corr >= 0.8:
        how = "usually moves in the same direction as"
    elif corr >= 0.5:
        how = "often, but not always, moves with"
    else:
        how = "only loosely follows"
    return f"On a typical day this fund {how} the Nifty (1.00 would be perfectly in step)."


def months_text(days: int) -> str:
    n = max(1, round(days / DAYS_PER_MONTH))
    return f"{n} month{'s' if n > 1 else ''}"


def recovery_text(dd: m.Drawdown | None) -> str:
    if dd is None:
        return "Not enough history."
    if dd.drawdown == 0:
        return "No fall from a previous high in this period."
    if dd.recovery_days is None:
        return "Not yet recovered"
    return f"Recovered in about {months_text(dd.recovery_days)}"


def drawdown_story(dd: m.Drawdown | None) -> str:
    if dd is None or dd.drawdown == 0:
        return recovery_text(dd)
    story = (f"From its high on {fmt_date(dd.peak_date)}, the fund fell {fmt_pct(-dd.drawdown, 1)} "
             f"to its low on {fmt_date(dd.trough_date)}.")
    if dd.recovery_date is None:
        return story + " It has not yet climbed back to that high."
    return story + (f" It got back to the old high on {fmt_date(dd.recovery_date)}, "
                    f"about {months_text(dd.recovery_days)} after the low.")


def label_check_text(an: Analysis, code: int, category: str) -> tuple[str, bool]:
    """(sentence, flagged). Careful wording: behaviour is not the same as holdings."""
    clf = an.classifier
    own = category_noun(category)
    if code in clf.excluded:
        return ((f"Not enough funds like this in the data to run this check "
                f"({category_name(category)}s are too few here to learn from)."), False)
    if code not in clf.oof.index:
        return "Not enough history to run this check for this fund.", False
    row = clf.oof.loc[code]
    if row["flag"]:
        other = category_noun(row["flag_category"])
        return ((f"Based on its past price movements, this fund has behaved more like "
                f"{other} than {own}. This doesn't mean it's mislabelled; check its portfolio. "
                f"(Model confidence: {row['flag_prob']:.0%}.)"), True)
    if row["predicted"] == row["target"] == OTHER:
        return ((f"The model could only compare it with a mixed group of smaller categories "
                 f"(\"{OTHER}\"), and it fits that group ({row['p_own']:.0%}). No strong "
                 "signal either way."), False)
    if row["predicted"] == row["target"]:
        return ((f"Based on its past price movements, this fund has behaved like a typical "
                f"{category_name(category)} (model confidence: {row['p_own']:.0%})."), False)
    leaned = category_noun(row["predicted"])
    return ((f"Its past price movements are a mix. The model leaned slightly towards {leaned} "
            f"({row['p_predicted']:.0%}), which is not enough to flag: we only flag at "
            f"{FLAG_PROB:.0%} or more."), False)


# Well-known, large schemes to offer as examples on the landing page (if present in the data).
WELL_KNOWN = ("Parag Parikh Flexi Cap", "HDFC Flexi Cap", "ICICI Prudential Bluechip",
              "ICICI Prudential Large Cap", "SBI Large Cap", "SBI Bluechip",
              "Nippon India Small Cap", "UTI Nifty 50 Index")


def example_funds(schemes: pd.DataFrame, bench_code: int, n: int = 3) -> list[int]:
    """Up to n scheme codes: well-known names first, then the oldest Flexi/Large Cap funds and
    the Nifty proxy."""
    picks: list[int] = []
    for key in WELL_KNOWN:
        hit = schemes[schemes["scheme_name"].str.contains(key, case=False, regex=False)]
        if len(hit) and int(hit.iloc[0]["scheme_code"]) not in picks:
            picks.append(int(hit.iloc[0]["scheme_code"]))
        if len(picks) == n:
            return picks
    by_age = schemes.sort_values(["first_date", "scheme_code"])
    for cat in ("Flexi Cap", "Large Cap"):
        hit = by_age[by_age["scheme_category"].str.contains(cat, regex=False)]
        if len(hit) and len(picks) < n and int(hit.iloc[0]["scheme_code"]) not in picks:
            picks.append(int(hit.iloc[0]["scheme_code"]))
    if len(picks) < n and bench_code not in picks:
        picks.append(bench_code)
    for code in by_age["scheme_code"]:
        if len(picks) >= n:
            break
        if int(code) not in picks:
            picks.append(int(code))
    return picks


def short_name(name: str) -> str:
    """Drop the plan/option suffix: 'X Fund - Direct Plan - Growth' -> 'X Fund'."""
    return str(name).split(" - ")[0].strip()
