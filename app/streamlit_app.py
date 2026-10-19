"""Fund Explained: a plain-English report card for any Indian mutual fund.

    streamlit run app/streamlit_app.py
"""
import ui  # noqa: I001 - must come first: puts the repo root on sys.path for `fundx`
import charts
import pandas as pd
import streamlit as st

from fundx import report as rp
from fundx.fmt import fmt_inr, fmt_pct
from fundx.glossary import GLOSSARY
from fundx.lookalikes import MIN_OVERLAP
from fundx.ml import FLAG_PROB, MIN_CLASS_SIZE, OTHER, build_report, cluster_summary, short_category
from fundx.search import fuzzy_search

st.set_page_config(page_title="Fund Explained", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")
ui.inject_css()
ds, an = ui.load()
schemes = ds.schemes
info = schemes.set_index("scheme_code")
bench_name = info.at[an.bench_code, "scheme_name"]


def pick(code: int) -> None:
    """Button callback: select a fund (clearing filters so it is always in the list)."""
    st.session_state["q"] = ""
    st.session_state["cats"] = []
    st.session_state["houses"] = []
    st.session_state["fund"] = code


# --- sidebar -------------------------------------------------------------------------------------

def sidebar() -> int | None:
    with st.sidebar:
        st.markdown("### 🔎 Find a fund")
        q = st.text_input("Search by name", key="q", placeholder="e.g. parag flexi, nifty 50")
        cat_short = schemes["scheme_category"].map(short_category)
        cats = st.multiselect("Category", sorted(cat_short.unique()), key="cats")
        houses = st.multiselect("Fund house", sorted(schemes["fund_house"].dropna().unique()),
                                key="houses")
        mask = pd.Series(True, index=schemes.index)
        if cats:
            mask &= cat_short.isin(cats)
        if houses:
            mask &= schemes["fund_house"].isin(houses)
        pool = schemes[mask]
        names = pool.groupby("scheme_name")["scheme_code"].apply(list).to_dict()
        codes = [c for n in fuzzy_search(q, sorted(names)) for c in names[n]]
        current = st.session_state.get("fund")
        if current is not None and current not in codes:
            codes = [current, *codes]
        st.selectbox(f"Pick a fund ({len(codes)} match{'es' if len(codes) != 1 else ''})", codes,
                     index=None, key="fund", format_func=lambda c: info.at[c, "scheme_name"],
                     placeholder="Choose a fund…")
        st.markdown("### 📖 What do these words mean?")
        for term, text in GLOSSARY.items():
            with st.expander(term):
                st.write(text)
    return st.session_state.get("fund")


# --- landing -------------------------------------------------------------------------------------

def landing() -> None:
    st.markdown(
        '<div class="fx-card fx-hero fx-landing"><h1>Understand any mutual fund in 30 seconds</h1>'
        '<p class="fx-sub">Pick a fund to see what ₹10,000 would have become, its worst fall, how '
        'it moves with the Nifty, and which funds it behaves like, all in plain English. '
        'We explain; we never recommend.</p>'
        f'<div class="fx-asof">{len(schemes)} funds · data as of {rp.fmt_date(ds.as_of)}</div>'
        '</div>', unsafe_allow_html=True)
    st.markdown("**Try one of these:**")
    examples = rp.example_funds(schemes, an.bench_code)
    for col, code in zip(st.columns(len(examples)), examples):
        col.button(rp.short_name(info.at[code, "scheme_name"]), key=f"ex{code}",
                   on_click=pick, args=(code,), width="stretch")
    st.markdown("#### How it works")
    steps = [
        ("🔎", "Pick a fund", ("Search by name in the sidebar, or filter by category and fund "
                              "house.")),
        ("📊", "Read the report card", ("Growth of ₹10,000, the worst fall, and risk compared "
                                       "with the Nifty, in plain words.")),
        ("👯", "See how it behaves", ("Which funds move almost the same, and whether it acts like "
                                     "its official category.")),
    ]
    st.markdown('<div class="fx-how">' + "".join(
        f'<div class="fx-card"><div class="fx-step">{i}</div><h4>{ui.esc(t)}</h4>'
        f'<p>{ui.esc(d)}</p></div>' for i, t, d in steps) + "</div>", unsafe_allow_html=True)


# --- one fund ------------------------------------------------------------------------------------

HOW_GROWTH = ("₹10,000 × (latest NAV ÷ the last NAV on or before the same date N years "
              "earlier).")
HOW_CAGR = "(latest NAV ÷ NAV N years ago) ^ (1 ÷ exact years between the two dates) − 1."
HOW_FALL = ("Biggest drop from a high point to a later low in the last 5 years; recovery = "
            "time from the low until the NAV first got back to that high.")
HOW_RISK = ("Volatility (std of daily returns × √252, last 3 years) ranked against every fund "
            "here: lowest 25% = Low, then Moderate, High, highest 25% = Very High.")
HOW_BETA = ("cov(fund, Nifty) ÷ var(Nifty), on daily returns over the same dates in the last 3 "
            "years.")
HOW_CORR = "Pearson correlation of the fund's and the Nifty's daily returns, last 3 years."
HOW_VOL = "Standard deviation of daily returns over the last 3 years × √252 (trading days)."


def growth_tile(r: rp.FundReport, years: int) -> str:
    g = r.growth[years]
    if g is None:
        return ui.tile(f"₹10,000, {years} yr ago", "—", note="Not enough history", how=HOW_GROWTH)
    change = g / 10_000 - 1
    return ui.tile(f"₹10,000, {years} yr{'s' if years > 1 else ''} ago", fmt_inr(g),
                   delta=f"{fmt_pct(change, 1, signed=True)} total", up=change >= 0,
                   note="is worth now", how=HOW_GROWTH)


def fund_page(code: int) -> None:
    r = rp.build_report(ds, an, code)
    risk_cls = f"fx-{(r.risk_label or '').replace(' ', '')}"
    chips = ui.chip(rp.category_name(r.category).removesuffix(" fund")) + (
        ui.chip(f"Risk: {r.risk_label}", risk_cls) if r.risk_label else "")
    st.markdown(
        f'<div class="fx-card fx-hero"><h1>{ui.esc(r.name)}</h1>'
        f'<p class="fx-sub">{ui.esc(r.fund_house)} · {ui.esc(r.category)}</p>{chips}'
        f'<div class="fx-asof">Data as of {rp.fmt_date(r.as_of)} · NAV history since '
        f'{rp.fmt_date(r.first_date)}</div></div>', unsafe_allow_html=True)
    if r.is_benchmark:
        st.info("This fund is the app's stand-in for the Nifty 50, so its beta and correlation "
                "are 1 by definition.")

    # Report card
    st.markdown("### 🧾 Report card")
    dd = r.drawdown
    cagr_note = f"3 years: {fmt_pct(r.cagr[3])} a year" if r.cagr[3] is not None else ""
    ui.tiles([
        *(growth_tile(r, y) for y in (1, 3, 5)),
        ui.tile("Yearly growth (CAGR), 5 yrs", fmt_pct(r.cagr[5]), note=cagr_note, how=HOW_CAGR),
        ui.tile("Worst fall (drawdown), 5 yrs", fmt_pct(dd.drawdown if dd else None),
                note=rp.recovery_text(dd), how=HOW_FALL),
        ui.tile("Risk level", r.risk_label or "—",
                note=(f"Riskier than {r.risk_percentile:.0f}% of funds here"
                      if r.risk_percentile is not None else ""), how=HOW_RISK),
    ])
    ui.how_calculated([
        (f"**₹10,000 → ₹X:** {HOW_GROWTH} Uses the published NAV, so fund costs are already "
        "taken out."),
        f"**Yearly growth (CAGR):** {HOW_CAGR}",
        f"**Worst fall:** {HOW_FALL}",
        f"**Risk level:** {HOW_RISK} It is relative: a Low here is still a stock fund.",
    ])

    # Growth chart
    st.markdown("### 📈 Growth of ₹10,000 over 5 years")
    fig = charts.growth_chart(ds.matrix, code, an.bench_code, dd, "This fund", ui.is_dark())
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Solid line: this fund. Dashed grey: the Nifty 50. Red band: the worst fall.")
    ui.how_calculated([
        ("Both lines start at ₹10,000 on the same date five years ago (or the fund's first NAV, "
        "if later) and follow each day's published NAV."),
        f"The Nifty 50 line uses an index fund's NAV as a stand-in: {bench_name}.",
        "The red band runs from the peak to the bottom of the worst fall in the window.",
    ])

    # Worst fall
    st.markdown("### 🌊 Worst fall")
    ui.card(f"<p>{ui.esc(rp.drawdown_story(dd))}</p>")
    ui.how_calculated([
        HOW_FALL,
        ("Months = calendar days ÷ 30.44, rounded. Missing NAV days are skipped, never "
        "guessed."),
    ])

    # Risk vs Nifty
    st.markdown("### 🧭 Risk: how it moves vs the Nifty")
    ui.tiles([
        ui.tile("Beta", f"{r.beta:.2f}" if r.beta is not None else "—", how=HOW_BETA,
                note="1.00 = moves like the Nifty"),
        ui.tile("Correlation", f"{r.correlation:.2f}" if r.correlation is not None else "—",
                how=HOW_CORR, note="1.00 = perfectly in step"),
        ui.tile("Volatility (yearly)", fmt_pct(r.volatility), how=HOW_VOL,
                note="how bumpy the ride is"),
    ], "fx-compact")
    ui.card(f"<p>{ui.esc(rp.beta_sentence(r.beta))}</p>"
            f"<p>{ui.esc(rp.correlation_sentence(r.correlation))}</p>")
    ui.how_calculated([f"**Beta:** {HOW_BETA}", f"**Correlation:** {HOW_CORR}",
                       f"**Volatility:** {HOW_VOL}"])

    # Lookalikes
    st.markdown("### 👯 Funds that behave almost the same")
    look = r.lookalikes
    if look.empty:
        st.write("Not enough shared history with other funds to compare.")
    else:
        table = pd.DataFrame({
            "Fund": look["name"].map(rp.short_name),
            "Category": look["category"],
            "Moves alike": (look["corr"] * 100).round(1),
        })
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Moves alike": st.column_config.ProgressColumn(
                "Moves alike", format="%.1f%%", min_value=0, max_value=100,
                help="Correlation of daily returns over the last 3 years, as a percentage."),
        })
        if look["is_index"].any():
            st.caption("Index funds simply copy an index.")
    ui.how_calculated([
        ("Correlation of daily returns over the last 3 years between this fund and every other "
        f"fund, using only days both have a NAV (at least {MIN_OVERLAP} shared days)."),
        "The 5 highest are shown. 100% would mean the two moved in perfect lockstep.",
        ("Moving alike in the past doesn't mean they hold the same stocks or will keep moving "
        "alike."),
    ])

    # Label check
    st.markdown("### 🏷️ Does it behave like its category?")
    ui.card(f"<p>{ui.esc(r.label_text)}</p>", "fx-flag" if r.label_flagged else "fx-ok")
    best = max(an.classifier.scores, key=lambda s: s.macro_f1)
    ui.how_calculated([
        (f"A {best.name.lower()} learned to guess each fund's official category from 8 numbers "
        "about its past behaviour (growth, volatility, worst fall, beta, and more)."),
        ("Each fund is judged by a model that never saw it during training (5-fold "
        f"cross-validation). That model guesses the right category {best.accuracy:.0%} of the "
        f"time, against {an.classifier.baseline.accuracy:.0%} for always guessing the most "
        "common one."),
        (f"A fund is flagged only if the model is at least {FLAG_PROB:.0%} sure it looks like a "
        "*different* category. Categories with fewer than "
        f"{MIN_CLASS_SIZE} funds are grouped as \"{OTHER}\"."),
        "Category is about what a fund may hold; this check only sees prices.",
    ])


def report_page() -> None:
    code = sidebar()
    if code is None:
        landing()
    else:
        fund_page(code)


# --- explore -------------------------------------------------------------------------------------

def explore_page() -> None:
    st.markdown("## 🗺️ Explore all funds")
    st.write("Every dot is a fund. Funds close together behaved alike over the last 3 to 5 "
             "years. Colours show groups the computer found on its own (k-means clusters); hover "
             "a dot to see the fund's official category.")
    cl = an.clusters
    st.plotly_chart(charts.cluster_scatter(an, schemes, ui.is_dark()), width="stretch",
                    config={"displayModeBar": False})
    ui.how_calculated([
        ("Each fund is described by 8 numbers: 3- and 5-year CAGR, volatility, worst fall, beta, "
        "correlation with the Nifty, downside volatility and tracking error. They're scaled so "
        "each counts equally."),
        (f"k-means split the funds into {cl.best_k} groups; {cl.best_k} had the best silhouette "
        "score (how clearly separated the groups are) among 3 to 12 groups."),
        ("To draw them in 2D, PCA squeezes the 8 numbers into 2 'behaviour patterns' that keep "
        f"{sum(cl.pca_explained):.0%} of the differences between funds."),
        (f"How well groups match official categories (adjusted Rand index, 1 = perfectly, "
        f"0 = chance): {cl.ari_best:.2f}."),
    ])

    st.markdown("#### What each group looks like")
    summ = cluster_summary(an, schemes)
    summ["Cluster"] = summ["Cluster"] + 1
    for col in ("Median 3y CAGR", "Median volatility", "Median worst fall"):
        summ[col] = summ[col].map(fmt_pct)
    summ["Median beta"] = summ["Median beta"].map("{:.2f}".format)
    st.dataframe(summ.rename(columns={"Cluster": "Group"}), hide_index=True, width="stretch")
    ui.how_calculated([
        ("Medians across the funds in each group: 3-year CAGR, volatility (last 3 years), beta "
        "vs the Nifty (last 3 years) and worst fall (last 5 years)."),
        "\"Most common category\" is the official category most funds in that group carry.",
    ])

    st.markdown("#### Evaluation report")
    st.caption("The same report as docs/EVALUATION.md, generated live from the data the app is "
               "using.")
    text = build_report(ds, an)
    demoted = "\n".join("#" + line if line.startswith("#") else line for line in text.splitlines())
    with st.container(border=True):
        st.markdown(demoted)


# --- run -----------------------------------------------------------------------------------------

ui.demo_banner(ds.is_demo)
nav = st.navigation([
    st.Page(report_page, title="Report card", icon="📈", default=True),
    st.Page(explore_page, title="Explore all funds", icon="🗺️", url_path="explore"),
])
nav.run()
ui.footer(ds.is_demo)
