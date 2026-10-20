"""Plotly charts with one consistent palette and ₹ hover labels (Indian digit grouping)."""
import math

import pandas as pd
import plotly.graph_objects as go

from fundx import metrics as m
from fundx.fmt import fmt_inr, fmt_pct
from fundx.ml import Analysis, short_category

# Fund = indigo, Nifty = slate grey (dashed, a muted reference line), worst fall = red band.
# Light/dark steps checked with the dataviz palette validator (separation passes in both modes).
FUND = {"light": "#4F46E5", "dark": "#7C7FF2"}
NIFTY = "#64748B"
FALL = "#EF4444"
# Validated 8-slot categorical order (light, dark). Clusters beyond 8 fold into grey.
CLUSTER = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
              "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
             "#e66767"],
}
GRID = "rgba(148, 163, 184, 0.18)"
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon"]


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values (1, 2, 2.5, 5 x 10^k steps) covering [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start = math.floor(lo / step) * step
    return [start + i * step for i in range(math.ceil((hi - start) / step) + 1)]


def _base_layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height, margin={"l": 8, "r": 8, "t": 36, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hoverlabel={"namelength": -1},
    )


def growth_chart(matrix: pd.DataFrame, code: int, bench_code: int, dd: m.Drawdown | None,
                 fund_label: str, dark: bool = False) -> go.Figure:
    mode = "dark" if dark else "light"
    nav = matrix[code].dropna()
    start = m.window_start(nav, 5) or nav.index[0]
    f = nav[nav.index >= start]
    f = 10_000 * f / f.iloc[0]
    fig = go.Figure()
    series = [(f, fund_label, FUND[mode], "solid", 2.5)]
    if bench_code != code:
        b = matrix[bench_code].dropna()
        before = b[b.index <= start]
        b0 = before.iloc[-1] if len(before) else b.iloc[0]
        b = 10_000 * b[b.index >= start] / b0
        series.insert(0, (b, "Nifty 50", NIFTY, "dash", 2))
    for s, name, color, dash, width in series:
        # The legend entry carries the end value, so identity never rests on colour alone and
        # labels can't collide at the line ends (they did when both lines finish close together).
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=f"{name}: {fmt_inr(s.values[-1])} now", mode="lines",
            line={"color": color, "width": width, "dash": dash},
            customdata=[fmt_inr(v) for v in s.values],
            hovertemplate="%{customdata}<extra>" + name + "</extra>",
        ))
    if dd is not None and dd.peak_date is not None:
        fig.add_vrect(x0=dd.peak_date, x1=dd.trough_date, fillcolor=FALL, opacity=0.13,
                      line_width=0, layer="below",
                      annotation_text=f"Worst fall {fmt_pct(dd.drawdown, 1)}",
                      annotation_position="top left", annotation_font_color=FALL)
    lo = min(s.values.min() for s, *_ in series)
    hi = max(s.values.max() for s, *_ in series)
    ticks = nice_ticks(lo, hi)
    _base_layout(fig, 400)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showgrid=False, hoverformat="%d %b %Y")
    fig.update_yaxes(tickvals=ticks, ticktext=[fmt_inr(t) for t in ticks], gridcolor=GRID,
                     zeroline=False)
    return fig


def cluster_scatter(an: Analysis, schemes: pd.DataFrame, dark: bool = False) -> go.Figure:
    palette = CLUSTER["dark" if dark else "light"]
    info = schemes.set_index("scheme_code")
    df = an.clusters.pca.join(an.clusters.labels)
    fig = go.Figure()
    for c in sorted(df["cluster"].unique()):
        g = df[df["cluster"] == c]
        color = palette[c] if c < len(palette) else NIFTY
        names = [info.at[i, "scheme_name"] for i in g.index]
        cats = [short_category(info.at[i, "scheme_category"]) for i in g.index]
        fig.add_trace(go.Scatter(
            x=g["pc1"], y=g["pc2"], mode="markers", name=f"Group {c + 1}",
            marker={"size": 10, "color": color, "symbol": SYMBOLS[c % len(SYMBOLS)],
                        "line": {"width": 1.5, "color": "rgba(255,255,255,0.9)" if not dark
                                  else "rgba(15,23,42,0.9)"}},
            customdata=list(zip(names, cats)),
            hovertemplate="<b>%{customdata[0]}</b><br>Category: %{customdata[1]}"
                          f"<extra>Group {c + 1}</extra>",
        ))
    _base_layout(fig, 460)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Behaviour pattern 1", showgrid=False, zeroline=False,
                     showticklabels=False)
    fig.update_yaxes(title="Behaviour pattern 2", showgrid=False, zeroline=False,
                     showticklabels=False)
    return fig
