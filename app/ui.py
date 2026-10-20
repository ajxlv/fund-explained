"""Shared Streamlit helpers: data loading (cached), CSS, banner, tiles, footer."""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from fundx.data import load_dataset, resolve_data_dir
from fundx.ml import analyse

DEMO_TEXT = ("Demo mode: synthetic data. Run scripts/fetch_navs.py on your laptop to load real "
             "funds.")
FOOTER_TEXT = ("Educational tool. Past performance does not indicate future returns. "
               "Not investment advice. Data: mfapi.in (AMFI NAVs).")


def _data_version() -> str:
    d, _ = resolve_data_dir()
    return "|".join(f"{p}:{p.stat().st_mtime_ns}" for p in sorted(d.glob("*.*")))


@st.cache_data(show_spinner="Crunching the numbers…")
def _load(version: str):
    ds = load_dataset()
    return ds, analyse(ds)


def load():
    """(Dataset, Analysis), computed once per data version and cached."""
    return _load(_data_version())


def inject_css() -> None:
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def demo_banner(is_demo: bool) -> None:
    if is_demo:
        st.markdown(f'<div class="fx-banner" role="alert">⚠️ {DEMO_TEXT}</div>',
                    unsafe_allow_html=True)


def footer(is_demo: bool) -> None:
    extra = " Currently showing synthetic demo data." if is_demo else ""
    st.markdown(f'<div class="fx-footer">{FOOTER_TEXT}{extra}</div>', unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def chip(text: str, cls: str = "") -> str:
    return f'<span class="fx-chip {cls}">{esc(text)}</span>'


def tile(label: str, value: str, note: str = "", delta: str = "", up: bool | None = None,
         how: str = "") -> str:
    """One stat tile. `how` is the one-line "how this is calculated" tooltip."""
    info = f' <span class="fx-i" title="{esc(how)}">ⓘ</span>' if how else ""
    d = ""
    if delta:
        tone = "" if up is None else ("fx-up" if up else "fx-down")
        arrow = "" if up is None else ("▲ " if up else "▼ ")
        d = f'<div class="fx-delta {tone}">{arrow}{esc(delta)}</div>'
    n = f'<div class="fx-note">{esc(note)}</div>' if note else ""
    return (f'<div class="fx-tile" title="{esc(how)}"><div class="fx-label">{esc(label)}{info}'
            f'</div><div class="fx-value">{esc(value)}</div>{d}{n}</div>')


def tiles(items: list[str], cls: str = "") -> None:
    st.markdown(f'<div class="fx-tiles {cls}">{"".join(items)}</div>', unsafe_allow_html=True)


def card(body_html: str, cls: str = "") -> None:
    st.markdown(f'<div class="fx-card {cls}">{body_html}</div>', unsafe_allow_html=True)


def how_calculated(lines: list[str], title: str = "How is this calculated?") -> None:
    with st.expander(title):
        st.markdown("\n".join(f"- {line}" for line in lines))


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit or no browser context
        return False
