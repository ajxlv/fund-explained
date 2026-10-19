"""Repo-wide guard: no recommending language in code, UI text or docs (tests/ excluded)."""
from pathlib import Path

import pytest

from tests.wording import banned_phrases

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    p for pattern in ("fundx/*.py", "app/*.py", "app/*.css", "scripts/*.py", "docs/*.md", "*.md")
    for p in ROOT.glob(pattern)
)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_recommending_language(path):
    assert banned_phrases(path.read_text(encoding="utf-8")) == []


def test_guard_actually_catches_phrases():
    assert banned_phrases("You should BUY this; best fund ever") == ["You should", "BUY", "best fund"]
    assert banned_phrases("a bluechip seller-free sell-off") == ["sell"]
