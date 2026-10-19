"""Forgiving fund-name search (typos, partial words, any word order). Stdlib only."""
import difflib
import re

_NOISE = {"fund", "direct", "plan", "growth", "option", "the", "-"}


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9&]+", s.lower()) if t and t not in _NOISE]


def score(query: str, name: str) -> float:
    """0..1: how well `name` matches `query`. Every query word must roughly appear in the name."""
    if query.lower().strip() in name.lower():
        return 1.0
    q, n = _tokens(query), _tokens(name)
    if not q:
        return 0.0
    per_word = []
    for qt in q:
        best = max((1.0 if nt.startswith(qt) else difflib.SequenceMatcher(None, qt, nt).ratio()
                    for nt in n), default=0.0)
        per_word.append(best)
    return sum(per_word) / len(per_word) if min(per_word) >= 0.6 else 0.0


def fuzzy_search(query: str, names: list[str], limit: int = 50) -> list[str]:
    """Names ranked by match quality. An empty query returns the names unchanged."""
    if not query.strip():
        return list(names)
    scored = [(score(query, n), n) for n in names]
    hits = [(s, n) for s, n in scored if s > 0]
    hits.sort(key=lambda t: (-t[0], t[1]))
    return [n for _, n in hits[:limit]]
