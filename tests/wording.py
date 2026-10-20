"""The one place the forbidden phrases are spelled out: the app explains and never recommends."""
import re

BANNED = ("buy", "sell", "best fund", "you should")
BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BANNED) + r")\b", re.IGNORECASE)


def banned_phrases(text: str) -> list[str]:
    return BANNED_RE.findall(text)
