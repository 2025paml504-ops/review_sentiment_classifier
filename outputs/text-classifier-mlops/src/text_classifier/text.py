from __future__ import annotations

import html
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize text while preserving useful intent and sentiment punctuation."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = EMAIL_RE.sub(" EMAIL ", text)
    text = URL_RE.sub(" URL ", text)
    text = "".join(ch for ch in text if ch.isprintable())
    return SPACE_RE.sub(" ", text).strip()


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", text.lower())
