"""Diagnose text-cleaning issues in the interim features (added 9 Aug - Ankita).

This is the *detection* side of the cleaning fixes recorded in
[Decisions §10-11](../docs/design/decisions.md): each check below reproduces the
symptom that motivated a fix, so the issue can be re-found in data instead of
being taken on trust. It is a report, never a gate — it always exits 0 and never
writes anything.

Run from the repo root:

    python -m validation.diagnose_cleaning              # full interim CSV
    python -m validation.diagnose_cleaning --limit 50000
"""

import argparse
import logging
import re
from collections import Counter

import pandas as pd

from features.build_features import (
    INTERIM_PATH,
    NEGATION_SKIP_WORDS,
    NEGATORS,
    _CURLY_APOSTROPHE,
    _SPLIT_NT_STEMS,
    build_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("diagnose_cleaning")

# How many concrete offending examples to print per check.
EXAMPLES = 5

# A negator merged onto the next token, e.g. "not_good" / "no_one".
_MERGED_NEGATION_RE = re.compile(r"\b(" + "|".join(sorted(NEGATORS)) + r")_(\w+)")
# A contraction that shattered into "<stem> t" and survived cleaning.
_SPLIT_NT_LEFTOVER_RE = re.compile(r"\b(?:" + "|".join(_SPLIT_NT_STEMS) + r") t\b")
# A contraction written with spaces around the apostrophe in the raw text.
_SPACED_CONTRACTION_RE = re.compile(r"\w\s+['\u2019]\s*\w|\w\s*['\u2019]\s+\w")
# Any apostrophe form still present after cleaning would mean the strip failed.
_ANY_APOSTROPHE_RE = re.compile(r"['\u2019]")


def _report(name: str, count: int, total: int, examples: list[str], fix: str) -> None:
    """Log one check: headline count, share of rows, examples, and the fix it drove."""
    if not count:
        logger.info("[OK]    %s: none found", name)
        return
    logger.warning(
        "[ISSUE] %s: %d hit(s) over %d rows (%.3f%%)", name, count, total, 100 * count / total
    )
    for ex in examples[:EXAMPLES]:
        logger.warning("           e.g. %s", ex)
    logger.warning("           fix: %s", fix)


def check_low_value_negations(df: pd.DataFrame) -> None:
    """Negators merged onto function words ('not_the', 'no_one').

    These are frequent but carry no sentiment, so they burn slots in the
    20,000-feature TF-IDF budget. Detecting them is what motivated
    NEGATION_SKIP_WORDS in features/build_features.py.
    """
    counts: Counter[str] = Counter()
    for text in df["clean_review"]:
        for negator, following in _MERGED_NEGATION_RE.findall(text):
            if following in NEGATION_SKIP_WORDS:
                counts[f"{negator}_{following}"] += 1
    examples = [f"{tok} x{n}" for tok, n in counts.most_common(EXAMPLES)]
    _report(
        "negator merged with a function word",
        sum(counts.values()),
        len(df),
        examples,
        "skip the merge when the next token is in NEGATION_SKIP_WORDS",
    )


def check_split_contractions(df: pd.DataFrame) -> None:
    """Contractions that shattered into '<stem> t' ('if you don t mind').

    The non-letter strip splits an apostrophe-less 'don t' into two junk tokens
    and the negation is lost. This is what _SPLIT_NT_RE repairs.
    """
    mask = df["clean_review"].str.contains(_SPLIT_NT_LEFTOVER_RE, regex=True)
    examples = df.loc[mask, "clean_review"].str.slice(0, 80).tolist()
    _report(
        "contraction split into '<stem> t'",
        int(mask.sum()),
        len(df),
        examples,
        "rejoin the explicit stems via _SPLIT_NT_RE after the non-letter strip",
    )


def check_spaced_apostrophes(df: pd.DataFrame) -> None:
    """Raw reviews spelling a contraction as \"don ' t\".

    The spacing hides the contraction from _CONTRACTIONS_RE, so it reaches the
    non-letter strip intact. This is what _SPACED_APOSTROPHE_RE collapses.
    """
    mask = df["full_review"].str.contains(_SPACED_CONTRACTION_RE, regex=True)
    examples = df.loc[mask, "full_review"].str.slice(0, 80).tolist()
    _report(
        "apostrophe surrounded by spaces in the raw text",
        int(mask.sum()),
        len(df),
        examples,
        "collapse the spacing with _SPACED_APOSTROPHE_RE before expanding contractions",
    )


def check_curly_apostrophes(df: pd.DataFrame) -> None:
    """Curly apostrophes (U+2019) or any apostrophe surviving into clean_review.

    Either means the normalization/strip order regressed.
    """
    raw_curly = int(df["full_review"].str.contains(_CURLY_APOSTROPHE, regex=False).sum())
    logger.info("raw reviews containing a curly apostrophe (U+2019): %d", raw_curly)

    mask = df["clean_review"].str.contains(_ANY_APOSTROPHE_RE, regex=True)
    examples = df.loc[mask, "clean_review"].str.slice(0, 80).tolist()
    _report(
        "apostrophe surviving into clean_review",
        int(mask.sum()),
        len(df),
        examples,
        "normalize U+2019 before contraction expansion; the non-letter strip removes the rest",
    )


def check_empty_documents(df: pd.DataFrame) -> None:
    """Rows that end up as empty documents in the feature store / TF-IDF split.

    Two sources: the raw text was blank, or it consisted only of placeholder
    phrases / non-letters and cleaned down to nothing. Both are dropped in
    build_features().
    """
    empty_raw = int((df["full_review"].str.strip() == "").sum())
    mask = df["clean_review"].str.strip() == ""
    examples = [
        f"row {i}: full_review={df.at[i, 'full_review'][:60]!r}" for i in df.index[mask].tolist()
    ]
    _report(
        "empty clean_review (empty document)",
        int(mask.sum()),
        len(df),
        examples,
        "drop rows whose full_review or clean_review is empty in build_features()",
    )
    logger.info("rows with an empty full_review: %d", empty_raw)


def check_duplicates(df: pd.DataFrame) -> None:
    """Duplicate reviews that survived the de-duplication step."""
    mask = df.duplicated(subset=["full_review", "Reviewer_Score"], keep="first")
    examples = df.loc[mask, "full_review"].str.slice(0, 80).tolist()
    _report(
        "duplicate [full_review, Reviewer_Score]",
        int(mask.sum()),
        len(df),
        examples,
        "drop_duplicates on [full_review, Reviewer_Score] right after combine_reviews()",
    )


CHECKS = [
    check_low_value_negations,
    check_split_contractions,
    check_spaced_apostrophes,
    check_curly_apostrophes,
    check_empty_documents,
    check_duplicates,
]


def load_features(limit: int | None = None) -> pd.DataFrame:
    """Read the interim CSV, rebuilding it from raw if it is missing."""
    if INTERIM_PATH.exists():
        df = pd.read_csv(INTERIM_PATH, nrows=limit)
    else:
        logger.info("%s missing - rebuilding from raw", INTERIM_PATH)
        df = build_features()
        if limit:
            df = df.head(limit)
    for col in ("full_review", "clean_review"):
        df[col] = df[col].fillna("").astype(str)
    return df


def diagnose(df: pd.DataFrame) -> None:
    """Run every check against the given features frame."""
    logger.info("Diagnosing %d rows", len(df))
    for check in CHECKS:
        check(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only read the first N rows")
    args = parser.parse_args()
    diagnose(load_features(args.limit))
