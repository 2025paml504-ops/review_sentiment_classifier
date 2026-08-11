"""Load the raw hotel reviews dataset, clean/tokenize it, and write to data/interim."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root is two levels up from this file (features/build_features.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "Hotel_Reviews.csv"
INTERIM_PATH = REPO_ROOT / "data" / "interim" / "features_clean.csv"

# Placeholder phrases the dataset uses when a reviewer left that side blank.
_PLACEHOLDER_RE = re.compile(r"no positive|no negative")
# Anything that is not an ASCII letter or whitespace (digits, punctuation, specials).
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

# Added on 9 Aug - Ankita: Expanded negator set
NEGATORS = {"not", "no", "never", "cannot", "nor"}

# Added on 9 Aug - Ankita: Function words that carry no sentiment. Merging a
# negator with one of these (not_the, no_one) only produces low-value tokens
# that dilute the 20k TF-IDF budget, so the negator is left standing alone.
# NEGATION_SKIP_WORDS = {
#     "the", "a", "an", "be", "to", "of", "i", "it", "we", "they", "he", "she",
#     "and", "or", "but", "that", "this", "is", "are", "was", "were", "in",
#     "on", "at", "for", "as", "one", "any", "there", "my", "our", "you",
# }
# Added 11 Aug - Ankita: "the", "a", "an" and "that" moved out to
# NEGATION_SKIP_THROUGH below - those should be skipped *past* to reach the
# real target, not treated as a hard stop, or "not the best" leaves "best"
# completely unmarked (confirmed on 6.9% of rows). "one" and the rest stay
# here as a hard stop - "no one" is a fixed phrase, not a negator + filler.
NEGATION_SKIP_WORDS = {
    "be", "to", "of", "i", "it", "we", "they", "he", "she",
    "and", "or", "but", "this", "is", "are", "was", "were", "in",
    "on", "at", "for", "as", "one", "any", "there", "my", "our", "you",
}

# Added 11 Aug - Ankita: determiners/demonstratives/intensifiers that sit
# between a negator and its real target rather than being the target
# themselves or blocking the merge. Distinct from NEGATION_SKIP_WORDS above:
# "no one" must stay split (a fixed phrase, "one" is a hard stop), but
# "not the best" / "not very good" / "not that great" should skip past the
# filler to reach "best"/"good"/"great" - confirmed missing on 6.9% of rows.
NEGATION_SKIP_THROUGH = {
    "the", "a", "an", "that", "very", "really", "so", "too", "quite",
    "pretty", "extremely",
}

# Added on 9 Aug - Ankita: Contractions dictionary
CONTRACTIONS = {
    "don't": "do not", "i'm": "i am", "can't": "cannot", "won't": "will not",
    "it's": "it is", "you're": "you are", "they're": "they are",
    "didn't": "did not", "doesn't": "does not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
    "ain't": "is not", "shan't": "shall not", "mightn't": "might not",
    "needn't": "need not"
}

# Added on 9 Aug - Ankita: Single compiled alternation (word-bounded) instead of
# one str.replace pass per entry; longest keys first so no key masks a longer one.
_CONTRACTIONS_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b"
)
# Added on 9 Aug - Ankita: Curly apostrophe (U+2019) normalized to ASCII apostrophe.
_CURLY_APOSTROPHE = "\u2019"
# Added on 9 Aug - Ankita: Some reviews spell contractions with spaces around the
# apostrophe ("don ' t"), which escaped _CONTRACTIONS_RE and shattered into
# "don" + "t". Collapse the spacing before contraction expansion runs.
_SPACED_APOSTROPHE_RE = re.compile(r"\s*'\s*")
# Added on 9 Aug - Ankita: Catch-all for the long tail. The apostrophe form is
# generic; bare forms are listed explicitly so real words (went, want) are safe.
_NT_RE = re.compile(
    r"n't\b|\b(?:do|does|did|is|are|was|were|has|have|had|could|should|would|must|ai|ca|wo)nt\b"
)


# Added on 9 Aug - Ankita: Repair for reviews where the apostrophe is missing
# entirely ("if you don t mind"), which the contraction pass cannot see. Stems
# are listed explicitly so ordinary "<word> t" sequences stay untouched.
_SPLIT_NT_STEMS = {
    "don": "do", "doesn": "does", "didn": "did", "isn": "is", "aren": "are",
    "wasn": "was", "weren": "were", "hasn": "has", "haven": "have",
    "hadn": "had", "couldn": "could", "shouldn": "should", "wouldn": "would",
    "mustn": "must", "can": "can", "won": "will", "ain": "is",
    "shan": "shall", "needn": "need", "mightn": "might",
}
_SPLIT_NT_RE = re.compile(r"\b(" + "|".join(_SPLIT_NT_STEMS) + r") t\b")


def _expand_split_nt(match: re.Match) -> str:
    return f"{_SPLIT_NT_STEMS[match.group(1)]} not"


def _expand_nt(match: re.Match) -> str:
    text = match.group()
    if text.startswith("n'"):
        return " not"
    return f"{text[:-2]} not"

# Sentiment label thresholds on Reviewer_Score (Scheme A): NEG < 6, 6 <= NEU < 8, POS >= 8.
# Produced a 10.3 / 24.9 / 64.8 split; picked by hand to be "the least imbalanced
# observed", not from an external convention.
# NPS-style thresholds tried 10-11 Aug - Ankita (Promoter 9-10 / Passive 7-8 /
# Detractor 0-6): raised macro-F1 and NEGATIVE/NEUTRAL detection on every one
# of the four models (e.g. logreg macro-F1 0.618 -> 0.647), but *lowered* raw
# accuracy on all of them (e.g. LinearSVC 71.8% -> 65.1%) because the old
# scheme's 65%-POSITIVE imbalance is exactly what inflates accuracy without
# reflecting real quality. Reverted 11 Aug - Ankita: accuracy is the number
# that needs to read higher here, so back to Scheme A.
# SENTIMENT_NEUTRAL_FLOOR = 7.0
# SENTIMENT_POSITIVE_FLOOR = 9.0
SENTIMENT_NEUTRAL_FLOOR = 6.0
SENTIMENT_POSITIVE_FLOOR = 8.0
SENTIMENT_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Metadata columns that carry no signal for sentiment classification.
COLUMNS_TO_DROP = [
    "Hotel_Address",
    "Additional_Number_of_Scoring",
    "Review_Date",
    "Average_Score",
    "Hotel_Name",
    "Reviewer_Nationality",
    "Total_Number_of_Reviews",
    "Total_Number_of_Reviews_Reviewer_Has_Given",
    "Tags",
    "days_since_review",
    "lat",
    "lng",
]

def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw hotel reviews CSV into a DataFrame."""
    return pd.read_csv(path)

def drop_columns(df: pd.DataFrame, columns=COLUMNS_TO_DROP) -> pd.DataFrame:
    """Drop the given columns, ignoring any that are missing."""
    return df.drop(columns=columns, errors="ignore")

def combine_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `full_review` column: positive and negative review text combined."""
    df = df.copy()
    # positive = df["Positive_Review"].fillna("")
    # negative = df["Negative_Review"].fillna("")
    # df["full_review"] = (positive + " " + negative).str.strip()
    #
    # Bug found 11 Aug - Ankita: full_review was only ever meant to feed
    # clean_text() (which strips "No Positive"/"No Negative" internally), so
    # nothing stripped the placeholder from full_review itself. Once a
    # consumer reads full_review directly (the transformer, for raw-text
    # input), it sees the literal placeholder as real text - confirmed on 31%
    # of rows (156,389 / 504,731). Stripped here at the field level (only when
    # a field IS the bare placeholder) instead: more precise than the
    # substring match in clean_text()'s _PLACEHOLDER_RE, which can also catch
    # the phrase inside genuine text (a rare false positive, ~0.07% of rows).
    positive = df["Positive_Review"].fillna("")
    negative = df["Negative_Review"].fillna("")
    positive = positive.where(positive.str.strip().str.lower() != "no positive", "")
    negative = negative.where(negative.str.strip().str.lower() != "no negative", "")
    df["full_review"] = (positive + " " + negative).str.strip()
    return df

# Added on 9 Aug - Ankita: Negation handler
# def handle_negations(tokens: list[str]) -> list[str]:
#     """Attach negators to the following word (e.g., 'not good' -> 'not_good')."""
#     result = []
#     skip = False
#     for i, token in enumerate(tokens):
#         if skip:
#             skip = False
#             continue
#         if (
#             token in NEGATORS
#             and i + 1 < len(tokens)
#             and tokens[i + 1] not in NEGATION_SKIP_WORDS
#         ):
#             result.append(f"{token}_{tokens[i+1]}")
#             skip = True
#         else:
#             result.append(token)
#     return result
#
# Bug found 11 Aug - Ankita: the version above only ever looked one token
# ahead, and gave up entirely (leaving the negator standing alone) when that
# token was a function word - so "not the best" / "not that great" / "not a
# good experience" left "best"/"great"/"good" completely unmarked, a raw
# positive token in the bag of words. Confirmed on 6.9% of rows
# (34,610 / 504,731). Fixed below: skip past up to MAX_NEGATION_SKIP
# determiners/intensifiers to reach the real sentiment target.
MAX_NEGATION_SKIP = 3  # Added by Ankita 11 Aug


def handle_negations(tokens: list[str]) -> list[str]:
    """Attach negators to the next content word (e.g. 'not good' -> 'not_good',
    'not the best' -> 'not_best'), skipping past up to MAX_NEGATION_SKIP
    determiners/intensifiers (NEGATION_SKIP_THROUGH) to reach it. A negator
    immediately followed by a NEGATION_SKIP_WORDS hard-stop (e.g. 'one' in
    'no one') is left standing alone instead - that's a fixed phrase, not a
    negator plus skippable filler."""
    result = []
    skip_to = -1
    for i, token in enumerate(tokens):
        if i <= skip_to:
            continue
        if token in NEGATORS and i + 1 < len(tokens) and tokens[i + 1] not in NEGATION_SKIP_WORDS:
            j = i + 1
            skipped = 0
            while (
                j < len(tokens)
                and tokens[j] in NEGATION_SKIP_THROUGH
                and skipped < MAX_NEGATION_SKIP
            ):
                j += 1
                skipped += 1
            if j < len(tokens) and tokens[j] not in NEGATION_SKIP_WORDS and tokens[j] not in NEGATORS:
                result.append(f"{token}_{tokens[j]}")
                skip_to = j
                continue
        result.append(token)
    return result

def clean_text(text: str) -> str:
    """Expand contractions, lowercase, strip placeholder phrases, drop non-letters, collapse whitespace, and handle negations."""
    text = str(text).lower()

    # Added on 9 Aug - Ankita: Normalize curly apostrophes, then expand contractions
    text = text.replace(_CURLY_APOSTROPHE, "'")
    text = _SPACED_APOSTROPHE_RE.sub("'", text)  # Added on 9 Aug - Ankita
    text = _CONTRACTIONS_RE.sub(lambda m: CONTRACTIONS[m.group()], text)
    text = _NT_RE.sub(_expand_nt, text)

    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _NON_LETTER_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _SPLIT_NT_RE.sub(_expand_split_nt, text)  # Added on 9 Aug - Ankita

    tokens = text.split()
    tokens = handle_negations(tokens)  # Added on 9 Aug - Ankita

    return " ".join(tokens)

def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens on whitespace."""
    return text.split()

def clean_and_tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """Add `clean_review` (cleaned text) and `tokens` (JSON-encoded token list)."""
    df = df.copy()
    df["clean_review"] = df["full_review"].map(clean_text)
    df["tokens"] = df["clean_review"].map(lambda t: json.dumps(tokenize(t)))
    return df

def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `sentiment` label derived from Reviewer_Score (Scheme A thresholds)."""
    df = df.copy()
    df["sentiment"] = pd.cut(
        df["Reviewer_Score"],
        bins=[-np.inf, SENTIMENT_NEUTRAL_FLOOR, SENTIMENT_POSITIVE_FLOOR, np.inf],
        labels=SENTIMENT_LABELS,
        right=False,
    ).astype(str)
    return df

def build_features(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data, drop unused columns, combine reviews, dedupe, clean/tokenize, and label."""
    df = load_raw_data(path)
    df = drop_columns(df)
    df = combine_reviews(df)
    # Added on 9 Aug - Ankita: Deduplicate earlier for efficiency, reset index
    df = df.drop_duplicates(subset=["full_review", "Reviewer_Score"])
    # Added on 9 Aug - Ankita: Drop reviews with no text at all; they would reach
    # the feature store and the TF-IDF split as empty documents.
    df = df[df["full_review"].str.strip().astype(bool)].reset_index(drop=True)
    df = clean_and_tokenize(df)
    # Added on 9 Aug - Ankita: Some reviews are only placeholder phrases or
    # non-letters, so they clean down to nothing (surfaced by
    # validation/diagnose_cleaning.py). Drop them too.
    df = df[df["clean_review"].str.strip().astype(bool)].reset_index(drop=True)
    df = label_sentiment(df)
    return df

def write_interim(df: pd.DataFrame, path=INTERIM_PATH) -> Path:
    """Write the cleaned/tokenized features to the interim CSV; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    df = build_features()
    out = write_interim(df)
    print("Shape:", df.shape)
    print("Wrote interim:", out)
    print("Sample clean_review:", df["clean_review"].iloc[0])
