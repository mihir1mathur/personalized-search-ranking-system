"""
ltr_features.py  --  Feature engineering for Learning-to-Rank (Week 5)
======================================================================

WHAT THIS DOES (beginner view)
------------------------------
Weeks 1-4 gave every (query, product) pair a handful of separate SIGNALS:

    BM25 score        -- keyword overlap          (Week 1)
    Embedding cosine  -- semantic similarity       (Week 2)
    Hybrid score      -- a fixed blend of the two  (Week 3)
    Cross-encoder      -- deep query x product read (Week 4)

Up to Week 4 we combined these signals with HAND-PICKED rules:
    - hybrid used a fixed weighted sum (alpha*BM25 + beta*Emb),
    - the re-ranker just sorted by the cross-encoder score alone.

Learning-to-Rank (Week 5) stops guessing the weights. Instead we turn each
(query, product) candidate into a small VECTOR OF FEATURES and let a gradient
boosted tree model LEARN, from the relevance labels, how to weigh and COMBINE
those features into one final ranking score. This module builds that feature
vector.

KEY IDEA: NO NEW HEAVY COMPUTATION.
Every feature here is either
    (a) a number we ALREADY computed in Weeks 1-4 (BM25 / embedding / hybrid /
        cross-encoder scores, the stage-1 candidate rank), or
    (b) a trivial, near-instant text statistic (token counts, overlaps, brand
        match) computed from data we already have in memory.
So generating features for all candidates is essentially free -- we do NOT
re-index, re-encode, or re-run the cross-encoder here.

The feature list is deliberately SMALL and INTERPRETABLE. In production the
single most important property of an LTR feature set is that every feature is
cheap to compute at serving time and easy to explain when a ranking looks
wrong. That matters more than squeezing out the last 0.1% of NDCG.
"""

import numpy as np

# Reuse the EXACT tokenizer Week 1's BM25 used, so "token overlap" here means
# the same thing it meant for keyword retrieval (no new tokenization rules).
try:
    from bm25_retriever import simple_tokenize
except ImportError:  # allow running from the project root too
    from src.bm25_retriever import simple_tokenize  # type: ignore


# ---------------------------------------------------------------------------
# The ordered list of feature names. The ORDER here is the order of the columns
# in the feature matrix, and it is what the model's feature-importance report
# refers to. Keep this list and build_feature_row() in perfect sync.
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "ce_score",             # cross-encoder relevance (Week 4)  -- expected strongest
    "ce_rank",              # 1-based rank by cross-encoder score (clean ordinal)
    "hybrid_score",         # alpha*bm25_norm + beta*emb_norm   (Week 3)
    "bm25_norm",            # normalized BM25 keyword score      (Week 1)
    "emb_norm",             # normalized embedding cosine        (Week 2)
    "bm25_emb_gap",         # bm25_norm - emb_norm (do the two signals agree?)
    "hybrid_rank",          # 1-based position in the stage-1 candidate list
    "query_len",            # number of query tokens
    "title_len",            # number of tokens in the product title
    "text_len_log",         # log1p(#tokens in full product text) -- length prior
    "title_overlap_count",  # how many distinct query tokens appear in the title
    "title_overlap_ratio",  # title_overlap_count / query_len (query coverage)
    "title_exact_contains", # 1 if the whole query string is a substring of title
    "brand_match",          # 1 if any query token matches the product brand
    "color_match",          # 1 if any query token matches the product color
]

NUM_FEATURES = len(FEATURE_NAMES)


def _token_set(text):
    """Lower-cased set of tokens for a piece of text (safe on None/NaN)."""
    if text is None:
        return set()
    text = str(text)
    if not text or text.lower() == "nan":
        return set()
    return set(simple_tokenize(text))


def build_feature_row(query_tokens, query_str,
                      bm25_norm, emb_norm, hybrid_score, ce_score, ce_rank,
                      hybrid_rank, title, brand, color, full_text):
    """
    Build ONE feature vector for a single (query, candidate product) pair.

    All the score inputs (bm25_norm, emb_norm, hybrid_score, ce_score, ce_rank)
    are REUSED from earlier weeks -- nothing is recomputed here. ce_rank is the
    candidate's 1-based position when the query's candidates are sorted by the
    cross-encoder score; it hands the model the cross-encoder's clean ORDERING
    (not just the raw logit), which the trees can reproduce exactly and then
    refine with the lexical features. The rest are tiny text statistics.

    Returns a Python list of floats, in FEATURE_NAMES order.
    """
    query_token_set = set(query_tokens)
    query_len = max(len(query_tokens), 1)  # avoid divide-by-zero in the ratio

    title_tokens = simple_tokenize(str(title)) if title is not None else []
    title_token_set = set(title_tokens)
    brand_tokens = _token_set(brand)
    color_tokens = _token_set(color)

    # How many DISTINCT query words show up in the product title.
    overlap_count = len(query_token_set & title_token_set)
    overlap_ratio = overlap_count / query_len

    # Whole-query substring match (a strong exact-intent signal for short queries).
    title_lower = str(title).lower() if title is not None else ""
    exact_contains = 1.0 if (query_str and query_str.lower() in title_lower) else 0.0

    # Length of the full searchable text, log-compressed so a 2000-word
    # description does not dwarf a 5-word title on a linear scale.
    full_tokens = simple_tokenize(str(full_text)) if full_text is not None else []
    text_len_log = float(np.log1p(len(full_tokens)))

    brand_match = 1.0 if (query_token_set & brand_tokens) else 0.0
    color_match = 1.0 if (query_token_set & color_tokens) else 0.0

    return [
        float(ce_score),
        float(ce_rank),
        float(hybrid_score),
        float(bm25_norm),
        float(emb_norm),
        float(bm25_norm) - float(emb_norm),
        float(hybrid_rank),
        float(query_len),
        float(len(title_tokens)),
        text_len_log,
        float(overlap_count),
        float(overlap_ratio),
        exact_contains,
        brand_match,
        color_match,
    ]


def build_feature_matrix(per_query_rows):
    """
    Stack many feature rows into a single (N x NUM_FEATURES) float32 matrix.

    per_query_rows : a flat list of feature-row lists (already built by
                     build_feature_row), across all queries and candidates.

    float32 keeps the matrix small (LightGBM bins features internally anyway).
    """
    if not per_query_rows:
        return np.zeros((0, NUM_FEATURES), dtype="float32")
    return np.asarray(per_query_rows, dtype="float32")


# ---------------------------------------------------------------------------
# Tiny self-test so this file can be run on its own.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    q = "iphone charger"
    qtok = simple_tokenize(q)
    row = build_feature_row(
        query_tokens=qtok, query_str=q,
        bm25_norm=0.9, emb_norm=0.7, hybrid_score=0.82, ce_score=8.1, ce_rank=1,
        hybrid_rank=1,
        title="Apple Lightning to USB Cable for iPhone charging cord",
        brand="Apple", color=None,
        full_text="Apple Lightning to USB Cable for iPhone charging cord",
    )
    print("Feature names :", FEATURE_NAMES)
    print("Example vector:", [round(v, 3) for v in row])
    print(f"NUM_FEATURES = {NUM_FEATURES}")
