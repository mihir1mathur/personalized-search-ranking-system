"""
metrics.py  --  Ranking-quality metrics for Week 1 retrieval baselines
======================================================================

This small module holds the three metrics we use to measure how good a
retrieval method is. They are written from scratch (with lots of comments)
so a beginner can see exactly how each number is computed.

VOCABULARY (read this first):
  - "retrieved_ids": the list of product_ids a method returned for a query,
                     already ordered best-first (rank 1, 2, 3, ...).
  - "qrels":         short for "query relevance judgments". A dictionary that
                     maps a product_id -> its known relevance_score for THIS
                     query. Example: {"B001": 3, "B002": 0}.
                     Any product NOT in this dictionary is unjudged, and we
                     treat unjudged products as irrelevant (relevance 0).
  - "k":             how many of the top results we look at (here, k = 10).
  - "rel_threshold": the smallest score that still counts as "relevant".
                     We use 1, so scores 1/2/3 are relevant and 0 is not.

All functions return None when a metric is undefined for a query (for example
Recall when the query has no known relevant products). The caller then simply
skips those queries when averaging.
"""

import math


def precision_at_k(retrieved_ids, qrels, k=10, rel_threshold=1):
    """
    Precision@K = (relevant items found in the top K) / K

    In plain words: of the K products we showed, what fraction were relevant?
    A precision of 0.3 means 3 out of the top 10 were relevant.
    """
    top_k = retrieved_ids[:k]
    # Count how many of the top-K products are relevant (score >= threshold).
    hits = sum(1 for pid in top_k if qrels.get(pid, 0) >= rel_threshold)
    return hits / k


def recall_at_k(retrieved_ids, qrels, k=10, rel_threshold=1):
    """
    Recall@K = (relevant items found in the top K) / (all known relevant items)

    In plain words: of all the products we KNOW are relevant for this query,
    what fraction did we manage to put in the top K?
    Returns None if the query has no known relevant products (recall undefined).
    """
    # Total number of products judged relevant for this query.
    total_relevant = sum(1 for score in qrels.values() if score >= rel_threshold)
    if total_relevant == 0:
        return None  # cannot compute recall with zero relevant items

    top_k = retrieved_ids[:k]
    hits = sum(1 for pid in top_k if qrels.get(pid, 0) >= rel_threshold)
    return hits / total_relevant


def _dcg(gains):
    """
    Discounted Cumulative Gain.

    We walk down the ranked list. Each item contributes its relevance "gain",
    but items further down are discounted (divided) by log2(position + 1).
    So a relevant item at rank 1 is worth more than the same item at rank 10.
    """
    total = 0.0
    for index, gain in enumerate(gains):
        rank = index + 1            # ranks start at 1, not 0
        total += gain / math.log2(rank + 1)
    return total


def ndcg_at_k(retrieved_ids, qrels, k=10):
    """
    NDCG@K = DCG@K / IDCG@K   (a value between 0 and 1)

    NDCG (Normalized Discounted Cumulative Gain) rewards putting the MOST
    relevant products at the very top. We use the graded relevance_score
    (0, 1, 2, 3) directly as the "gain", so a score-3 item counts more than a
    score-1 item.

      - DCG@K  = the score our ranking actually achieved.
      - IDCG@K = the score of the BEST POSSIBLE ranking (ideal order), using
                 the known judged relevances for this query.
    Dividing one by the other gives a fair 0-to-1 score. Returns None if the
    ideal score is 0 (the query has no positive relevance to rank).
    """
    # Gains for what we actually returned (unjudged -> 0).
    actual_gains = [qrels.get(pid, 0) for pid in retrieved_ids[:k]]
    dcg_value = _dcg(actual_gains)

    # The ideal ranking: sort ALL known relevances high-to-low, take the top K.
    ideal_gains = sorted(qrels.values(), reverse=True)[:k]
    idcg_value = _dcg(ideal_gains)

    if idcg_value == 0:
        return None  # no relevant items -> NDCG undefined
    return dcg_value / idcg_value
