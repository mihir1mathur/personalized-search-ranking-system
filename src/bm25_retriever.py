"""
bm25_retriever.py  --  BM25 retrieval baseline (Week 1)
=======================================================

WHAT THIS DOES (beginner view):
BM25 is a classic keyword-matching ranking formula used by real search engines.
Like TF-IDF, it rewards documents that share words with the query. But BM25
adds two smart ideas that usually make it better:
  1. Term-frequency SATURATION: seeing a word 10 times is not 10x better than
     seeing it once. Extra repeats help less and less.
  2. Document-length NORMALIZATION: long documents naturally contain more words,
     so BM25 gently penalizes very long documents so they do not win unfairly.

HOW WE USE IT:
  1. "Tokenize" every product_text (split it into a list of lowercase words).
  2. Build a BM25 index over those token lists using the `rank_bm25` library.
  3. For a query, tokenize it the same way and ask BM25 to score every product.
  4. Return the highest-scoring products, best first.

This file can be imported (use the BM25Retriever class) or run directly for a
tiny self-test (`python src/bm25_retriever.py`).
"""

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError:
    # Friendly message if a required library is missing.
    raise ImportError(
        "BM25 needs the 'rank_bm25' library (and numpy).\n"
        "Please install them with:\n"
        "    pip install rank_bm25 numpy"
    )


def simple_tokenize(text):
    """
    Turn text into a list of lowercase word "tokens".

    We keep this deliberately simple (lowercase + split on spaces) so beginners
    can follow it, and so the QUERY and the PRODUCTS are tokenized the exact
    same way (very important for fair matching).
    """
    if text is None:
        return []
    return str(text).lower().split()


class BM25Retriever:
    """A simple BM25 retriever built on the rank_bm25 library."""

    def __init__(self, tokenizer=simple_tokenize):
        self.tokenizer = tokenizer
        self.bm25 = None         # the BM25 index
        self.product_ids = None  # product_id matching each document

    def fit(self, texts, product_ids):
        """
        Build the BM25 index from the product corpus.
        texts        : list of product_text strings.
        product_ids  : list of product_id strings (same order as texts).
        """
        if len(texts) != len(product_ids):
            raise ValueError("texts and product_ids must have the same length.")
        # Tokenize every product into a list of words.
        tokenized_corpus = [self.tokenizer(t) for t in texts]
        # BM25Okapi precomputes the statistics it needs to score quickly later.
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.product_ids = np.asarray(product_ids)
        return self

    def _top_k_from_scores(self, scores, top_k):
        """Given a BM25 score for every product, return the top_k (id, score)."""
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(str(self.product_ids[i]), float(scores[i])) for i in top_idx]

    def retrieve(self, query, top_k=10):
        """
        Retrieve the top_k products for a single query.
        Returns a list of (product_id, score, rank), best first (rank starts at 1).
        """
        if self.bm25 is None:
            raise RuntimeError("Call fit(...) before retrieve(...).")
        query_tokens = self.tokenizer(query)
        # get_scores returns one BM25 score per product in the corpus.
        scores = np.asarray(self.bm25.get_scores(query_tokens))
        ranked = self._top_k_from_scores(scores, top_k)
        return [(pid, score, rank) for rank, (pid, score) in enumerate(ranked, start=1)]

    def retrieve_batch(self, queries, top_k=10):
        """
        Retrieve for MANY queries (just calls retrieve() in a loop).
        Returns a list (one per query) of [(product_id, score, rank), ...].
        """
        return [self.retrieve(q, top_k=top_k) for q in queries]


# ---------------------------------------------------------------------------
# Tiny self-test so you can run this file on its own and see it work.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts = [
        "red running shoes for men lightweight",
        "blue running shoes women breathable",
        "stainless steel kitchen blender",
        "wireless bluetooth headphones noise cancelling",
    ]
    demo_ids = ["P1", "P2", "P3", "P4"]

    retriever = BM25Retriever().fit(demo_texts, demo_ids)
    print("Query: 'running shoes'")
    for pid, score, rank in retriever.retrieve("running shoes", top_k=3):
        print(f"  rank {rank}: {pid}  score={score:.3f}")
    print("\nBM25Retriever self-test completed.")
