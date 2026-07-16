"""
cross_encoder_reranker.py  --  Cross-Encoder re-ranking (Week 4)
================================================================

WHAT THIS DOES (beginner view)
------------------------------
Weeks 1-3 built RETRIEVERS. A retriever's job is to look at ~48,000 products
very fast and pull out a short list of, say, the 50 most promising ones for a
query. They are fast because they compare CHEAP, PRECOMPUTED representations:
  - BM25         : word-overlap counts.
  - Embeddings   : one vector per product (computed once), compared by cosine.
  - Hybrid       : a weighted mix of the two.

The catch: to stay fast, the query and the product are turned into numbers
SEPARATELY and only compared at the very end. The model never reads the query
and the product TOGETHER, so it misses fine-grained clues ("is this charger
actually for an iPhone, or just near the word 'iphone'?").

A CROSS-ENCODER fixes exactly that. It is a transformer that takes the query
and ONE product text GLUED TOGETHER as a single input:

        [CLS] iphone charger [SEP] Apple Lightning Cable for iPhone [SEP]

and reads them with full attention between every query word and every product
word. It then outputs ONE number: how relevant this product is to this query.

        ("iphone charger", "Apple Lightning Cable")  ->  8.72
        ("iphone charger", "Samsung TV remote")      -> -7.10

This is far more ACCURATE than the bi-encoder retrievers, but also far more
EXPENSIVE: it must run the transformer once PER (query, product) pair, so you
could never run it over all 48k products for every query. That is why we use it
as a SECOND STAGE -- a "re-ranker":

        Hybrid retrieval  ->  Top 50 candidates  ->  Cross-Encoder  ->  Top 10

The retriever throws away the obvious junk cheaply; the cross-encoder carefully
re-orders the small survivor list. This two-stage "retrieve then re-rank" design
is exactly what real search engines (Amazon, Google, Bing) use.

MODEL WE USE
------------
  cross-encoder/ms-marco-MiniLM-L-6-v2  -- a small, fast, very popular
  cross-encoder trained on the MS MARCO search-relevance dataset. It outputs a
  single relevance score per (query, document) pair. Higher = more relevant.
  (The raw score is an unbounded "logit"; only the ORDER of scores matters for
  ranking, so we never need to interpret the absolute value.)

This file exposes a CrossEncoderReranker class. It does NOT retrieve anything by
itself -- you give it a query plus a candidate list (from the Week 3 hybrid
retriever), and it returns those same candidates re-ordered, best first.
"""

try:
    import numpy as np
    from sentence_transformers import CrossEncoder
except ImportError:
    # Friendly message if a required library is missing.
    raise ImportError(
        "Cross-encoder re-ranking needs 'sentence-transformers' (and numpy).\n"
        "Please install them with:\n"
        "    pip install sentence-transformers numpy"
    )


# The model name is kept as a constant so it is easy to find and change.
DEFAULT_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-rank a short candidate list using a cross-encoder relevance model.

    Typical use (one query):

        reranker = CrossEncoderReranker()
        candidates = [("P1", "Apple Lightning Cable"), ("P2", "Samsung remote")]
        ranked = reranker.rerank("iphone charger", candidates, top_k=10)
        # ranked = [("P1", 8.72, 1), ("P2", -7.10, 2)]

    The output format -- a list of (product_id, score, rank) tuples, best first,
    rank starting at 1 -- is IDENTICAL to the Week 1/2/3 retrievers, so the
    evaluation code can score it with the exact same metric functions.
    """

    def __init__(self, model_name=DEFAULT_CROSS_ENCODER_NAME, batch_size=32):
        """
        model_name : which cross-encoder to load.
        batch_size : how many (query, product) pairs to score at once. Bigger is
                     faster but uses more memory. 32 is a safe laptop/CPU default.
        """
        print(f"      Loading cross-encoder model: {model_name}")
        # This downloads the model the first time, then caches it locally.
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    # ----------------------------------------------------------------------
    # Score a batch of (query, product_text) pairs -> raw relevance numbers
    # ----------------------------------------------------------------------
    def score_pairs(self, query_product_pairs, show_progress=False):
        """
        Score a list of [query_text, product_text] pairs.

        Returns a 1D numpy array of relevance scores (one per pair). Higher means
        more relevant. The cross-encoder reads each pair TOGETHER, so the score
        reflects true query-product interaction, not two separate embeddings.
        """
        if len(query_product_pairs) == 0:
            return np.zeros(0, dtype="float64")
        scores = self.model.predict(
            query_product_pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype="float64")

    # ----------------------------------------------------------------------
    # Re-rank ONE query's candidate list
    # ----------------------------------------------------------------------
    def rerank(self, query, candidates, top_k=10):
        """
        Re-rank the candidate products for a SINGLE query.

        query      : the search query string.
        candidates : list of (product_id, product_text) tuples -- usually the
                     Top-N from the hybrid retriever.
        top_k      : how many re-ranked results to return.

        Returns a list of (product_id, cross_encoder_score, rank), best first.
        """
        if len(candidates) == 0:
            return []

        # Build the [query, product_text] pairs the model expects.
        candidate_ids = [product_id for (product_id, _text) in candidates]
        pairs = [[query, product_text] for (_pid, product_text) in candidates]

        # One transformer pass per pair (batched internally) -> one score each.
        scores = self.score_pairs(pairs)

        # Sort candidates by score, highest (most relevant) first.
        order = np.argsort(-scores)[:top_k]
        ranked = [
            (str(candidate_ids[pos]), float(scores[pos]), rank)
            for rank, pos in enumerate(order, start=1)
        ]
        return ranked

    # ----------------------------------------------------------------------
    # Re-rank MANY queries (each with its own candidate list)
    # ----------------------------------------------------------------------
    def rerank_batch(self, queries, candidates_per_query, top_k=10,
                     progress_every=200):
        """
        Re-rank candidate lists for MANY queries efficiently.

        queries              : list of query strings.
        candidates_per_query : list aligned with queries; each item is that
                               query's list of (product_id, product_text) tuples.
        top_k                : how many results to keep per query.
        progress_every       : print a progress line every N queries.

        We FLATTEN every (query, product) pair across all queries into one big
        list, score them in batches with a single model call (much faster than
        one call per query), then split the scores back out per query and sort.

        Returns a list (one per query) of [(product_id, score, rank), ...].
        """
        total_queries = len(queries)

        # 1) Flatten all pairs into one list, remembering which query each
        #    pair belongs to and how many candidates each query has.
        all_pairs = []
        candidate_ids_per_query = []
        counts = []
        for query, candidates in zip(queries, candidates_per_query):
            ids_here = [product_id for (product_id, _text) in candidates]
            candidate_ids_per_query.append(ids_here)
            counts.append(len(candidates))
            for _pid, product_text in candidates:
                all_pairs.append([query, product_text])

        total_pairs = len(all_pairs)
        print(f"      [reranker] Scoring {total_pairs:,} (query, product) pairs "
              f"across {total_queries:,} queries...")

        # 2) Score the pairs in manageable chunks so we can print progress and
        #    keep memory bounded. (model.predict already batches internally, but
        #    chunking lets us show that the long step is making progress.)
        chunk_size = max(self.batch_size * 20, 1000)
        all_scores = np.zeros(total_pairs, dtype="float64")
        for start in range(0, total_pairs, chunk_size):
            end = min(start + chunk_size, total_pairs)
            all_scores[start:end] = self.score_pairs(all_pairs[start:end])
            done_pct = end / total_pairs * 100 if total_pairs else 100
            print(f"        ...scored {end:,}/{total_pairs:,} pairs "
                  f"({done_pct:5.1f}%)")

        # 3) Split the flat score array back into per-query slices and sort each.
        all_results = []
        cursor = 0
        for query_number in range(total_queries):
            count = counts[query_number]
            query_scores = all_scores[cursor:cursor + count]
            ids_here = candidate_ids_per_query[query_number]
            cursor += count

            order = np.argsort(-query_scores)[:top_k]
            ranked = [
                (str(ids_here[pos]), float(query_scores[pos]), rank)
                for rank, pos in enumerate(order, start=1)
            ]
            all_results.append(ranked)

            if progress_every and (query_number + 1) % progress_every == 0:
                print(f"        ...re-ranked {query_number + 1:,}/"
                      f"{total_queries:,} queries")

        return all_results


# ---------------------------------------------------------------------------
# Tiny self-test so you can run this file on its own and see it work.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A query and four candidate products (as a retriever would hand them over).
    demo_query = "iphone charger"
    demo_candidates = [
        ("P1", "Apple Lightning USB Cable for iPhone charging cord"),
        ("P2", "Samsung 4K UHD Smart Television remote control"),
        ("P3", "USB-C fast charger power adapter for Android phones"),
        ("P4", "Stainless steel kitchen blender 700W"),
    ]

    reranker = CrossEncoderReranker()
    print(f"\nQuery: '{demo_query}'  -- cross-encoder re-ranking 4 candidates:")
    for pid, score, rank in reranker.rerank(demo_query, demo_candidates, top_k=4):
        print(f"  rank {rank}: {pid}  cross_encoder_score={score:+.3f}")
    print("\nCrossEncoderReranker self-test completed.")
