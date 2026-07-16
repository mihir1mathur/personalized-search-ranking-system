"""
hybrid_retriever.py  --  Hybrid retrieval: BM25 + Embeddings (Week 3)
=====================================================================

WHAT THIS DOES (beginner view):
Week 1 gave us BM25 (keyword search). Week 2 gave us embeddings (meaning
search). Each is strong where the other is weak:
  - BM25 nails EXACT words (brand names, model numbers).
  - Embeddings understand MEANING (synonyms, typos, paraphrases).

HYBRID search combines BOTH into a single score, so we get keyword precision
AND semantic recall:

    hybrid_score = alpha * (BM25 score)  +  beta * (embedding score)

THE TWO SCORES ARE ON DIFFERENT SCALES, so we cannot just add them:
  - BM25 scores are unbounded (often 0 .. ~40).
  - Embedding cosine scores are roughly -1 .. 1.
To make them comparable we MIN-MAX NORMALIZE each one into the range 0..1
FIRST, then take the weighted sum.

    normalized = (score - min) / (max - min)

HOW WE COMBINE PER QUERY:
  1. Ask BM25 for a score for every product (get_scores).
  2. Ask the embedding model for a cosine score for every product
     (product_vectors . query_vector).
  3. Take a CANDIDATE POOL = the top-N products from each method, unioned
     (so we only rank a small, promising set, which is fast).
  4. Min-max normalize the BM25 scores and the embedding scores of that pool
     SEPARATELY.
  5. hybrid = alpha*bm25_norm + beta*emb_norm.
  6. Sort by hybrid score, return the Top-K.

This file exposes a HybridRetriever class with the SAME fit / retrieve /
retrieve_batch interface as the Week 1 and Week 2 retrievers.
"""

import numpy as np

# Reuse the Week 1 BM25 retriever and the Week 2 embedding retriever as-is.
# (We import, we do NOT modify them.)
try:
    from bm25_retriever import BM25Retriever, simple_tokenize
    from embedding_retriever import EmbeddingRetriever, DEFAULT_MODEL_NAME
    from faiss_index import normalize_vectors
except ImportError:  # allow running from the project root too
    from src.bm25_retriever import BM25Retriever, simple_tokenize  # type: ignore
    from src.embedding_retriever import EmbeddingRetriever, DEFAULT_MODEL_NAME  # type: ignore
    from src.faiss_index import normalize_vectors  # type: ignore


def min_max_normalize(scores):
    """
    Min-Max normalization: rescale a list of numbers into the range 0..1.

        normalized = (score - min) / (max - min)

    The smallest score becomes 0, the largest becomes 1, everything else lands
    in between. If every score is the same (max == min), we return all zeros to
    avoid dividing by zero.
    """
    scores = np.asarray(scores, dtype="float64")
    smallest = scores.min()
    largest = scores.max()
    spread = largest - smallest
    if spread < 1e-12:
        # All scores identical -> nothing to separate; return zeros.
        return np.zeros_like(scores)
    return (scores - smallest) / spread


class HybridRetriever:
    """Combine BM25 (keyword) and embedding (semantic) scores into one ranking."""

    def __init__(self, alpha=0.5, beta=0.5, candidate_pool_size=100,
                 embedding_model_name=DEFAULT_MODEL_NAME,
                 bm25_tokenizer=simple_tokenize):
        """
        alpha               : weight on the BM25 (keyword) score.
        beta                : weight on the embedding (semantic) score.
                              Defaults 0.5 / 0.5 (equal mix).
        candidate_pool_size : how many top products to take from EACH method
                              before combining (bigger = more thorough, slower).
        """
        self.alpha = alpha
        self.beta = beta
        self.candidate_pool_size = candidate_pool_size

        # The two underlying retrievers (built in fit()).
        self.bm25 = BM25Retriever(tokenizer=bm25_tokenizer)
        self.embedder = EmbeddingRetriever(model_name=embedding_model_name)

        self.product_ids = None       # product_id for each corpus row
        self.product_vectors = None   # normalized product embeddings (N x 384)

    # ----------------------------------------------------------------------
    # Build both indexes
    # ----------------------------------------------------------------------
    def fit(self, product_texts, product_ids, precomputed_embeddings=None):
        """
        Build BOTH the BM25 index and the embedding index over the corpus.

        product_texts         : list of product_text strings.
        product_ids           : list of product_id strings (same order).
        precomputed_embeddings: OPTIONAL cached product embeddings, so we can
                                skip the slow encoding step (highly recommended).
        """
        print("      [hybrid] Building BM25 index...")
        self.bm25.fit(product_texts, product_ids)

        print("      [hybrid] Building embedding index...")
        self.embedder.fit(product_texts, product_ids,
                          precomputed_vectors=precomputed_embeddings)

        self.product_ids = np.asarray(product_ids)
        # The embedder already normalized these (length 1), so the dot product
        # of a normalized query with them gives cosine similarity.
        self.product_vectors = self.embedder.product_vectors
        return self

    # ----------------------------------------------------------------------
    # Raw full-corpus scores for one query (BM25 + embedding)
    # ----------------------------------------------------------------------
    def _bm25_scores_all(self, query):
        """BM25 score for EVERY product in the corpus (a numpy array)."""
        query_tokens = self.bm25.tokenizer(query)
        return np.asarray(self.bm25.bm25.get_scores(query_tokens), dtype="float64")

    def _embedding_scores_all(self, query_vector_normalized):
        """Cosine score for EVERY product = product_vectors . query_vector."""
        return self.product_vectors @ query_vector_normalized

    def _top_k_ranking(self, scores_all, top_k):
        """Turn a full score array into a Top-K [(product_id, score, rank)] list."""
        k = min(top_k, scores_all.shape[0])
        top_positions = np.argpartition(-scores_all, k - 1)[:k]
        top_positions = top_positions[np.argsort(-scores_all[top_positions])]
        return [
            (str(self.product_ids[pos]), float(scores_all[pos]), rank)
            for rank, pos in enumerate(top_positions, start=1)
        ]

    # ----------------------------------------------------------------------
    # Candidate pool + normalized scores for a set of queries (the slow step)
    # ----------------------------------------------------------------------
    def precompute_candidates(self, queries, pure_top_k=10):
        """
        For each query, compute the candidate pool and the SEPARATELY
        min-max-normalized BM25 and embedding scores for that pool.

        Returns a list (one entry per query) of dicts:
            {
              "product_ids" : np.array of candidate product_id strings,
              "bm25_norm"   : np.array of normalized BM25 scores (0..1),
              "emb_norm"    : np.array of normalized embedding scores (0..1),
              "bm25_ranking": pure BM25 Top-K [(product_id, raw_score, rank)],
              "emb_ranking" : pure embedding Top-K [(product_id, cosine, rank)],
            }

        The two "pure" rankings are free byproducts of the same score pass, so
        the evaluation can report BM25-only and embedding-only baselines without
        recomputing anything (and they stay perfectly consistent with hybrid).

        WHY a candidate pool? Scoring/sorting all ~48k products for every weight
        setting is wasteful. The truly relevant product is virtually always in
        the top-N of BM25 OR embeddings, so we union those two short lists and
        only rank that small set. This is the standard, fast way to do hybrid.
        """
        # Encode all queries at once (fast), then normalize so dot == cosine.
        print(f"      [hybrid] Encoding {len(queries):,} queries into embeddings...")
        query_vectors = self.embedder.encode_texts(list(queries), show_progress=False)
        query_vectors = normalize_vectors(query_vectors)

        pool_size = self.candidate_pool_size
        candidates = []

        print(f"      [hybrid] Scoring {len(queries):,} queries with BM25 + embeddings...")
        for query_number, query in enumerate(queries):
            # 1) full-corpus scores from each method
            bm25_all = self._bm25_scores_all(query)
            emb_all = self._embedding_scores_all(query_vectors[query_number])

            # 2) top-N positions from each method
            k = min(pool_size, bm25_all.shape[0])
            top_bm25_positions = np.argpartition(-bm25_all, k - 1)[:k]
            top_emb_positions = np.argpartition(-emb_all, k - 1)[:k]

            # 3) union of the two candidate sets (unique product positions)
            candidate_positions = np.unique(
                np.concatenate([top_bm25_positions, top_emb_positions])
            )

            # 4) normalize each score type separately over the candidate pool
            bm25_norm = min_max_normalize(bm25_all[candidate_positions])
            emb_norm = min_max_normalize(emb_all[candidate_positions])

            # 5) pure BM25-only and embedding-only Top-K (free baselines)
            bm25_ranking = self._top_k_ranking(bm25_all, pure_top_k)
            emb_ranking = self._top_k_ranking(emb_all, pure_top_k)

            candidates.append({
                "product_ids": self.product_ids[candidate_positions],
                "bm25_norm": bm25_norm,
                "emb_norm": emb_norm,
                "bm25_ranking": bm25_ranking,
                "emb_ranking": emb_ranking,
            })

            # Progress every 500 queries so the user sees it is alive.
            if (query_number + 1) % 500 == 0:
                print(f"        ...scored {query_number + 1:,}/{len(queries):,} queries")

        return candidates

    # ----------------------------------------------------------------------
    # Combine precomputed candidates with a given alpha/beta into a ranking
    # ----------------------------------------------------------------------
    def rank_from_candidates(self, candidates, alpha=None, beta=None, top_k=10):
        """
        Given precomputed candidates, apply the weighted hybrid formula and
        return the Top-K per query. Reuses the (expensive) precomputed scores,
        so trying many alpha/beta settings is cheap.
        """
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta

        all_results = []
        for one in candidates:
            hybrid_scores = alpha * one["bm25_norm"] + beta * one["emb_norm"]
            # Sort candidates by hybrid score, highest first.
            order = np.argsort(-hybrid_scores)[:top_k]
            ranked = [
                (str(one["product_ids"][pos]), float(hybrid_scores[pos]), rank)
                for rank, pos in enumerate(order, start=1)
            ]
            all_results.append(ranked)
        return all_results

    # ----------------------------------------------------------------------
    # Public retrieval API (same shape as the Week 1 / Week 2 retrievers)
    # ----------------------------------------------------------------------
    def retrieve_batch(self, queries, top_k=10, alpha=None, beta=None):
        """
        Retrieve the Top-K hybrid results for MANY queries.
        Returns a list (one per query) of [(product_id, hybrid_score, rank), ...].
        """
        candidates = self.precompute_candidates(queries)
        return self.rank_from_candidates(candidates, alpha=alpha, beta=beta, top_k=top_k)

    def retrieve(self, query, top_k=10, alpha=None, beta=None):
        """Retrieve the Top-K hybrid results for a SINGLE query."""
        return self.retrieve_batch([query], top_k=top_k, alpha=alpha, beta=beta)[0]


# ---------------------------------------------------------------------------
# Tiny self-test so you can run this file on its own and see it work.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts = [
        "apple lightning usb cable for iphone charging cord",
        "samsung 4k uhd smart television",
        "logitech wireless mouse for laptop",
        "stainless steel kitchen blender",
    ]
    demo_ids = ["P1", "P2", "P3", "P4"]

    hybrid = HybridRetriever(alpha=0.5, beta=0.5, candidate_pool_size=4)
    hybrid.fit(demo_texts, demo_ids)

    print("\nQuery: 'iphone charger cord'  (cord != cable for keyword search)")
    for pid, score, rank in hybrid.retrieve("iphone charger cord", top_k=3):
        print(f"  rank {rank}: {pid}  hybrid={score:.3f}")
    print("\nHybridRetriever self-test completed.")
