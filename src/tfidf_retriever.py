"""
tfidf_retriever.py  --  TF-IDF retrieval baseline (Week 1)
==========================================================

WHAT THIS DOES (beginner view):
TF-IDF turns each piece of text into a list of numbers ("a vector") based on
which words it contains and how important those words are. To find products
for a search query, we:
  1. Turn every product_text into a TF-IDF vector (this is the "index").
  2. Turn the query into a TF-IDF vector the same way.
  3. Measure how similar the query vector is to each product vector using
     COSINE SIMILARITY (the angle between two vectors).
  4. Return the products with the highest similarity, best first.

WHY COSINE SIMILARITY? scikit-learn's TfidfVectorizer makes vectors of length
1 (it "L2-normalizes" them) by default. For length-1 vectors, the dot product
IS the cosine similarity. So a simple matrix multiply gives us cosine scores.

This file can be:
  - imported as a module (use the TfidfRetriever class), or
  - run directly (`python src/tfidf_retriever.py`) for a tiny self-test.
"""

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    # Friendly message if a required library is missing.
    raise ImportError(
        "TF-IDF needs scikit-learn and numpy.\n"
        "Please install them with:\n"
        "    pip install scikit-learn numpy"
    )


class TfidfRetriever:
    """A simple TF-IDF + cosine-similarity retriever."""

    def __init__(self, max_features=50000, ngram_range=(1, 2)):
        """
        max_features : keep at most this many distinct words/phrases (keeps the
                       index small and fast). 50,000 is plenty for our corpus.
        ngram_range  : (1, 2) means we use single words AND two-word phrases,
                       which helps match things like "running shoes".
        """
        # norm="l2" (the default) makes dot product == cosine similarity.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.doc_matrix = None   # the TF-IDF vectors for every product
        self.product_ids = None  # product_id that matches each row of doc_matrix

    def fit(self, texts, product_ids):
        """
        Build the index from the product corpus.
        texts        : list of product_text strings.
        product_ids  : list of product_id strings (same order as texts).
        """
        if len(texts) != len(product_ids):
            raise ValueError("texts and product_ids must have the same length.")
        # Learn the vocabulary and create one TF-IDF vector per product.
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.product_ids = np.asarray(product_ids)
        return self

    def _top_k_from_scores(self, scores, top_k):
        """Given a score for every product, return the top_k as (id, score)."""
        # If we have fewer products than top_k, just rank them all.
        k = min(top_k, scores.shape[0])
        # argpartition quickly finds the k highest scores (unordered)...
        top_idx = np.argpartition(-scores, k - 1)[:k]
        # ...then we sort just those k by score, highest first.
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(str(self.product_ids[i]), float(scores[i])) for i in top_idx]

    def retrieve(self, query, top_k=10):
        """
        Retrieve the top_k products for a single query.
        Returns a list of (product_id, score, rank), best first (rank starts at 1).
        """
        if self.doc_matrix is None:
            raise RuntimeError("Call fit(...) before retrieve(...).")
        # Turn the query into a TF-IDF vector using the SAME vocabulary.
        q_vec = self.vectorizer.transform([query])
        # Cosine similarity to every product = q_vec . doc_matrix^T (vectors are L2-normalized).
        scores = (q_vec @ self.doc_matrix.T).toarray().ravel()
        ranked = self._top_k_from_scores(scores, top_k)
        return [(pid, score, rank) for rank, (pid, score) in enumerate(ranked, start=1)]

    def retrieve_batch(self, queries, top_k=10, chunk_size=256):
        """
        Retrieve for MANY queries efficiently.
        We process queries in chunks so we never build one giant score matrix.
        Returns a list (one per query) of [(product_id, score, rank), ...].
        """
        if self.doc_matrix is None:
            raise RuntimeError("Call fit(...) before retrieve_batch(...).")
        results = []
        doc_matrix_T = self.doc_matrix.T  # transpose once, reuse for every chunk
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start:start + chunk_size]
            q_mat = self.vectorizer.transform(chunk)          # (chunk x vocab)
            score_block = (q_mat @ doc_matrix_T).toarray()    # (chunk x num_products)
            for row in score_block:
                ranked = self._top_k_from_scores(row, top_k)
                results.append(
                    [(pid, score, rank) for rank, (pid, score) in enumerate(ranked, start=1)]
                )
        return results


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

    retriever = TfidfRetriever().fit(demo_texts, demo_ids)
    print("Query: 'running shoes'")
    for pid, score, rank in retriever.retrieve("running shoes", top_k=3):
        print(f"  rank {rank}: {pid}  score={score:.3f}")
    print("\nTfidfRetriever self-test completed.")
