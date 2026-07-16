"""
embedding_retriever.py  --  Semantic (meaning-based) retrieval (Week 2)
=======================================================================

WHAT THIS DOES (beginner view):
Week 1 used keyword methods (TF-IDF, BM25) that match the exact WORDS in the
query and the product. They miss synonyms: "pjs" vs "pajamas", "cord" vs
"cable". Week 2 fixes this with SEMANTIC search.

The idea:
  1. A neural network called a "sentence transformer" reads each product's
     text and turns it into a short list of numbers (an "embedding") that
     captures its MEANING. Products that mean similar things get similar
     embeddings.
  2. We turn the search query into an embedding the SAME way.
  3. We find the products whose embeddings are closest in MEANING to the
     query's embedding (using cosine similarity, made fast by FAISS).
  4. We return those closest products, best first.

Because matching happens in "meaning space" instead of "exact-word space",
"holiday pjs" can match a "Christmas pajamas" product even though they share
no words.

MODEL WE USE:
  sentence-transformers/all-MiniLM-L6-v2  -- small (384-number embeddings),
  fast, and a very common industry baseline for semantic search.

This file gives an EmbeddingRetriever class with the SAME simple interface as
the Week 1 retrievers (fit, retrieve, retrieve_batch), so the evaluation code
can treat all three methods the same way.
"""

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Friendly message if a required library is missing.
    raise ImportError(
        "Semantic retrieval needs 'sentence-transformers' (and numpy).\n"
        "Please install them with:\n"
        "    pip install sentence-transformers numpy"
    )

# We import our small FAISS helpers. This works whether the file is run from
# the project root or imported as a module from the src/ folder.
try:
    from faiss_index import (
        normalize_vectors,
        build_ip_index,
        search_index,
        save_index,
        load_index,
    )
except ImportError:
    from src.faiss_index import (  # type: ignore
        normalize_vectors,
        build_ip_index,
        search_index,
        save_index,
        load_index,
    )


# The model name is kept as a constant so it is easy to find and change.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever:
    """A semantic retriever: sentence-transformer embeddings + FAISS search."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=64):
        """
        model_name : which sentence-transformer model to load.
        batch_size : how many texts to encode at once (bigger = faster but
                     uses more memory). 64 is a safe default for a laptop.
        """
        print(f"      Loading embedding model: {model_name}")
        # This downloads the model the first time, then caches it locally.
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

        # These get filled in by fit():
        self.faiss_index = None      # the searchable FAISS index of products
        self.product_ids = None      # product_id for each row in the index
        self.product_vectors = None  # the normalized product embeddings (kept
                                     # so we can save them / reuse them)

    # ----------------------------------------------------------------------
    # Encoding helpers
    # ----------------------------------------------------------------------
    def encode_texts(self, texts, show_progress=True):
        """
        Turn a list of texts into a 2D numpy array of embeddings.

        We DO NOT normalize here; normalization happens right before we build
        or search the FAISS index, so the logic lives in one place.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    # ----------------------------------------------------------------------
    # Building the index (the "fit" step)
    # ----------------------------------------------------------------------
    def fit(self, product_texts, product_ids, precomputed_vectors=None):
        """
        Build the semantic index from the product corpus.

        product_texts       : list of product_text strings.
        product_ids         : list of product_id strings (same order as texts).
        precomputed_vectors : OPTIONAL. If you already encoded the products
                              before (e.g. loaded from a cache), pass them here
                              to skip the slow encoding step.
        """
        if len(product_texts) != len(product_ids):
            raise ValueError("product_texts and product_ids must be the same length.")

        if precomputed_vectors is None:
            print(f"      Encoding {len(product_texts):,} products into embeddings...")
            product_vectors = self.encode_texts(product_texts, show_progress=True)
        else:
            print("      Using precomputed product embeddings (skipping encoding).")
            product_vectors = precomputed_vectors

        # Normalize so the FAISS inner product behaves like cosine similarity.
        print("      Normalizing product embeddings (so inner product = cosine)...")
        product_vectors = normalize_vectors(product_vectors)

        # Build the FAISS index over the (already normalized) product vectors.
        print("      Building the FAISS IndexFlatIP...")
        self.faiss_index = build_ip_index(product_vectors, already_normalized=True)

        # Remember the ids and the vectors for searching and saving.
        self.product_ids = np.asarray(product_ids)
        self.product_vectors = product_vectors
        return self

    # ----------------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------------
    def retrieve(self, query, top_k=10):
        """
        Retrieve the top_k products for a SINGLE query.
        Returns a list of (product_id, score, rank), best first (rank from 1).
        """
        results_for_one = self.retrieve_batch([query], top_k=top_k)
        return results_for_one[0]

    def retrieve_batch(self, queries, top_k=10):
        """
        Retrieve the top_k products for MANY queries at once (efficient).
        Returns a list (one per query) of [(product_id, score, rank), ...].
        """
        if self.faiss_index is None:
            raise RuntimeError("Call fit(...) before retrieve(...).")

        # Encode all queries into embeddings (one row per query).
        query_vectors = self.encode_texts(list(queries), show_progress=False)

        # Ask FAISS for the top_k nearest product vectors for every query.
        # 'scores' are cosine similarities; 'positions' are row numbers into
        # our product_ids array.
        scores, positions = search_index(
            self.faiss_index, query_vectors, top_k=top_k, already_normalized=False
        )

        # Convert FAISS's raw output into our (product_id, score, rank) format.
        all_results = []
        for query_scores, query_positions in zip(scores, positions):
            one_query_results = []
            rank = 1
            for score, position in zip(query_scores, query_positions):
                # FAISS uses -1 to mark "no result" when there are fewer
                # products than top_k; we skip those.
                if position == -1:
                    continue
                product_id = str(self.product_ids[position])
                one_query_results.append((product_id, float(score), rank))
                rank += 1
            all_results.append(one_query_results)
        return all_results

    # ----------------------------------------------------------------------
    # Saving / loading the index (so we don't re-encode every run)
    # ----------------------------------------------------------------------
    def save(self, index_path, ids_path, vectors_path=None):
        """Save the FAISS index, the product ids, and (optionally) the vectors."""
        if self.faiss_index is None:
            raise RuntimeError("Nothing to save: call fit(...) first.")
        save_index(self.faiss_index, index_path)
        np.save(ids_path, self.product_ids)
        if vectors_path is not None:
            np.save(vectors_path, self.product_vectors)

    def load(self, index_path, ids_path):
        """Load a previously saved FAISS index and product ids."""
        self.faiss_index = load_index(index_path)
        self.product_ids = np.load(ids_path, allow_pickle=True)
        return self

    # ----------------------------------------------------------------------
    # Convenience: cosine similarity between two short texts (for demos)
    # ----------------------------------------------------------------------
    def cosine_similarity(self, text_a, text_b):
        """
        Return the cosine similarity (a number from -1 to 1, usually 0 to 1)
        between two pieces of text. Handy for showing that "tv" and
        "television" are close in meaning.
        """
        two_vectors = self.encode_texts([text_a, text_b], show_progress=False)
        two_vectors = normalize_vectors(two_vectors)  # length 1 each
        # With unit vectors, the dot product IS the cosine similarity.
        return float(np.dot(two_vectors[0], two_vectors[1]))


# ---------------------------------------------------------------------------
# Tiny self-test so you can run this file on its own and see it work.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts = [
        "red running shoes for men lightweight",
        "blue running sneakers women breathable",
        "stainless steel kitchen blender",
        "wireless bluetooth headphones noise cancelling",
    ]
    demo_ids = ["P1", "P2", "P3", "P4"]

    retriever = EmbeddingRetriever().fit(demo_texts, demo_ids)
    print("\nQuery: 'jogging trainers'  (note: NO exact word overlap!)")
    for pid, score, rank in retriever.retrieve("jogging trainers", top_k=3):
        print(f"  rank {rank}: {pid}  cosine={score:.3f}")

    print("\nSimilarity demo:")
    print(f"  shoes vs sneakers : {retriever.cosine_similarity('shoes', 'sneakers'):.3f}")
    print(f"  shoes vs blender  : {retriever.cosine_similarity('shoes', 'blender'):.3f}")
    print("\nEmbeddingRetriever self-test completed.")
