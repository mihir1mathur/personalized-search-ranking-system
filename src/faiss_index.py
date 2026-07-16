"""
faiss_index.py  --  FAISS vector index helpers (Week 2)
=======================================================

WHAT THIS DOES (beginner view):
After we turn every product into a list of numbers (an "embedding"), we need
a fast way to ask: "which product vectors are MOST SIMILAR to this query
vector?" Comparing the query to all ~48,000 products one-by-one in plain
Python is slow. FAISS (Facebook AI Similarity Search) is a library that
stores the vectors in a special "index" and finds the nearest ones very fast.

WHICH INDEX WE USE:
  IndexFlatIP  --  "Flat" means it keeps the exact vectors (no approximation,
                   so results are exact), and "IP" means it scores by INNER
                   PRODUCT (a dot product).

WHY INNER PRODUCT GIVES COSINE SIMILARITY:
  Cosine similarity = dot product divided by the lengths of both vectors.
  If we first NORMALIZE every vector to length 1 (so its length = 1), then the
  division does nothing, and the inner product IS exactly the cosine
  similarity. So the recipe is: normalize first, then use IndexFlatIP.

This file gives small, well-commented helper functions:
  - normalize_vectors(...)   make vectors length 1 (so IP == cosine)
  - build_ip_index(...)      create the FAISS IndexFlatIP and add vectors
  - search_index(...)        find the top-k nearest vectors for queries
  - save_index(...)          write the index to disk
  - load_index(...)          read the index back from disk
"""

try:
    import numpy as np
    import faiss
except ImportError:
    # Friendly message if a required library is missing.
    raise ImportError(
        "Week 2 needs the 'faiss' library (and numpy).\n"
        "Please install them with:\n"
        "    pip install faiss-cpu numpy"
    )


def normalize_vectors(vectors):
    """
    Make every row vector have length 1 ("L2 normalization").

    WHY: once vectors have length 1, the inner product between two of them
    equals their cosine similarity. FAISS has a fast in-place helper for this,
    faiss.normalize_L2, which expects a contiguous float32 array.

    vectors : a 2D numpy array of shape (number_of_items, embedding_size).
    returns : the same data as a float32 array with each row normalized.
    """
    # FAISS works with 32-bit floats; make sure the array is the right type
    # and laid out contiguously in memory (a requirement of normalize_L2).
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    # This modifies 'vectors' in place, dividing each row by its own length.
    faiss.normalize_L2(vectors)
    return vectors


def build_ip_index(product_vectors, already_normalized=False):
    """
    Build a FAISS IndexFlatIP (inner-product) index from product embeddings.

    product_vectors    : 2D numpy array (num_products, embedding_size).
    already_normalized : set True if you normalized the vectors yourself
                         already. If False (default) we normalize here so the
                         inner product behaves like cosine similarity.

    returns : a ready-to-search FAISS index containing all product vectors.
    """
    if not already_normalized:
        product_vectors = normalize_vectors(product_vectors)
    else:
        # Even if the caller normalized, make sure the dtype/layout is correct.
        product_vectors = np.ascontiguousarray(product_vectors, dtype="float32")

    embedding_size = product_vectors.shape[1]  # how many numbers per vector

    # IndexFlatIP = exact search using inner product (= cosine on unit vectors).
    index = faiss.IndexFlatIP(embedding_size)

    # Add every product vector to the index so we can search them later.
    index.add(product_vectors)

    return index


def search_index(index, query_vectors, top_k=10, already_normalized=False):
    """
    Find the top_k most similar product vectors for each query vector.

    index            : a FAISS index built with build_ip_index().
    query_vectors    : 2D numpy array (num_queries, embedding_size).
    top_k            : how many nearest products to return per query.
    already_normalized : set True if you normalized the query vectors yourself.

    returns : (scores, positions)
      scores    -> 2D array (num_queries, top_k) of cosine similarities.
      positions -> 2D array (num_queries, top_k) of ROW NUMBERS into the
                   product list (i.e. which product, by its position).
    """
    if not already_normalized:
        query_vectors = normalize_vectors(query_vectors)
    else:
        query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")

    # FAISS returns the similarity scores and the positions (indices) of the
    # best matches, both sorted best-first for every query.
    scores, positions = index.search(query_vectors, top_k)
    return scores, positions


def save_index(index, path):
    """Write a FAISS index to disk so we don't have to rebuild it next time."""
    faiss.write_index(index, path)


def load_index(path):
    """Read a FAISS index back from disk (created earlier by save_index)."""
    return faiss.read_index(path)
