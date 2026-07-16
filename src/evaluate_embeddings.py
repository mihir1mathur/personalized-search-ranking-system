"""
evaluate_embeddings.py  --  Week 2 main script: semantic retrieval + comparison
================================================================================

Run it with:

    python src/evaluate_embeddings.py

WHAT IT DOES, STEP BY STEP:
  1. Loads the same cleaned 50k sample used in Week 0 / Week 1.
  2. Builds the SAME product corpus (one row per unique product) and the SAME
     relevance judgments (qrels) and the SAME 3,000 evaluation queries as
     Week 1 -- by reusing Week 1's own functions, so the comparison is fair.
  3. Builds a SEMANTIC retriever: it encodes every product into an embedding
     with sentence-transformers/all-MiniLM-L6-v2, normalizes the embeddings,
     and indexes them in FAISS (IndexFlatIP = cosine similarity).
  4. Also builds the Week 1 TF-IDF and BM25 retrievers, so we can compare all
     three methods head-to-head on the exact same queries.
  5. Retrieves the Top-10 products per query with each method.
  6. Scores Precision@10, Recall@10, NDCG@10 (using Week 1's metrics module).
  7. Writes all Week 2 output files into results/.

WHAT IT DOES NOT DO (on purpose -- these are future weeks):
  rerankers, cross-encoders, fine-tuning, reinforcement learning, LLM
  retrieval, or hybrid (keyword + vector) search.

NOTE ON SPEED:
  Encoding ~48k products on a CPU takes a few minutes the first time. We CACHE
  the product embeddings to results/ so future runs are fast.
"""

import os
import sys
import gc

# ---------------------------------------------------------------------------
# Make our own modules importable (src/ and evaluation/ folders).
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
for path in (SRC_DIR, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# ---------------------------------------------------------------------------
# Imports with friendly error messages.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERROR: This script needs pandas and numpy.")
    print("Install them with:  pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")  # "Agg" lets us save images without opening a window
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: This script needs matplotlib for the plots.")
    print("Install it with:  pip install matplotlib")
    sys.exit(1)

# We REUSE Week 1's building blocks so Week 2 is a fair, apples-to-apples
# comparison: the SAME corpus, the SAME qrels, and the SAME 3,000 queries.
try:
    import metrics  # evaluation/metrics.py (Precision@K, Recall@K, NDCG@K)
    from tfidf_retriever import TfidfRetriever
    from bm25_retriever import BM25Retriever, simple_tokenize
    import evaluate_retrieval as week1  # reuse load/corpus/qrels/eval-query logic
except ImportError as e:
    print(f"ERROR: Could not import a Week 1 module: {e}")
    print("Make sure Week 1 files (evaluate_retrieval.py, tfidf_retriever.py,")
    print("bm25_retriever.py) are in src/, and metrics.py is in evaluation/.")
    sys.exit(1)

# The semantic retriever is the new Week 2 piece.
try:
    from embedding_retriever import EmbeddingRetriever, DEFAULT_MODEL_NAME
except ImportError as e:
    print(f"ERROR: Could not import the Week 2 embedding retriever: {e}")
    print("Install its dependencies with:")
    print("    pip install sentence-transformers faiss-cpu")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration. We reuse Week 1's constants so the setup is identical.
# ---------------------------------------------------------------------------
RESULTS_DIR = week1.RESULTS_DIR
TOP_K = week1.TOP_K                      # 10
REL_THRESHOLD = week1.REL_THRESHOLD      # 1 (score >= 1 counts as relevant)

# Output files for Week 2 (all go into results/).
WEEK2_METRICS_TXT = os.path.join(RESULTS_DIR, "week2_metrics.txt")
EMB_FAILURE_TXT = os.path.join(RESULTS_DIR, "embedding_failure_cases.txt")
SAMPLE_EMB_TXT = os.path.join(RESULTS_DIR, "sample_embedding_queries.txt")
BM25_VS_EMB_PNG = os.path.join(RESULTS_DIR, "bm25_vs_embedding.png")
SIMILARITY_TXT = os.path.join(RESULTS_DIR, "embedding_similarity_examples.txt")

# Cache files so we do not re-encode 48k products every single run.
EMB_CACHE_VECTORS = os.path.join(RESULTS_DIR, "product_embeddings_minilm.npy")
EMB_CACHE_IDS = os.path.join(RESULTS_DIR, "product_embeddings_ids.npy")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def shorten(text, limit=70):
    """Trim long product text so the report tables stay readable."""
    text = " ".join(str(text).split())  # collapse whitespace
    return text if len(text) <= limit else text[:limit - 1] + "…"


def per_query_scores(retrieved_per_query, eval_queries, qrels):
    """
    Compute per-query Precision@10, Recall@10, NDCG@10 for one method.
    Returns three lists aligned with eval_queries (Recall/NDCG may hold None
    when undefined for a query, exactly like Week 1).
    """
    precisions, recalls, ndcgs = [], [], []
    for query, retrieved in zip(eval_queries, retrieved_per_query):
        retrieved_ids = [pid for (pid, _score, _rank) in retrieved]
        query_qrels = qrels.get(query, {})
        precisions.append(metrics.precision_at_k(retrieved_ids, query_qrels, TOP_K, REL_THRESHOLD))
        recalls.append(metrics.recall_at_k(retrieved_ids, query_qrels, TOP_K, REL_THRESHOLD))
        ndcgs.append(metrics.ndcg_at_k(retrieved_ids, query_qrels, TOP_K))
    return precisions, recalls, ndcgs


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_metrics_report(tfidf_metrics, bm25_metrics, emb_metrics,
                         n_queries, n_products, n_eval):
    """Write results/week2_metrics.txt with the 3-way comparison table."""
    lines = []
    bar = "=" * 64
    lines.append(bar)
    lines.append("WEEK 2: SEMANTIC RETRIEVAL (EMBEDDINGS + FAISS) -- METRICS REPORT")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append("")
    lines.append("SETUP")
    lines.append("-" * 64)
    lines.append(f"Embedding model                 : {DEFAULT_MODEL_NAME}")
    lines.append("Vector index                    : FAISS IndexFlatIP (cosine, normalized)")
    lines.append(f"Search corpus (unique products) : {n_products:,}")
    lines.append(f"Total unique queries in sample  : {n_queries:,}")
    lines.append(f"Queries evaluated (same as Week1): {n_eval:,}")
    lines.append(f"Top-K retrieved per query       : {TOP_K}")
    lines.append(f"Relevance threshold (relevant if score >= ): {REL_THRESHOLD}")
    lines.append("Relevance scale: 3=highly relevant, 2=relevant, "
                 "1=weakly relevant, 0=irrelevant")
    lines.append("")
    lines.append("COMPARISON TABLE (averages across the SAME evaluated queries)")
    lines.append("-" * 64)
    lines.append("| Method     | Precision@10 | Recall@10 | NDCG@10 |")
    lines.append("| ---------- | ------------ | --------- | ------- |")
    lines.append("| TF-IDF     | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        tfidf_metrics["Precision@10"], tfidf_metrics["Recall@10"], tfidf_metrics["NDCG@10"]))
    lines.append("| BM25       | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        bm25_metrics["Precision@10"], bm25_metrics["Recall@10"], bm25_metrics["NDCG@10"]))
    lines.append("| Embeddings | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        emb_metrics["Precision@10"], emb_metrics["Recall@10"], emb_metrics["NDCG@10"]))
    lines.append("")

    # Decide the winner on NDCG@10 (the headline ranking metric).
    by_ndcg = sorted(
        [("TF-IDF", tfidf_metrics), ("BM25", bm25_metrics), ("Embeddings", emb_metrics)],
        key=lambda kv: kv[1]["NDCG@10"], reverse=True,
    )
    winner = by_ndcg[0][0]
    lines.append("WHICH METHOD PERFORMS BEST?")
    lines.append("-" * 64)
    lines.append(f"On NDCG@10 (our headline ranking metric), the best method is: {winner}.")
    lines.append("")

    # A small, automatically-computed comparison of embeddings vs BM25.
    def pct_change(new, old):
        return (new - old) / old * 100.0 if old else float("nan")
    lines.append("EMBEDDINGS vs BM25 (relative change):")
    lines.append(f"  Precision@10: {pct_change(emb_metrics['Precision@10'], bm25_metrics['Precision@10']):+.1f}%")
    lines.append(f"  Recall@10   : {pct_change(emb_metrics['Recall@10'],    bm25_metrics['Recall@10']):+.1f}%")
    lines.append(f"  NDCG@10     : {pct_change(emb_metrics['NDCG@10'],      bm25_metrics['NDCG@10']):+.1f}%")
    lines.append("")
    lines.append("WHY SEMANTIC RETRIEVAL CAN HELP:")
    lines.append("  - Embeddings match MEANING, not exact words, so synonyms")
    lines.append("    (cord/cable), abbreviations (pjs/pajamas) and paraphrases")
    lines.append("    can match even with zero shared words.")
    lines.append("  - This mainly improves RECALL (finding the relevant product")
    lines.append("    at all) and NDCG (ranking it near the top).")
    lines.append("")
    lines.append("LIMITATIONS (honest notes):")
    lines.append("  - Embeddings can miss EXACT identifiers (model numbers, rare")
    lines.append("    brand codes) that keyword search nails -- a reason hybrid")
    lines.append("    search (a future week) often wins.")
    lines.append("  - A single small model (MiniLM) caps quality; bigger models /")
    lines.append("    fine-tuning / re-rankers (future weeks) can do better.")
    lines.append("  - Precision@10 stays low here because of SPARSE labels: most")
    lines.append("    queries have only 1-2 judged-relevant products, so 1 relevant")
    lines.append("    item caps Precision@10 at 0.10. Recall@10 and NDCG@10 are the")
    lines.append("    informative metrics in this setup.")
    lines.append("")
    with open(WEEK2_METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved metrics report to: {WEEK2_METRICS_TXT}")


def write_failure_cases(eval_queries, qrels, product_text_by_id,
                        bm25_retrieved, emb_retrieved,
                        bm25_ndcg, bm25_recall, emb_ndcg, emb_recall):
    """
    Find and write 5 real queries where BM25 FAILS but EMBEDDINGS SUCCEED.
    "BM25 fails"        = no relevant product in BM25's Top-10 (recall == 0).
    "Embeddings succeed"= at least one relevant product in embeddings' Top-10
                          (recall > 0). We rank candidates by the embedding
                          NDCG gain so the clearest wins are shown first.
    """
    candidates = []
    for i, query in enumerate(eval_queries):
        b_rec = bm25_recall[i]
        e_rec = emb_recall[i]
        # Skip queries where recall is undefined (no known relevant items).
        if b_rec is None or e_rec is None:
            continue
        if b_rec == 0.0 and e_rec > 0.0:
            gain = (emb_ndcg[i] or 0.0) - (bm25_ndcg[i] or 0.0)
            candidates.append((gain, i))
    # Biggest embedding advantage first.
    candidates.sort(reverse=True)
    chosen = [i for (_gain, i) in candidates[:5]]

    lines = []
    bar = "=" * 64
    lines.append(bar)
    lines.append("WEEK 2: FAILURE CASES -- BM25 FAILS, EMBEDDINGS SUCCEED")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append("")
    lines.append("These are real evaluated queries where keyword search (BM25)")
    lines.append("put NO relevant product in its Top-10, but semantic retrieval")
    lines.append("(embeddings + FAISS) DID. For each we show:")
    lines.append("  1. Query   2. Ground truth   3. BM25 Top-5   4. Embedding Top-5")
    lines.append("  5. Why embeddings worked")
    lines.append("(rel = known relevance_score; 3=best ... 0=irrelevant. A retrieved")
    lines.append(" product with no judgment for this query shows rel=0.)")
    lines.append("")

    if not chosen:
        lines.append("No qualifying cases were found in this run (BM25 already")
        lines.append("retrieved a relevant item for every evaluated query where")
        lines.append("embeddings did). Try a larger query sample.")
    for n, i in enumerate(chosen, start=1):
        query = eval_queries[i]
        query_qrels = qrels.get(query, {})
        known = sorted(query_qrels.items(), key=lambda kv: -kv[1])
        lines.append(bar)
        lines.append(f"CASE {n}")
        lines.append(bar)
        lines.append(f"1. QUERY: {query}")
        lines.append("")
        lines.append("2. GROUND TRUTH (known judged products):")
        for pid, s in known:
            lines.append(f"   - {pid} (rel={s})  {shorten(product_text_by_id.get(pid, ''))}")
        lines.append("")
        lines.append("3. BM25 TOP-5  (keyword search -- it MISSED the relevant item):")
        for pid, score, rank in bm25_retrieved[i][:5]:
            rel = query_qrels.get(pid, 0)
            lines.append(f"   {rank}. {pid}  score={score:7.3f}  rel={rel}  "
                         f"{shorten(product_text_by_id.get(pid, ''), 55)}")
        lines.append("")
        lines.append("4. EMBEDDING TOP-5  (semantic search -- it FOUND a relevant item):")
        for pid, score, rank in emb_retrieved[i][:5]:
            rel = query_qrels.get(pid, 0)
            mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
            lines.append(f"   {rank}. {pid}  cos={score:6.3f}   rel={rel}  "
                         f"{shorten(product_text_by_id.get(pid, ''), 55)}{mark}")
        lines.append("")
        lines.append("5. WHY EMBEDDINGS WORKED:")
        lines.append(f"   BM25's Top-10 contained NO judged-relevant product (Recall@10=0,")
        lines.append(f"   NDCG@10={ (bm25_ndcg[i] or 0.0):.3f}), because the shopper's wording did not")
        lines.append("   literally overlap the relevant product's wording. The embedding")
        lines.append("   model encoded the MEANING of the query and the product into nearby")
        lines.append(f"   vectors, so it pulled the relevant item into the Top-10 (Recall@10")
        lines.append(f"   ={emb_recall[i]:.3f}, NDCG@10={ (emb_ndcg[i] or 0.0):.3f}) -- matching by meaning,")
        lines.append("   not by exact words.")
        lines.append("")

    with open(EMB_FAILURE_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved embedding failure cases to: {EMB_FAILURE_TXT}")


def write_sample_queries(eval_queries, qrels, product_text_by_id,
                         bm25_retrieved, emb_retrieved, seed=42):
    """Write results/sample_embedding_queries.txt: 10 random queries, Top-5 each."""
    rng = np.random.default_rng(seed)
    n = min(10, len(eval_queries))
    pick = sorted(rng.choice(len(eval_queries), size=n, replace=False).tolist())

    lines = []
    bar = "=" * 64
    lines.append(bar)
    lines.append("WEEK 2: SAMPLE EMBEDDING QUERIES (10 random queries)")
    lines.append(bar)
    lines.append("For each query: Top-5 from BM25 (keyword) vs Embeddings (semantic).")
    lines.append("rel = known relevance_score (3=best ... 0=irrelevant).")
    lines.append("")
    for i in pick:
        query = eval_queries[i]
        query_qrels = qrels.get(query, {})
        known = sorted(query_qrels.items(), key=lambda kv: -kv[1])
        lines.append("-" * 64)
        lines.append(f"QUERY: {query}")
        lines.append("Known judged products: " +
                     ", ".join(f"{pid}(rel={s})" for pid, s in known))
        lines.append("")
        lines.append("  Top-5 BM25:")
        for pid, score, rank in bm25_retrieved[i][:5]:
            rel = query_qrels.get(pid, 0)
            lines.append(f"    {rank}. {pid}  score={score:7.3f}  rel={rel}  "
                         f"{shorten(product_text_by_id.get(pid, ''), 50)}")
        lines.append("  Top-5 Embeddings:")
        for pid, score, rank in emb_retrieved[i][:5]:
            rel = query_qrels.get(pid, 0)
            lines.append(f"    {rank}. {pid}  cos={score:6.3f}   rel={rel}  "
                         f"{shorten(product_text_by_id.get(pid, ''), 50)}")
        lines.append("")
    with open(SAMPLE_EMB_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved sample embedding queries to: {SAMPLE_EMB_TXT}")


def write_similarity_examples(retriever):
    """
    Write results/embedding_similarity_examples.txt showing cosine similarity
    between meaning-related phrase pairs (and a few unrelated pairs to contrast).
    """
    related_pairs = [
        ("holiday pjs", "christmas pajamas"),
        ("tv", "television"),
        ("iphone cable", "lightning charger"),
        ("laptop", "notebook"),
        ("cell phone", "smartphone"),
    ]
    # Unrelated control pairs: these SHOULD score low, proving the model is not
    # just calling everything similar.
    control_pairs = [
        ("laptop", "banana"),
        ("running shoes", "kitchen blender"),
        ("television", "garden hose"),
    ]

    lines = []
    bar = "=" * 64
    lines.append(bar)
    lines.append("WEEK 2: EMBEDDING SIMILARITY EXAMPLES")
    lines.append(bar)
    lines.append(f"Model: {DEFAULT_MODEL_NAME}")
    lines.append("Cosine similarity ranges from -1 (opposite) to 1 (identical")
    lines.append("meaning). Keyword search would score every pair below as 0,")
    lines.append("because the two phrases share NO words. The embedding model")
    lines.append("still sees the related ones as close -- that is the whole point.")
    lines.append("")
    lines.append("RELATED PAIRS (different words, same meaning -> should be HIGH):")
    lines.append("-" * 64)
    for a, b in related_pairs:
        sim = retriever.cosine_similarity(a, b)
        lines.append(f"  cos( '{a}' , '{b}' ) = {sim:.3f}")
    lines.append("")
    lines.append("CONTROL PAIRS (unrelated -> should be LOW):")
    lines.append("-" * 64)
    for a, b in control_pairs:
        sim = retriever.cosine_similarity(a, b)
        lines.append(f"  cos( '{a}' , '{b}' ) = {sim:.3f}")
    lines.append("")
    lines.append("READING THIS: the related pairs score much higher than the")
    lines.append("control pairs, which is why semantic retrieval can connect a")
    lines.append("shopper's everyday words to a product's different wording.")
    lines.append("")
    with open(SIMILARITY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved embedding similarity examples to: {SIMILARITY_TXT}")


def plot_bm25_vs_embedding(bm25_metrics, emb_metrics):
    """Grouped bar chart: BM25 vs Embeddings across the three metrics."""
    metric_names = ["Precision@10", "Recall@10", "NDCG@10"]
    bm25_vals = [bm25_metrics[m] for m in metric_names]
    emb_vals = [emb_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, bm25_vals, width, label="BM25 (keyword)")
    bars2 = ax.bar(x + width / 2, emb_vals, width, label="Embeddings (semantic)")
    ax.set_ylabel("Score")
    ax.set_title("BM25 vs Embeddings  (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(BM25_VS_EMB_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved BM25-vs-Embedding chart to: {BM25_VS_EMB_PNG}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("WEEK 2: SEMANTIC RETRIEVAL (EMBEDDINGS + FAISS)")
    print("=" * 64)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load the SAME 50k sample as Week 0 / Week 1.
    print("\n[1/7] Loading the Week 0 sample...")
    df = week1.load_sample()
    print(f"      Loaded {len(df):,} rows.")

    # 2) Build the SAME corpus, qrels, and 3,000 eval queries as Week 1.
    print("[2/7] Building the corpus, qrels, and eval queries (reusing Week 1)...")
    product_ids, product_texts = week1.build_corpus(df)
    qrels = week1.build_qrels(df)
    eval_queries = week1.choose_eval_queries(qrels)
    n_unique_queries = len(qrels)
    print(f"      Corpus: {len(product_ids):,} unique products.")
    print(f"      {n_unique_queries:,} unique queries; evaluating {len(eval_queries):,}.")

    # A lookup so reports can show readable product text next to each id.
    product_text_by_id = dict(zip(product_ids, product_texts))

    # The big 50k DataFrame is no longer needed (we have corpus + qrels +
    # queries). Free it now to keep peak memory low during encoding.
    del df
    gc.collect()

    # 3) Build the SEMANTIC retriever (encode -> normalize -> FAISS).
    print("[3/7] Building the semantic (embedding) index...")
    retriever = EmbeddingRetriever()

    # Try to reuse cached embeddings so we do not re-encode 48k products.
    cached_vectors = None
    if os.path.exists(EMB_CACHE_VECTORS) and os.path.exists(EMB_CACHE_IDS):
        cached_ids = np.load(EMB_CACHE_IDS, allow_pickle=True)
        if list(map(str, cached_ids)) == list(map(str, product_ids)):
            print("      Found a matching embedding cache -- reusing it.")
            cached_vectors = np.load(EMB_CACHE_VECTORS)
        else:
            print("      Cache found but corpus changed -- will re-encode.")

    retriever.fit(product_texts, product_ids, precomputed_vectors=cached_vectors)

    # Save the cache for next time (only if we just encoded from scratch).
    if cached_vectors is None:
        print("      Caching product embeddings for faster future runs...")
        np.save(EMB_CACHE_VECTORS, retriever.product_vectors)
        np.save(EMB_CACHE_IDS, retriever.product_ids)

    # 3b) Build the Week 1 keyword retrievers for the head-to-head comparison.
    print("      Building the TF-IDF index (Week 1 baseline)...")
    tfidf = TfidfRetriever().fit(product_texts, product_ids)
    print("      Building the BM25 index (Week 1 baseline)...")
    bm25 = BM25Retriever(tokenizer=simple_tokenize).fit(product_texts, product_ids)

    # 4-5) Retrieve Top-10 per query with each method.
    print(f"[4/7] Retrieving Top-{TOP_K} per query (Embeddings, semantic)...")
    emb_retrieved = retriever.retrieve_batch(eval_queries, top_k=TOP_K)
    print(f"[5/7] Retrieving Top-{TOP_K} per query (TF-IDF, then BM25)...")
    tfidf_retrieved = tfidf.retrieve_batch(eval_queries, top_k=TOP_K)
    print("      (BM25 is the slow keyword part, please wait)...")
    bm25_retrieved = bm25.retrieve_batch(eval_queries, top_k=TOP_K)

    # 6) Score all three methods (reusing Week 1's evaluate_method for averages).
    print("[6/7] Scoring Precision@10, Recall@10, NDCG@10 for all 3 methods...")
    tfidf_metrics, _ = week1.evaluate_method("TF-IDF", tfidf_retrieved, eval_queries, qrels)
    bm25_metrics, _ = week1.evaluate_method("BM25", bm25_retrieved, eval_queries, qrels)
    emb_metrics, _ = week1.evaluate_method("Embeddings", emb_retrieved, eval_queries, qrels)
    print("      TF-IDF    :", {k: round(v, 4) for k, v in tfidf_metrics.items()})
    print("      BM25      :", {k: round(v, 4) for k, v in bm25_metrics.items()})
    print("      Embeddings:", {k: round(v, 4) for k, v in emb_metrics.items()})

    # Per-query scores (needed to find failure cases where BM25 fails).
    _, bm25_recall, bm25_ndcg = per_query_scores(bm25_retrieved, eval_queries, qrels)
    _, emb_recall, emb_ndcg = per_query_scores(emb_retrieved, eval_queries, qrels)

    # 7) Write all the output files.
    print("[7/7] Writing output files...")
    write_metrics_report(tfidf_metrics, bm25_metrics, emb_metrics,
                         n_unique_queries, len(product_ids), len(eval_queries))
    write_failure_cases(eval_queries, qrels, product_text_by_id,
                        bm25_retrieved, emb_retrieved,
                        bm25_ndcg, bm25_recall, emb_ndcg, emb_recall)
    write_sample_queries(eval_queries, qrels, product_text_by_id,
                         bm25_retrieved, emb_retrieved)
    write_similarity_examples(retriever)
    plot_bm25_vs_embedding(bm25_metrics, emb_metrics)

    print("")
    print("WEEK 2 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
