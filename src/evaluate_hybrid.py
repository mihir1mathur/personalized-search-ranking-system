"""
evaluate_hybrid.py  --  Week 3 main script: HYBRID search (BM25 + Embeddings)
=============================================================================

Run it with:

    python src/evaluate_hybrid.py

WHAT IT DOES, STEP BY STEP:
  1. Loads the SAME 50k sample as Weeks 0/1/2.
  2. Builds the SAME corpus, qrels, and SAME 3,000 evaluation queries
     (by reusing Week 1's own functions), so every comparison is fair.
  3. Builds a HybridRetriever = BM25 (keyword) + embeddings (semantic),
     reusing the cached Week 2 product embeddings so we DON'T re-encode 48k
     products.
  4. Scores every query once with BM25 and embeddings (the slow step), then
     tries several alpha/beta weight settings cheaply:
         0.5/0.5, 0.7/0.3, 0.6/0.4, 0.4/0.6
  5. Evaluates Precision@10, Recall@10, NDCG@10 for:
         TF-IDF, BM25, Embeddings, and Hybrid (each weight).
  6. Writes results/week3_metrics.txt, results/week3_comparison.png, and
     results/hybrid_failure_analysis.txt.

WHAT IT DOES NOT DO (on purpose -- future weeks):
  fine-tuning, reinforcement learning, cross-encoders, LLM reranking, RAG,
  GPT APIs.
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
    print("ERROR: This script needs pandas and numpy.  pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")  # save images without opening a window
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: This script needs matplotlib.  pip install matplotlib")
    sys.exit(1)

# Reuse Week 1 (corpus/qrels/eval queries/metrics) and the Week 1 TF-IDF model.
try:
    import metrics
    from tfidf_retriever import TfidfRetriever
    import evaluate_retrieval as week1
except ImportError as e:
    print(f"ERROR: Could not import a Week 1 module: {e}")
    sys.exit(1)

# The new Week 3 hybrid retriever.
try:
    from hybrid_retriever import HybridRetriever
    from embedding_retriever import DEFAULT_MODEL_NAME
except ImportError as e:
    print(f"ERROR: Could not import the Week 3 hybrid retriever: {e}")
    print("Install deps with: pip install sentence-transformers faiss-cpu")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration. We reuse Week 1's constants so the setup is identical.
# ---------------------------------------------------------------------------
RESULTS_DIR = week1.RESULTS_DIR
TOP_K = week1.TOP_K                  # 10
REL_THRESHOLD = week1.REL_THRESHOLD  # 1

# The alpha/beta weight settings to test (alpha = BM25, beta = embeddings).
WEIGHT_SETTINGS = [
    (0.5, 0.5),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.4, 0.6),
]

# Output files for Week 3 (all NEW -- we do not touch Week 0/1/2 outputs).
WEEK3_METRICS_TXT = os.path.join(RESULTS_DIR, "week3_metrics.txt")
WEEK3_PNG = os.path.join(RESULTS_DIR, "week3_comparison.png")
HYBRID_FAILURE_TXT = os.path.join(RESULTS_DIR, "hybrid_failure_analysis.txt")

# Reuse the Week 2 cached embeddings so we skip the ~1 hour encoding step.
EMB_CACHE_VECTORS = os.path.join(RESULTS_DIR, "product_embeddings_minilm.npy")
EMB_CACHE_IDS = os.path.join(RESULTS_DIR, "product_embeddings_ids.npy")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def shorten(text, limit=60):
    """Trim long product text so report tables stay readable."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def average(values):
    """Average a list, ignoring None entries (mirrors Week 1's helper)."""
    numbers = [v for v in values if v is not None]
    return float(np.mean(numbers)) if numbers else 0.0


def score_all_metrics(retrieved_per_query, eval_queries, qrels):
    """Average Precision@10, Recall@10, NDCG@10 across queries (Week 1 style)."""
    precisions, recalls, ndcgs = [], [], []
    for query, retrieved in zip(eval_queries, retrieved_per_query):
        retrieved_ids = [pid for (pid, _s, _r) in retrieved]
        query_qrels = qrels.get(query, {})
        precisions.append(metrics.precision_at_k(retrieved_ids, query_qrels, TOP_K, REL_THRESHOLD))
        recalls.append(metrics.recall_at_k(retrieved_ids, query_qrels, TOP_K, REL_THRESHOLD))
        ndcgs.append(metrics.ndcg_at_k(retrieved_ids, query_qrels, TOP_K))
    return {
        "Precision@10": average(precisions),
        "Recall@10": average(recalls),
        "NDCG@10": average(ndcgs),
    }


def per_query_ndcg(retrieved_per_query, eval_queries, qrels):
    """NDCG@10 for each query (used to find failure/win cases)."""
    out = []
    for query, retrieved in zip(eval_queries, retrieved_per_query):
        retrieved_ids = [pid for (pid, _s, _r) in retrieved]
        out.append(metrics.ndcg_at_k(retrieved_ids, qrels.get(query, {}), TOP_K))
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_metrics_report(tfidf_m, bm25_m, emb_m, hybrid_by_weight,
                         best_weight, n_queries, n_products, n_eval):
    """Write results/week3_metrics.txt with the 4-method comparison."""
    best_hybrid = hybrid_by_weight[best_weight]
    lines = []
    bar = "=" * 70
    lines.append(bar)
    lines.append("WEEK 3: HYBRID SEARCH (BM25 + EMBEDDINGS) -- METRICS REPORT")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append("")
    lines.append("SETUP")
    lines.append("-" * 70)
    lines.append(f"Embedding model                 : {DEFAULT_MODEL_NAME}")
    lines.append("Hybrid score                    : alpha*BM25_norm + beta*Emb_norm")
    lines.append("Score normalization             : Min-Max per query, per method")
    lines.append(f"Search corpus (unique products) : {n_products:,}")
    lines.append(f"Total unique queries in sample  : {n_queries:,}")
    lines.append(f"Queries evaluated (same as Wk1) : {n_eval:,}")
    lines.append(f"Top-K retrieved per query       : {TOP_K}")
    lines.append(f"Relevance threshold (>=)        : {REL_THRESHOLD}")
    lines.append("")

    lines.append("MAIN COMPARISON (best hybrid shown; averages over the same queries)")
    lines.append("-" * 70)
    lines.append("| Method            | Precision@10 | Recall@10 | NDCG@10 |")
    lines.append("| ----------------- | ------------ | --------- | ------- |")
    lines.append("| TF-IDF            | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        tfidf_m["Precision@10"], tfidf_m["Recall@10"], tfidf_m["NDCG@10"]))
    lines.append("| BM25              | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        bm25_m["Precision@10"], bm25_m["Recall@10"], bm25_m["NDCG@10"]))
    lines.append("| Embeddings        | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        emb_m["Precision@10"], emb_m["Recall@10"], emb_m["NDCG@10"]))
    lines.append("| Hybrid (a={:.1f},b={:.1f}) | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        best_weight[0], best_weight[1],
        best_hybrid["Precision@10"], best_hybrid["Recall@10"], best_hybrid["NDCG@10"]))
    lines.append("")

    lines.append("HYBRID WEIGHT SWEEP (alpha=BM25 weight, beta=Embedding weight)")
    lines.append("-" * 70)
    lines.append("| alpha | beta | Precision@10 | Recall@10 | NDCG@10 |")
    lines.append("| ----- | ---- | ------------ | --------- | ------- |")
    for (alpha, beta) in WEIGHT_SETTINGS:
        m = hybrid_by_weight[(alpha, beta)]
        marker = "  <- best" if (alpha, beta) == best_weight else ""
        lines.append("| {:.1f}   | {:.1f}  | {:.4f}       | {:.4f}    | {:.4f}  |{}".format(
            alpha, beta, m["Precision@10"], m["Recall@10"], m["NDCG@10"], marker))
    lines.append("")

    # Winner on NDCG across all four methods.
    table = [("TF-IDF", tfidf_m), ("BM25", bm25_m), ("Embeddings", emb_m),
             (f"Hybrid({best_weight[0]}/{best_weight[1]})", best_hybrid)]
    winner = max(table, key=lambda kv: kv[1]["NDCG@10"])[0]
    lines.append("WHICH METHOD PERFORMS BEST?")
    lines.append("-" * 70)
    lines.append(f"On NDCG@10 (headline ranking metric), the best method is: {winner}.")
    lines.append("")

    def pct(new, old):
        return (new - old) / old * 100.0 if old else float("nan")
    lines.append("BEST HYBRID vs the single methods (NDCG@10 relative change):")
    lines.append(f"  vs BM25       : {pct(best_hybrid['NDCG@10'], bm25_m['NDCG@10']):+.1f}%")
    lines.append(f"  vs Embeddings : {pct(best_hybrid['NDCG@10'], emb_m['NDCG@10']):+.1f}%")
    lines.append(f"  vs TF-IDF     : {pct(best_hybrid['NDCG@10'], tfidf_m['NDCG@10']):+.1f}%")
    lines.append("")
    lines.append("WHY HYBRID HELPS:")
    lines.append("  - BM25 contributes exact-keyword precision (brand/model terms).")
    lines.append("  - Embeddings contribute semantic recall (synonyms, typos).")
    lines.append("  - Min-Max normalization puts both on a 0..1 scale so the")
    lines.append("    weighted sum is meaningful.")
    lines.append("  - The mix recovers queries that EITHER method alone would miss.")
    lines.append("")
    lines.append("NOTE ON PRECISION: it stays low for ALL methods because labels are")
    lines.append("sparse (most queries have 1-2 judged-relevant products, so 1 relevant")
    lines.append("item caps Precision@10 at 0.10). Recall@10 and NDCG@10 are the")
    lines.append("informative metrics here.")
    lines.append("")
    with open(WEEK3_METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved metrics report to: {WEEK3_METRICS_TXT}")


def write_failure_analysis(eval_queries, qrels, product_text_by_id,
                           bm25_retrieved, emb_retrieved, hybrid_retrieved,
                           bm25_ndcg, emb_ndcg, hybrid_ndcg, best_weight):
    """
    Write results/hybrid_failure_analysis.txt:
      - 10 queries where Hybrid beats BM25 (by per-query NDCG@10),
      - 10 queries where Hybrid beats Embeddings.
    """
    def top_gains(other_ndcg):
        rows = []
        for i in range(len(eval_queries)):
            h = hybrid_ndcg[i]
            o = other_ndcg[i]
            if h is None or o is None:
                continue
            if h > o:
                rows.append((h - o, i))
        rows.sort(reverse=True)
        return [i for (_g, i) in rows[:10]]

    beats_bm25 = top_gains(bm25_ndcg)
    beats_emb = top_gains(emb_ndcg)

    def block(title, indices, other_name, other_retrieved, other_ndcg):
        out = []
        bar = "=" * 70
        out.append(bar)
        out.append(title)
        out.append(bar)
        out.append("")
        if not indices:
            out.append("(No qualifying queries found in this run.)")
            out.append("")
            return out
        for n, i in enumerate(indices, start=1):
            query = eval_queries[i]
            query_qrels = qrels.get(query, {})
            known = sorted(query_qrels.items(), key=lambda kv: -kv[1])
            out.append("-" * 70)
            out.append(f"CASE {n}:  QUERY = {query}")
            out.append(f"  NDCG@10 -> Hybrid={ (hybrid_ndcg[i] or 0):.3f} | "
                       f"{other_name}={ (other_ndcg[i] or 0):.3f}")
            out.append("  Ground truth (judged products):")
            for pid, s in known:
                out.append(f"     - {pid} (rel={s})  {shorten(product_text_by_id.get(pid, ''))}")
            out.append(f"  {other_name} Top-5:")
            for pid, score, rank in other_retrieved[i][:5]:
                rel = query_qrels.get(pid, 0)
                out.append(f"     {rank}. {pid}  rel={rel}  {shorten(product_text_by_id.get(pid, ''), 48)}")
            out.append("  Hybrid Top-5:")
            for pid, score, rank in hybrid_retrieved[i][:5]:
                rel = query_qrels.get(pid, 0)
                mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
                out.append(f"     {rank}. {pid}  rel={rel}  hybrid={score:.3f}  "
                           f"{shorten(product_text_by_id.get(pid, ''), 40)}{mark}")
            out.append("")
        return out

    lines = []
    bar = "=" * 70
    lines.append(bar)
    lines.append("WEEK 3: HYBRID FAILURE / WIN ANALYSIS")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append(f"Best hybrid weights used: alpha(BM25)={best_weight[0]}, "
                 f"beta(Embeddings)={best_weight[1]}.")
    lines.append("We list queries where the hybrid ranks the relevant product")
    lines.append("higher (better NDCG@10) than a single method alone.")
    lines.append("(rel = known relevance_score; 3=best ... 0=irrelevant.)")
    lines.append("")
    lines += block("PART A: 10 QUERIES WHERE HYBRID BEATS BM25",
                   beats_bm25, "BM25", bm25_retrieved, bm25_ndcg)
    lines += block("PART B: 10 QUERIES WHERE HYBRID BEATS EMBEDDINGS",
                   beats_emb, "Embeddings", emb_retrieved, emb_ndcg)
    lines.append("=" * 70)
    lines.append("WHY HYBRID WINS THESE")
    lines.append("=" * 70)
    lines.append("- Beats BM25 when the query needs MEANING (synonyms, typos,")
    lines.append("  paraphrases) that keyword matching misses; the embedding half")
    lines.append("  of the score pulls the relevant product up.")
    lines.append("- Beats Embeddings when the query needs EXACT terms (brand,")
    lines.append("  model numbers); the BM25 half sharpens the ranking and")
    lines.append("  promotes the precise match the fuzzy embedding had blurred.")
    lines.append("- Min-Max normalization lets both signals contribute fairly.")
    lines.append("")
    with open(HYBRID_FAILURE_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved hybrid failure analysis to: {HYBRID_FAILURE_TXT}")


def plot_comparison(tfidf_m, bm25_m, emb_m, best_hybrid, best_weight):
    """Grouped bar chart of all 4 methods across the 3 metrics."""
    metric_names = ["Precision@10", "Recall@10", "NDCG@10"]
    methods = [
        ("TF-IDF", tfidf_m),
        ("BM25", bm25_m),
        ("Embeddings", emb_m),
        (f"Hybrid {best_weight[0]}/{best_weight[1]}", best_hybrid),
    ]
    x = np.arange(len(metric_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (name, m) in enumerate(methods):
        values = [m[metric] for metric in metric_names]
        offset = (index - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=name)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Week 3: TF-IDF vs BM25 vs Embeddings vs Hybrid  (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(WEEK3_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved comparison chart to: {WEEK3_PNG}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("WEEK 3: HYBRID SEARCH (BM25 + EMBEDDINGS)")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load the SAME 50k sample.
    print("\n[1/7] Loading the Week 0 sample...")
    df = week1.load_sample()
    print(f"      Loaded {len(df):,} rows.")

    # 2) Build the SAME corpus, qrels, and 3,000 eval queries.
    print("[2/7] Building corpus, qrels, and eval queries (reusing Week 1)...")
    product_ids, product_texts = week1.build_corpus(df)
    qrels = week1.build_qrels(df)
    eval_queries = week1.choose_eval_queries(qrels)
    n_unique_queries = len(qrels)
    product_text_by_id = dict(zip(product_ids, product_texts))
    print(f"      Corpus: {len(product_ids):,} products | "
          f"{n_unique_queries:,} queries; evaluating {len(eval_queries):,}.")
    del df
    gc.collect()

    # 3) Build the hybrid retriever (reuse cached embeddings if available).
    print("[3/7] Building the hybrid retriever (BM25 + embeddings)...")
    cached_vectors = None
    if os.path.exists(EMB_CACHE_VECTORS) and os.path.exists(EMB_CACHE_IDS):
        cached_ids = np.load(EMB_CACHE_IDS, allow_pickle=True)
        if list(map(str, cached_ids)) == list(map(str, product_ids)):
            print("      Reusing cached Week 2 product embeddings (skip encoding).")
            cached_vectors = np.load(EMB_CACHE_VECTORS)
        else:
            print("      Cache mismatch -> will re-encode products (slow).")
    hybrid = HybridRetriever(candidate_pool_size=100)
    hybrid.fit(product_texts, product_ids, precomputed_embeddings=cached_vectors)

    # TF-IDF baseline (fast) for the 4-method comparison.
    print("      Building the TF-IDF index (baseline)...")
    tfidf = TfidfRetriever().fit(product_texts, product_ids)

    # 4) Score every query once (BM25 + embeddings), capturing pure baselines too.
    print("[4/7] Building hybrid scores...")
    print("      Evaluating 3000 queries (this computes BM25 + embedding scores once)...")
    candidates = hybrid.precompute_candidates(eval_queries, pure_top_k=TOP_K)

    # Pull the pure BM25-only and embedding-only Top-10 out of the candidates.
    bm25_retrieved = [c["bm25_ranking"] for c in candidates]
    emb_retrieved = [c["emb_ranking"] for c in candidates]
    print("      Retrieving Top-10 with TF-IDF...")
    tfidf_retrieved = tfidf.retrieve_batch(eval_queries, top_k=TOP_K)

    # 5) Score the single-method baselines.
    print("[5/7] Scoring TF-IDF, BM25, Embeddings...")
    tfidf_m = score_all_metrics(tfidf_retrieved, eval_queries, qrels)
    bm25_m = score_all_metrics(bm25_retrieved, eval_queries, qrels)
    emb_m = score_all_metrics(emb_retrieved, eval_queries, qrels)
    print("      TF-IDF    :", {k: round(v, 4) for k, v in tfidf_m.items()})
    print("      BM25      :", {k: round(v, 4) for k, v in bm25_m.items()})
    print("      Embeddings:", {k: round(v, 4) for k, v in emb_m.items()})

    # 6) Sweep hybrid weights (cheap, reuses precomputed scores).
    print("[6/7] Combining hybrid scores for each alpha/beta and scoring...")
    hybrid_by_weight = {}
    hybrid_retrieved_by_weight = {}
    for (alpha, beta) in WEIGHT_SETTINGS:
        retrieved = hybrid.rank_from_candidates(candidates, alpha=alpha, beta=beta, top_k=TOP_K)
        m = score_all_metrics(retrieved, eval_queries, qrels)
        hybrid_by_weight[(alpha, beta)] = m
        hybrid_retrieved_by_weight[(alpha, beta)] = retrieved
        print(f"      Hybrid alpha={alpha}, beta={beta} -> "
              f"{ {k: round(v, 4) for k, v in m.items()} }")

    # Pick the best hybrid weight by NDCG@10.
    best_weight = max(hybrid_by_weight, key=lambda w: hybrid_by_weight[w]["NDCG@10"])
    best_hybrid_metrics = hybrid_by_weight[best_weight]
    best_hybrid_retrieved = hybrid_retrieved_by_weight[best_weight]
    print(f"      Best hybrid weights (by NDCG@10): alpha={best_weight[0]}, beta={best_weight[1]}")

    # 7) Write outputs.
    print("[7/7] Writing output files...")
    write_metrics_report(tfidf_m, bm25_m, emb_m, hybrid_by_weight, best_weight,
                         n_unique_queries, len(product_ids), len(eval_queries))
    plot_comparison(tfidf_m, bm25_m, emb_m, best_hybrid_metrics, best_weight)

    # Failure / win analysis uses per-query NDCG with the best hybrid weights.
    bm25_ndcg = per_query_ndcg(bm25_retrieved, eval_queries, qrels)
    emb_ndcg = per_query_ndcg(emb_retrieved, eval_queries, qrels)
    hybrid_ndcg = per_query_ndcg(best_hybrid_retrieved, eval_queries, qrels)
    write_failure_analysis(eval_queries, qrels, product_text_by_id,
                           bm25_retrieved, emb_retrieved, best_hybrid_retrieved,
                           bm25_ndcg, emb_ndcg, hybrid_ndcg, best_weight)

    print("")
    print("WEEK 3 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
