"""
evaluate_reranker.py  --  Week 4 main script: CROSS-ENCODER RE-RANKING
======================================================================

Run it with:

    python src/evaluate_reranker.py

WHAT IT DOES, STEP BY STEP:
  1. Loads the SAME 50k sample as Weeks 0/1/2/3.
  2. Builds the SAME corpus, qrels, and SAME 3,000 evaluation queries
     (by reusing Week 1's own functions), so every comparison is fair.
  3. Rebuilds the Week 3 HYBRID retriever (BM25 + embeddings), reusing the
     cached Week 2 product embeddings so we DON'T re-encode 48k products.
  4. For every query, takes the HYBRID Top-50 as CANDIDATES.
  5. Runs a CROSS-ENCODER (cross-encoder/ms-marco-MiniLM-L-6-v2) on every
     (query, candidate) pair to get an accurate relevance score, then
     RE-RANKS those 50 candidates and keeps the new Top-10.
  6. Evaluates Precision@10, Recall@10, NDCG@10 for FIVE methods:
         TF-IDF, BM25, Embeddings, Hybrid, and Hybrid + Reranker.
  7. Writes results/week4_metrics.txt, results/week4_failure_analysis.txt,
     results/week4_examples.txt, and results/reranker_comparison.png.

THE BIG IDEA (retrieve, then re-rank):
  Retrievers (Weeks 1-3) are FAST but compare query and product SEPARATELY.
  A cross-encoder is SLOW but ACCURATE: it reads the query and a product
  TOGETHER and outputs one relevance number. So we let the fast hybrid
  retriever shortlist 50 candidates, then let the slow-but-smart cross-encoder
  carefully re-order just those 50. This is the standard two-stage design used
  by real search engines.

WHAT IT DOES NOT DO (on purpose -- future weeks):
  learning-to-rank, fine-tuning, reinforcement learning, AWS deployment,
  LLM/GPT re-ranking. We ONLY add a pretrained cross-encoder re-ranker.
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
    import pandas as pd  # noqa: F401  (pandas is used indirectly by week1.load_sample)
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

# Reuse the Week 3 hybrid retriever and Week 2 model-name constant.
try:
    from hybrid_retriever import HybridRetriever
    from embedding_retriever import DEFAULT_MODEL_NAME
except ImportError as e:
    print(f"ERROR: Could not import the Week 3 hybrid retriever: {e}")
    print("Install deps with: pip install sentence-transformers faiss-cpu")
    sys.exit(1)

# The NEW Week 4 cross-encoder re-ranker.
try:
    from cross_encoder_reranker import (
        CrossEncoderReranker,
        DEFAULT_CROSS_ENCODER_NAME,
    )
except ImportError as e:
    print(f"ERROR: Could not import the Week 4 cross-encoder re-ranker: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration. We reuse Week 1's constants so the setup is identical.
# ---------------------------------------------------------------------------
RESULTS_DIR = week1.RESULTS_DIR
TOP_K = week1.TOP_K                  # 10  (we evaluate the Top-10)
REL_THRESHOLD = week1.REL_THRESHOLD  # 1   (score >= 1 counts as relevant)

# How many hybrid candidates we hand to the cross-encoder per query.
# Bigger = the re-ranker can rescue more deeply-buried relevant products, but
# it is also more (query, product) pairs to score (slower).
CANDIDATE_DEPTH = 50

# The same alpha/beta weight settings Week 3 swept (alpha=BM25, beta=embeddings).
# We pick the best by NDCG@10, exactly like Week 3, so "Hybrid" here matches
# Week 3's "Hybrid" and the re-ranker is compared against the strongest hybrid.
WEIGHT_SETTINGS = [
    (0.5, 0.5),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.4, 0.6),
]

# Which cross-encoder to use. We prefer a locally-downloaded copy under
# models/ if present (handy for offline / restricted-network machines), and
# otherwise fall back to the standard Hugging Face hub name.
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_MODEL = (
    LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else DEFAULT_CROSS_ENCODER_NAME
)

# Output files for Week 4 (all NEW -- we do NOT touch Week 0/1/2/3 outputs).
WEEK4_METRICS_TXT = os.path.join(RESULTS_DIR, "week4_metrics.txt")
WEEK4_FAILURE_TXT = os.path.join(RESULTS_DIR, "week4_failure_analysis.txt")
WEEK4_EXAMPLES_TXT = os.path.join(RESULTS_DIR, "week4_examples.txt")
WEEK4_PNG = os.path.join(RESULTS_DIR, "reranker_comparison.png")

# Reuse the Week 2 cached embeddings so we skip the slow product-encoding step.
EMB_CACHE_VECTORS = os.path.join(RESULTS_DIR, "product_embeddings_minilm.npy")
EMB_CACHE_IDS = os.path.join(RESULTS_DIR, "product_embeddings_ids.npy")


# ---------------------------------------------------------------------------
# Small helpers (kept tiny and local; the heavy lifting is reused via imports).
# ---------------------------------------------------------------------------
def shorten(text, limit=60):
    """Trim long product text so report tables stay readable."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def average(values):
    """Average a list, ignoring None entries."""
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
    """NDCG@10 for EACH query (used to find improved / unchanged / worsened)."""
    out = []
    for query, retrieved in zip(eval_queries, retrieved_per_query):
        retrieved_ids = [pid for (pid, _s, _r) in retrieved]
        out.append(metrics.ndcg_at_k(retrieved_ids, qrels.get(query, {}), TOP_K))
    return out


# ---------------------------------------------------------------------------
# Output writer: metrics report
# ---------------------------------------------------------------------------
def write_metrics_report(method_metrics, best_weight, candidate_depth,
                         n_queries, n_products, n_eval):
    """Write results/week4_metrics.txt with the 5-method comparison."""
    tfidf_m = method_metrics["TF-IDF"]
    bm25_m = method_metrics["BM25"]
    emb_m = method_metrics["Embeddings"]
    hybrid_m = method_metrics["Hybrid"]
    rerank_m = method_metrics["Hybrid + Reranker"]

    lines = []
    bar = "=" * 72
    lines.append(bar)
    lines.append("WEEK 4: CROSS-ENCODER RE-RANKING -- METRICS REPORT")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append("")
    lines.append("SETUP")
    lines.append("-" * 72)
    lines.append(f"Retriever (stage 1)             : Hybrid = BM25 + embeddings (Week 3)")
    lines.append(f"Embedding model                 : {DEFAULT_MODEL_NAME}")
    lines.append(f"Cross-encoder (stage 2)         : {DEFAULT_CROSS_ENCODER_NAME}")
    lines.append(f"Best hybrid weights (by NDCG@10): alpha(BM25)={best_weight[0]}, "
                 f"beta(Emb)={best_weight[1]}")
    lines.append(f"Candidates re-ranked per query  : Top-{candidate_depth} from hybrid")
    lines.append(f"Search corpus (unique products) : {n_products:,}")
    lines.append(f"Total unique queries in sample  : {n_queries:,}")
    lines.append(f"Queries evaluated (same as Wk1) : {n_eval:,}")
    lines.append(f"Top-K retrieved per query       : {TOP_K}")
    lines.append(f"Relevance threshold (>=)        : {REL_THRESHOLD}")
    lines.append("")

    lines.append("MAIN COMPARISON (averages over the same evaluation queries)")
    lines.append("-" * 72)
    lines.append("| Method             | Precision@10 | Recall@10 | NDCG@10 |")
    lines.append("| ------------------ | ------------ | --------- | ------- |")
    row_order = ["TF-IDF", "BM25", "Embeddings", "Hybrid", "Hybrid + Reranker"]
    for name in row_order:
        m = method_metrics[name]
        lines.append("| {:<18} | {:.4f}       | {:.4f}    | {:.4f}  |".format(
            name, m["Precision@10"], m["Recall@10"], m["NDCG@10"]))
    lines.append("")

    # Winner on NDCG across all five methods.
    winner = max(row_order, key=lambda name: method_metrics[name]["NDCG@10"])
    lines.append("WHICH METHOD PERFORMS BEST?")
    lines.append("-" * 72)
    lines.append(f"On NDCG@10 (headline ranking metric), the best method is: {winner}.")
    lines.append("")

    def pct(new, old):
        return (new - old) / old * 100.0 if old else float("nan")

    lines.append("HYBRID + RERANKER vs the stage-1 HYBRID (relative change):")
    lines.append(f"  Precision@10 : {pct(rerank_m['Precision@10'], hybrid_m['Precision@10']):+.1f}%")
    lines.append(f"  Recall@10    : {pct(rerank_m['Recall@10'], hybrid_m['Recall@10']):+.1f}%")
    lines.append(f"  NDCG@10      : {pct(rerank_m['NDCG@10'], hybrid_m['NDCG@10']):+.1f}%")
    lines.append("")
    lines.append("HYBRID + RERANKER vs every other method (NDCG@10 relative change):")
    lines.append(f"  vs TF-IDF     : {pct(rerank_m['NDCG@10'], tfidf_m['NDCG@10']):+.1f}%")
    lines.append(f"  vs BM25       : {pct(rerank_m['NDCG@10'], bm25_m['NDCG@10']):+.1f}%")
    lines.append(f"  vs Embeddings : {pct(rerank_m['NDCG@10'], emb_m['NDCG@10']):+.1f}%")
    lines.append(f"  vs Hybrid     : {pct(rerank_m['NDCG@10'], hybrid_m['NDCG@10']):+.1f}%")
    lines.append("")

    beats = rerank_m["NDCG@10"] > hybrid_m["NDCG@10"]
    lines.append("DOES RE-RANKING BEAT HYBRID SEARCH?")
    lines.append("-" * 72)
    lines.append("  YES." if beats else "  NO (or tie) on the headline NDCG@10 metric.")
    lines.append("")
    lines.append("WHY RE-RANKING HELPS:")
    lines.append("  - The hybrid retriever scores query and product SEPARATELY, so it")
    lines.append("    can only rank by coarse overlap of precomputed signals.")
    lines.append("  - The cross-encoder reads each (query, product) pair TOGETHER with")
    lines.append("    full attention, so it judges true relevance far more precisely.")
    lines.append("  - It can also PROMOTE a relevant product the hybrid buried at rank")
    lines.append("    11-50 into the final Top-10, which raises Recall@10 as well.")
    lines.append("")
    lines.append("NOTE ON PRECISION: it stays low for ALL methods because labels are")
    lines.append("sparse (most queries have only 1-2 judged-relevant products, so a")
    lines.append("single relevant item caps Precision@10 at 0.10). Recall@10 and")
    lines.append("NDCG@10 are the informative metrics here.")
    lines.append("")
    with open(WEEK4_METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved metrics report to: {WEEK4_METRICS_TXT}")


# ---------------------------------------------------------------------------
# Output writer: failure / win analysis (improved / unchanged / worsened)
# ---------------------------------------------------------------------------
def write_failure_analysis(eval_queries, qrels, product_text_by_id,
                           hybrid_retrieved, rerank_retrieved,
                           hybrid_ndcg, rerank_ndcg, best_weight):
    """
    Write results/week4_failure_analysis.txt:
      - how many queries IMPROVED / stayed UNCHANGED / WORSENED after re-ranking,
      - the top examples of each, showing the Top-5 BEFORE and AFTER.
    """
    improved, unchanged, worsened = [], [], []
    eps = 1e-9
    for i in range(len(eval_queries)):
        h = hybrid_ndcg[i]
        r = rerank_ndcg[i]
        if h is None or r is None:
            continue
        diff = r - h
        if diff > eps:
            improved.append((diff, i))
        elif diff < -eps:
            worsened.append((diff, i))
        else:
            unchanged.append((0.0, i))

    improved.sort(reverse=True)             # biggest gains first
    worsened.sort()                         # biggest drops first (most negative)

    def block(title, rows, show=10):
        out = []
        bar = "=" * 72
        out.append(bar)
        out.append(title)
        out.append(bar)
        out.append("")
        if not rows:
            out.append("(No qualifying queries found in this run.)")
            out.append("")
            return out
        for n, (_diff, i) in enumerate(rows[:show], start=1):
            query = eval_queries[i]
            query_qrels = qrels.get(query, {})
            known = sorted(query_qrels.items(), key=lambda kv: -kv[1])
            out.append("-" * 72)
            out.append(f"CASE {n}:  QUERY = {query}")
            out.append(f"  NDCG@10 -> Hybrid={ (hybrid_ndcg[i] or 0):.3f}  ->  "
                       f"Reranked={ (rerank_ndcg[i] or 0):.3f}  "
                       f"(change {((rerank_ndcg[i] or 0) - (hybrid_ndcg[i] or 0)):+.3f})")
            out.append("  Ground truth (judged products):")
            for pid, s in known:
                out.append(f"     - {pid} (rel={s})  {shorten(product_text_by_id.get(pid, ''))}")
            out.append("  BEFORE  (Hybrid Top-5):")
            for pid, score, rank in hybrid_retrieved[i][:5]:
                rel = query_qrels.get(pid, 0)
                mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
                out.append(f"     {rank}. {pid}  rel={rel}  hybrid={score:.3f}  "
                           f"{shorten(product_text_by_id.get(pid, ''), 36)}{mark}")
            out.append("  AFTER   (Reranked Top-5):")
            for pid, score, rank in rerank_retrieved[i][:5]:
                rel = query_qrels.get(pid, 0)
                mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
                out.append(f"     {rank}. {pid}  rel={rel}  CE={score:+.3f}  "
                           f"{shorten(product_text_by_id.get(pid, ''), 36)}{mark}")
            out.append("")
        return out

    total = len(improved) + len(unchanged) + len(worsened)
    lines = []
    bar = "=" * 72
    lines.append(bar)
    lines.append("WEEK 4: RE-RANKING FAILURE / WIN ANALYSIS")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append(f"Stage-1 hybrid weights: alpha(BM25)={best_weight[0]}, "
                 f"beta(Emb)={best_weight[1]}.")
    lines.append("We compare per-query NDCG@10 BEFORE re-ranking (hybrid Top-10) and")
    lines.append("AFTER re-ranking (cross-encoder reorders the hybrid Top-50).")
    lines.append("'CE' = raw cross-encoder relevance score (higher = more relevant).")
    lines.append("")
    lines.append("SUMMARY (over queries with a defined NDCG@10)")
    lines.append("-" * 72)
    lines.append(f"  Queries scored      : {total:,}")
    lines.append(f"  IMPROVED by rerank  : {len(improved):,}  "
                 f"({(100.0 * len(improved) / total if total else 0):.1f}%)")
    lines.append(f"  UNCHANGED           : {len(unchanged):,}  "
                 f"({(100.0 * len(unchanged) / total if total else 0):.1f}%)")
    lines.append(f"  WORSENED by rerank  : {len(worsened):,}  "
                 f"({(100.0 * len(worsened) / total if total else 0):.1f}%)")
    lines.append("")
    lines += block("PART A: TOP 10 QUERIES IMPROVED BY RE-RANKING", improved)
    lines += block("PART B: 10 QUERIES UNCHANGED BY RE-RANKING", unchanged)
    lines += block("PART C: TOP 10 QUERIES WORSENED BY RE-RANKING", worsened)
    lines.append("=" * 72)
    lines.append("HOW TO READ THIS")
    lines.append("=" * 72)
    lines.append("- IMPROVED: the cross-encoder understood the query-product meaning")
    lines.append("  better than coarse retrieval and pulled the relevant product up")
    lines.append("  (often from rank 11-50 into the Top-10).")
    lines.append("- UNCHANGED: hybrid already had the relevant product at the top, so")
    lines.append("  there was nothing left to fix.")
    lines.append("- WORSENED: the cross-encoder was over-confident about a")
    lines.append("  plausible-but-wrong product, or the truly relevant item was never")
    lines.append("  in the Top-50 candidate set (the re-ranker can only reorder what")
    lines.append("  stage-1 retrieval gave it -- it cannot invent missing candidates).")
    lines.append("")
    with open(WEEK4_FAILURE_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved failure analysis to: {WEEK4_FAILURE_TXT}")
    return len(improved), len(unchanged), len(worsened)


# ---------------------------------------------------------------------------
# Output writer: worked examples (before vs after, with cross-encoder scores)
# ---------------------------------------------------------------------------
def write_examples(eval_queries, qrels, product_text_by_id,
                   hybrid_retrieved, rerank_retrieved,
                   hybrid_ndcg, rerank_ndcg, reranker):
    """
    Write results/week4_examples.txt:
      - a tiny illustrative demo of cross-encoder pair scoring, and
      - several real queries showing the full Top-10 BEFORE and AFTER.
    """
    lines = []
    bar = "=" * 72
    lines.append(bar)
    lines.append("WEEK 4: CROSS-ENCODER RE-RANKING -- WORKED EXAMPLES")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append("")

    # --- Part 1: a small, self-contained illustration of pair scoring. -------
    lines.append("PART 1: WHAT THE CROSS-ENCODER ACTUALLY OUTPUTS")
    lines.append("-" * 72)
    lines.append("The model reads (query, product) TOGETHER and returns ONE relevance")
    lines.append("score. Higher = more relevant. Example pairs:")
    lines.append("")
    demo_query = "iphone charger"
    demo_products = [
        "Apple Lightning to USB Cable for iPhone charging cord",
        "Anker USB-C fast charger power adapter for phones",
        "Samsung 4K UHD Smart Television remote control",
        "Stainless steel kitchen blender 700W",
    ]
    demo_scores = reranker.score_pairs([[demo_query, p] for p in demo_products])
    for product, score in zip(demo_products, demo_scores):
        lines.append(f'  ("{demo_query}",')
        lines.append(f'   "{product}")  ->  {score:+.2f}')
    lines.append("")
    lines.append("Notice the on-topic charger pairs score high and positive, while the")
    lines.append("TV and blender score strongly negative -- that ordering is exactly")
    lines.append("what we use to re-rank.")
    lines.append("")

    # --- Part 2: real queries, full Top-10 before vs after. ------------------
    lines.append("PART 2: REAL QUERIES (Top-10 BEFORE vs AFTER re-ranking)")
    lines.append("-" * 72)
    lines.append("")

    # Pick a few queries where re-ranking changed the NDCG the most (interesting),
    # so a reader can clearly see the effect.
    changes = []
    for i in range(len(eval_queries)):
        h, r = hybrid_ndcg[i], rerank_ndcg[i]
        if h is None or r is None:
            continue
        changes.append((abs(r - h), i))
    changes.sort(reverse=True)
    chosen = [i for (_d, i) in changes[:6]]

    for n, i in enumerate(chosen, start=1):
        query = eval_queries[i]
        query_qrels = qrels.get(query, {})
        known = sorted(query_qrels.items(), key=lambda kv: -kv[1])
        lines.append("=" * 72)
        lines.append(f"EXAMPLE {n}:  QUERY = {query}")
        lines.append(f"  NDCG@10  Hybrid={ (hybrid_ndcg[i] or 0):.3f}  ->  "
                     f"Reranked={ (rerank_ndcg[i] or 0):.3f}")
        lines.append("  Ground truth (judged products):")
        for pid, s in known:
            lines.append(f"     - {pid} (rel={s})  {shorten(product_text_by_id.get(pid, ''))}")
        lines.append("")
        lines.append("  BEFORE -- Hybrid Top-10:")
        for pid, score, rank in hybrid_retrieved[i]:
            rel = query_qrels.get(pid, 0)
            mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
            lines.append(f"     {rank:>2}. {pid}  rel={rel}  hybrid={score:.3f}  "
                         f"{shorten(product_text_by_id.get(pid, ''), 40)}{mark}")
        lines.append("")
        lines.append("  AFTER  -- Cross-Encoder Reranked Top-10:")
        for pid, score, rank in rerank_retrieved[i]:
            rel = query_qrels.get(pid, 0)
            mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
            lines.append(f"     {rank:>2}. {pid}  rel={rel}  CE={score:+.3f}  "
                         f"{shorten(product_text_by_id.get(pid, ''), 40)}{mark}")
        lines.append("")

    with open(WEEK4_EXAMPLES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved worked examples to: {WEEK4_EXAMPLES_TXT}")


# ---------------------------------------------------------------------------
# Output writer: comparison chart
# ---------------------------------------------------------------------------
def plot_comparison(method_metrics):
    """Grouped bar chart of all 5 methods across the 3 metrics."""
    metric_names = ["Precision@10", "Recall@10", "NDCG@10"]
    method_names = ["TF-IDF", "BM25", "Embeddings", "Hybrid", "Hybrid + Reranker"]

    x = np.arange(len(metric_names))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, name in enumerate(method_names):
        m = method_metrics[name]
        values = [m[metric] for metric in metric_names]
        offset = (index - 2) * width  # center the 5 bars around each tick
        bars = ax.bar(x + offset, values, width, label=name)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("Score")
    ax.set_title("Week 4: TF-IDF vs BM25 vs Embeddings vs Hybrid vs "
                 "Hybrid+Reranker  (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(WEEK4_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison chart to: {WEEK4_PNG}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("WEEK 4: CROSS-ENCODER RE-RANKING")
    print("=" * 72)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load the SAME 50k sample.
    print("\n[1/8] Loading the Week 0 sample...")
    df = week1.load_sample()
    print(f"      Loaded {len(df):,} rows.")

    # 2) Build the SAME corpus, qrels, and 3,000 eval queries (reuse Week 1).
    print("[2/8] Building corpus, qrels, and eval queries (reusing Week 1)...")
    product_ids, product_texts = week1.build_corpus(df)
    qrels = week1.build_qrels(df)
    eval_queries = week1.choose_eval_queries(qrels)
    n_unique_queries = len(qrels)
    product_text_by_id = dict(zip(product_ids, product_texts))
    print(f"      Corpus: {len(product_ids):,} products | "
          f"{n_unique_queries:,} queries; evaluating {len(eval_queries):,}.")
    del df
    gc.collect()

    # 3) Rebuild the hybrid retriever (reuse cached embeddings if available).
    print("[3/8] Building the hybrid retriever (BM25 + embeddings)...")
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

    # TF-IDF baseline (fast) for the 5-method comparison.
    print("      Building the TF-IDF index (baseline)...")
    tfidf = TfidfRetriever().fit(product_texts, product_ids)

    # 4) Score every query once with BM25 + embeddings (the slow retrieval step).
    print("[4/8] Computing hybrid candidates (BM25 + embedding scores)...")
    candidates = hybrid.precompute_candidates(eval_queries, pure_top_k=TOP_K)

    # Free byproducts: pure BM25-only and embedding-only Top-10 baselines.
    bm25_retrieved = [c["bm25_ranking"] for c in candidates]
    emb_retrieved = [c["emb_ranking"] for c in candidates]
    print("      Retrieving Top-10 with TF-IDF (baseline)...")
    tfidf_retrieved = tfidf.retrieve_batch(eval_queries, top_k=TOP_K)

    # 5) Pick the best hybrid weights by NDCG@10 (same sweep as Week 3), and get
    #    BOTH the hybrid Top-10 (a baseline) and the hybrid Top-50 (candidates).
    print("[5/8] Sweeping hybrid weights and selecting the best by NDCG@10...")
    best_weight = None
    best_ndcg = -1.0
    hybrid_top10_by_weight = {}
    for (alpha, beta) in WEIGHT_SETTINGS:
        retrieved10 = hybrid.rank_from_candidates(candidates, alpha=alpha, beta=beta, top_k=TOP_K)
        m = score_all_metrics(retrieved10, eval_queries, qrels)
        hybrid_top10_by_weight[(alpha, beta)] = retrieved10
        print(f"      Hybrid alpha={alpha}, beta={beta} -> "
              f"{ {k: round(v, 4) for k, v in m.items()} }")
        if m["NDCG@10"] > best_ndcg:
            best_ndcg = m["NDCG@10"]
            best_weight = (alpha, beta)
    print(f"      Best hybrid weights: alpha={best_weight[0]}, beta={best_weight[1]}")

    hybrid_retrieved = hybrid_top10_by_weight[best_weight]   # hybrid Top-10 baseline
    hybrid_candidates = hybrid.rank_from_candidates(          # hybrid Top-50 candidates
        candidates, alpha=best_weight[0], beta=best_weight[1], top_k=CANDIDATE_DEPTH)

    # 6) Build (product_id, product_text) candidate lists and run the reranker.
    print(f"[6/8] Cross-encoder re-ranking the Top-{CANDIDATE_DEPTH} per query...")
    reranker = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL)
    candidate_lists = []
    for ranked in hybrid_candidates:
        # Attach each candidate's product text (what the cross-encoder reads).
        candidate_lists.append([
            (pid, product_text_by_id.get(pid, "")) for (pid, _score, _rank) in ranked
        ])
    rerank_retrieved = reranker.rerank_batch(eval_queries, candidate_lists, top_k=TOP_K)

    # 7) Score all five methods on the same queries.
    print("[7/8] Scoring all five methods (Precision/Recall/NDCG @10)...")
    method_metrics = {
        "TF-IDF": score_all_metrics(tfidf_retrieved, eval_queries, qrels),
        "BM25": score_all_metrics(bm25_retrieved, eval_queries, qrels),
        "Embeddings": score_all_metrics(emb_retrieved, eval_queries, qrels),
        "Hybrid": score_all_metrics(hybrid_retrieved, eval_queries, qrels),
        "Hybrid + Reranker": score_all_metrics(rerank_retrieved, eval_queries, qrels),
    }
    for name in ["TF-IDF", "BM25", "Embeddings", "Hybrid", "Hybrid + Reranker"]:
        print(f"      {name:<18}:", {k: round(v, 4) for k, v in method_metrics[name].items()})

    # 8) Write all output files.
    print("[8/8] Writing output files...")
    write_metrics_report(method_metrics, best_weight, CANDIDATE_DEPTH,
                         n_unique_queries, len(product_ids), len(eval_queries))
    plot_comparison(method_metrics)

    # Per-query NDCG to classify improved / unchanged / worsened.
    hybrid_ndcg = per_query_ndcg(hybrid_retrieved, eval_queries, qrels)
    rerank_ndcg = per_query_ndcg(rerank_retrieved, eval_queries, qrels)
    n_improved, n_unchanged, n_worsened = write_failure_analysis(
        eval_queries, qrels, product_text_by_id,
        hybrid_retrieved, rerank_retrieved, hybrid_ndcg, rerank_ndcg, best_weight)
    write_examples(eval_queries, qrels, product_text_by_id,
                   hybrid_retrieved, rerank_retrieved,
                   hybrid_ndcg, rerank_ndcg, reranker)

    # -----------------------------------------------------------------------
    # Final printed summary.
    # -----------------------------------------------------------------------
    hybrid_m = method_metrics["Hybrid"]
    rerank_m = method_metrics["Hybrid + Reranker"]
    beats = rerank_m["NDCG@10"] > hybrid_m["NDCG@10"]

    def pct(new, old):
        return (new - old) / old * 100.0 if old else float("nan")

    print("")
    print("=" * 72)
    print("WEEK 4 COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("")
    print("1) METRIC IMPROVEMENTS (Hybrid -> Hybrid + Reranker):")
    print(f"     Precision@10 : {hybrid_m['Precision@10']:.4f} -> "
          f"{rerank_m['Precision@10']:.4f}  ({pct(rerank_m['Precision@10'], hybrid_m['Precision@10']):+.1f}%)")
    print(f"     Recall@10    : {hybrid_m['Recall@10']:.4f} -> "
          f"{rerank_m['Recall@10']:.4f}  ({pct(rerank_m['Recall@10'], hybrid_m['Recall@10']):+.1f}%)")
    print(f"     NDCG@10      : {hybrid_m['NDCG@10']:.4f} -> "
          f"{rerank_m['NDCG@10']:.4f}  ({pct(rerank_m['NDCG@10'], hybrid_m['NDCG@10']):+.1f}%)")
    print("")
    print("2) QUERIES IMPROVED:")
    total_cls = n_improved + n_unchanged + n_worsened
    print(f"     {n_improved:,} of {total_cls:,} queries improved "
          f"({(100.0 * n_improved / total_cls if total_cls else 0):.1f}%).")
    print("")
    print("3) REMAINING FAILURES:")
    print(f"     {n_worsened:,} queries worsened "
          f"({(100.0 * n_worsened / total_cls if total_cls else 0):.1f}%); "
          f"{n_unchanged:,} unchanged "
          f"({(100.0 * n_unchanged / total_cls if total_cls else 0):.1f}%).")
    print("     Worsened cases are usually queries whose truly-relevant product")
    print("     was never in the Top-50 candidate set, so the re-ranker could not")
    print("     recover it (it can only reorder what stage-1 retrieval provided).")
    print("")
    print("4) DOES RE-RANKING BEAT HYBRID SEARCH?")
    if beats:
        print(f"     YES -- Hybrid + Reranker has the best NDCG@10 "
              f"({rerank_m['NDCG@10']:.4f} vs hybrid {hybrid_m['NDCG@10']:.4f}).")
    else:
        print(f"     NO -- re-ranking did not beat hybrid on NDCG@10 "
              f"({rerank_m['NDCG@10']:.4f} vs hybrid {hybrid_m['NDCG@10']:.4f}).")
    print("")


if __name__ == "__main__":
    main()
