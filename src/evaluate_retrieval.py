"""
evaluate_retrieval.py  --  Week 1 main script: build + compare TF-IDF and BM25
==============================================================================

This is the MAIN script for Week 1. Run it with:

    python src/evaluate_retrieval.py

WHAT IT DOES, STEP BY STEP:
  1. Loads the cleaned 50k sample from Week 0.
  2. Builds a product "corpus" = every unique product_text (the searchable set).
  3. Builds "qrels" = for each query, which products were judged and how relevant.
  4. Builds two retrievers over the corpus: TF-IDF and BM25.
  5. For a set of evaluation queries, retrieves the Top 10 products with each.
  6. Scores both methods with Precision@10, Recall@10, and NDCG@10.
  7. Writes all of the Week 1 output files (CSV, metrics, sample analysis, plots).

A NOTE ON THE EVALUATION SETUP (important to understand the numbers):
  - Our Week 0 balanced sample picked rows by relevance level, which split most
    queries across the file. So in this sample most queries have only 1-2 judged
    products. That is fine for "global" retrieval: we treat ALL unique products
    as the search corpus, retrieve the Top 10 for a query from the whole corpus,
    and judge a retrieved product using its known relevance_score (a product we
    have no judgment for is treated as irrelevant, score 0).
  - This is the standard way to evaluate retrieval when relevance labels are
    "sparse". Recall@10 and NDCG@10 are the most informative metrics here:
    they tell us whether the truly-relevant products are pulled into the Top 10
    and ranked near the top.

WHY WE EVALUATE ON A SAMPLE OF QUERIES:
  The BM25 library scores one query against all ~48k products at a time, so
  scoring every one of the 33k+ queries would take ~50 minutes. To keep Week 1
  fast and repeatable, we evaluate on a fixed RANDOM SAMPLE of queries (same
  sample every run, thanks to a fixed seed). Increase EVAL_QUERY_SAMPLE_SIZE
  below if you want to evaluate on more queries.
"""

import os
import sys

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

# Our own modules (created earlier in Week 1).
try:
    from tfidf_retriever import TfidfRetriever
    from bm25_retriever import BM25Retriever, simple_tokenize
    import metrics
except ImportError as e:
    print(f"ERROR: Could not import a Week 1 module: {e}")
    print("Make sure tfidf_retriever.py and bm25_retriever.py are in src/,")
    print("and metrics.py is in the evaluation/ folder.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration (easy knobs to tweak).
# ---------------------------------------------------------------------------
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "sample_esci_50k.parquet")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

RESULTS_CSV = os.path.join(RESULTS_DIR, "retrieval_results.csv")
METRICS_TXT = os.path.join(RESULTS_DIR, "week1_metrics.txt")
SAMPLE_ANALYSIS_TXT = os.path.join(RESULTS_DIR, "sample_query_analysis.txt")
METRICS_PNG = os.path.join(RESULTS_DIR, "tfidf_vs_bm25_metrics.png")
QLEN_PNG = os.path.join(RESULTS_DIR, "query_length_distribution.png")

TOP_K = 10                      # we retrieve and evaluate the Top 10
REL_THRESHOLD = 1               # score >= 1 counts as "relevant"
EVAL_QUERY_SAMPLE_SIZE = 3000   # how many queries to evaluate (set higher for more)
RANDOM_SEED = 42                # fixed seed -> same results every run


def load_sample():
    """Load the Week 0 sample, with clear errors if something is wrong."""
    if not os.path.exists(INPUT_PATH):
        print("ERROR: Could not find the Week 0 sample file at:")
        print(f"    {INPUT_PATH}")
        print("Please run Week 0 first to create data/processed/sample_esci_50k.parquet")
        sys.exit(1)
    try:
        df = pd.read_parquet(INPUT_PATH)
    except ImportError:
        print("ERROR: Reading parquet needs 'pyarrow'. Install with: pip install pyarrow")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read the sample file: {e}")
        sys.exit(1)

    # Make sure the columns we rely on actually exist.
    needed = ["query", "product_id", "product_text", "relevance_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: The sample is missing required columns: {missing}")
        print(f"Available columns are: {list(df.columns)}")
        sys.exit(1)
    return df


def build_corpus(df):
    """
    Build the searchable corpus = one row per unique product.
    Returns (product_ids list, product_texts list).
    """
    corpus_df = df.drop_duplicates(subset="product_id")[["product_id", "product_text"]]
    product_ids = corpus_df["product_id"].astype(str).tolist()
    product_texts = corpus_df["product_text"].astype(str).tolist()
    return product_ids, product_texts


def build_qrels(df):
    """
    Build the relevance judgments.
    Returns a dict: query -> {product_id: relevance_score}.
    This is our ground truth for scoring the retrieved results.
    """
    qrels = {}
    for query, pid, score in zip(
        df["query"].astype(str),
        df["product_id"].astype(str),
        df["relevance_score"].astype(int),
    ):
        # If the same (query, product) appears twice, keep the highest score.
        bucket = qrels.setdefault(query, {})
        if pid not in bucket or score > bucket[pid]:
            bucket[pid] = score
    return qrels


def choose_eval_queries(qrels):
    """
    Pick the queries we will evaluate on.
    We only keep queries that have at least one RELEVANT judged product, so that
    Recall@10 and NDCG@10 are well-defined. Then we randomly sample up to
    EVAL_QUERY_SAMPLE_SIZE of them using a fixed seed (for reproducibility).
    """
    eligible = [
        q for q, rels in qrels.items()
        if any(s >= REL_THRESHOLD for s in rels.values())
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    eligible = sorted(eligible)  # sort first so the random pick is reproducible
    if len(eligible) > EVAL_QUERY_SAMPLE_SIZE:
        idx = rng.choice(len(eligible), size=EVAL_QUERY_SAMPLE_SIZE, replace=False)
        eligible = [eligible[i] for i in sorted(idx)]
    return eligible


def average(values):
    """Average a list, ignoring None entries. Returns 0.0 if the list is empty."""
    nums = [v for v in values if v is not None]
    return float(np.mean(nums)) if nums else 0.0


def evaluate_method(method_name, retrieved_per_query, eval_queries, qrels):
    """
    Score one method (TF-IDF or BM25) and collect per-query metrics + result rows.

    retrieved_per_query : list aligned with eval_queries; each item is the
                          method's Top-K list of (product_id, score, rank).
    Returns (avg_metrics_dict, list_of_result_rows).
    """
    precisions, recalls, ndcgs = [], [], []
    result_rows = []

    for query, retrieved in zip(eval_queries, retrieved_per_query):
        retrieved_ids = [pid for (pid, _score, _rank) in retrieved]
        query_qrels = qrels.get(query, {})

        # Compute the three metrics for this query.
        precisions.append(metrics.precision_at_k(retrieved_ids, query_qrels, TOP_K, REL_THRESHOLD))
        recalls.append(metrics.recall_at_k(retrieved_ids, query_qrels, TOP_K, REL_THRESHOLD))
        ndcgs.append(metrics.ndcg_at_k(retrieved_ids, query_qrels, TOP_K))

        # Save every retrieved row so we can write the results CSV later.
        for pid, score, rank in retrieved:
            result_rows.append({
                "method": method_name,
                "query": query,
                "product_id": pid,
                "score": round(float(score), 6),
                "rank": rank,
                # Handy extra: the known relevance of this retrieved product.
                "relevance_score": query_qrels.get(pid, 0),
            })

    avg_metrics = {
        "Precision@10": average(precisions),
        "Recall@10": average(recalls),
        "NDCG@10": average(ndcgs),
    }
    return avg_metrics, result_rows


def write_metrics_report(tfidf_metrics, bm25_metrics, n_queries, n_products, n_eval):
    """Write results/week1_metrics.txt with the comparison table and explanations."""
    lines = []
    lines.append("=" * 64)
    lines.append("WEEK 1: RETRIEVAL BASELINES -- METRICS REPORT")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append("=" * 64)
    lines.append("")
    lines.append("SETUP")
    lines.append("-" * 64)
    lines.append(f"Search corpus (unique products): {n_products:,}")
    lines.append(f"Total unique queries in sample : {n_queries:,}")
    lines.append(f"Queries evaluated (random sample): {n_eval:,}")
    lines.append(f"Top-K retrieved per query        : {TOP_K}")
    lines.append(f"Relevance threshold (relevant if score >= ): {REL_THRESHOLD}")
    lines.append("Relevance scale: 3=highly relevant, 2=relevant, "
                 "1=weakly relevant, 0=irrelevant")
    lines.append("")

    lines.append("COMPARISON TABLE (averages across evaluated queries)")
    lines.append("-" * 64)
    lines.append("| Method | Precision@10 | Recall@10 | NDCG@10 |")
    lines.append("| ------ | ------------ | --------- | ------- |")
    lines.append("| TF-IDF | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        tfidf_metrics["Precision@10"], tfidf_metrics["Recall@10"], tfidf_metrics["NDCG@10"]))
    lines.append("| BM25   | {:.4f}       | {:.4f}    | {:.4f}  |".format(
        bm25_metrics["Precision@10"], bm25_metrics["Recall@10"], bm25_metrics["NDCG@10"]))
    lines.append("")

    # Decide which method won on NDCG (the headline ranking metric).
    if bm25_metrics["NDCG@10"] > tfidf_metrics["NDCG@10"]:
        winner = "BM25"
    elif tfidf_metrics["NDCG@10"] > bm25_metrics["NDCG@10"]:
        winner = "TF-IDF"
    else:
        winner = "Tie"

    lines.append("WHICH METHOD PERFORMS BETTER?")
    lines.append("-" * 64)
    lines.append(f"On NDCG@10 (our headline ranking metric), the better method is: {winner}.")
    lines.append("")
    lines.append("WHY BM25 OFTEN BEATS TF-IDF:")
    lines.append("  - Term-frequency saturation: BM25 stops rewarding a word after")
    lines.append("    it appears a few times, so 'keyword stuffed' products do not")
    lines.append("    dominate. Plain TF-IDF keeps rewarding repeats more linearly.")
    lines.append("  - Document-length normalization: BM25 fairly handles long product")
    lines.append("    descriptions, so a short, on-topic title can still win over a")
    lines.append("    long, loosely-related description.")
    lines.append("  - These two ideas usually make BM25 rank the truly relevant")
    lines.append("    products a little higher, which NDCG rewards.")
    lines.append("")
    lines.append("WHERE TF-IDF STILL WORKS WELL:")
    lines.append("  - Short, clean documents where length normalization matters less.")
    lines.append("  - When you need cosine similarity vectors for other tasks")
    lines.append("    (clustering, simple similarity, feeding another model).")
    lines.append("  - As a fast, easy-to-explain baseline that is built into scikit-learn.")
    lines.append("")
    lines.append("These keyword baselines are the bar that future dense-retrieval and")
    lines.append("transformer models (later weeks) must beat.")
    lines.append("")

    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved metrics report to: {METRICS_TXT}")


def write_sample_query_analysis(eval_queries, tfidf_retrieved, bm25_retrieved, qrels):
    """
    Pick 10 random evaluated queries and show the Top-5 results from each method,
    along with the known relevance score of each retrieved product.
    Saved to results/sample_query_analysis.txt.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = min(10, len(eval_queries))
    pick_idx = sorted(rng.choice(len(eval_queries), size=n, replace=False).tolist())

    lines = []
    lines.append("=" * 64)
    lines.append("WEEK 1: SAMPLE QUERY ANALYSIS (10 random queries)")
    lines.append("=" * 64)
    lines.append("For each query we show the Top-5 products from each method.")
    lines.append("'rel' is the known relevance_score (3=best ... 0=irrelevant;")
    lines.append(" a product with no judgment for this query shows rel=0).")
    lines.append("")

    for i in pick_idx:
        query = eval_queries[i]
        query_qrels = qrels.get(query, {})
        lines.append("-" * 64)
        lines.append(f"QUERY: {query}")
        # Show what we actually know is relevant for this query.
        known = sorted(query_qrels.items(), key=lambda kv: -kv[1])
        known_str = ", ".join(f"{pid}(rel={s})" for pid, s in known)
        lines.append(f"Known judged products: {known_str}")
        lines.append("")

        lines.append("  Top-5 TF-IDF:")
        for pid, score, rank in tfidf_retrieved[i][:5]:
            rel = query_qrels.get(pid, 0)
            lines.append(f"    {rank}. {pid}  score={score:.3f}  rel={rel}")

        lines.append("  Top-5 BM25:")
        for pid, score, rank in bm25_retrieved[i][:5]:
            rel = query_qrels.get(pid, 0)
            lines.append(f"    {rank}. {pid}  score={score:.3f}  rel={rel}")
        lines.append("")

    with open(SAMPLE_ANALYSIS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved sample query analysis to: {SAMPLE_ANALYSIS_TXT}")


def plot_metrics(tfidf_metrics, bm25_metrics):
    """Bar chart comparing TF-IDF vs BM25 across the three metrics."""
    metric_names = ["Precision@10", "Recall@10", "NDCG@10"]
    tfidf_vals = [tfidf_metrics[m] for m in metric_names]
    bm25_vals = [bm25_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, tfidf_vals, width, label="TF-IDF")
    bars2 = ax.bar(x + width / 2, bm25_vals, width, label="BM25")

    ax.set_ylabel("Score")
    ax.set_title("TF-IDF vs BM25  (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    # Write the value on top of each bar so the chart is easy to read.
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(METRICS_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved metrics chart to: {METRICS_PNG}")


def plot_query_length_distribution(df):
    """Histogram of how many words queries contain (over unique queries)."""
    unique_queries = df["query"].dropna().astype(str).drop_duplicates()
    lengths = unique_queries.str.split().apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    max_len = int(lengths.max())
    # One bin per word-count up to a sensible cap so the chart stays readable.
    bins = range(1, min(max_len, 15) + 2)
    ax.hist(lengths.clip(upper=15), bins=bins, edgecolor="black", align="left")
    ax.set_xlabel("Query length (number of words; 15 = '15 or more')")
    ax.set_ylabel("Number of queries")
    ax.set_title("Query Length Distribution")
    fig.tight_layout()
    fig.savefig(QLEN_PNG, dpi=120)
    plt.close(fig)
    print(f"Saved query length chart to: {QLEN_PNG}")


def main():
    print("=" * 64)
    print("WEEK 1: RETRIEVAL BASELINES (TF-IDF + BM25)")
    print("=" * 64)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load data.
    print("\n[1/7] Loading the Week 0 sample...")
    df = load_sample()
    print(f"      Loaded {len(df):,} rows.")

    # 2) Build the searchable corpus.
    print("[2/7] Building the product corpus (unique products)...")
    product_ids, product_texts = build_corpus(df)
    print(f"      Corpus has {len(product_ids):,} unique products.")

    # 3) Build the relevance judgments (ground truth).
    print("[3/7] Building relevance judgments (qrels)...")
    qrels = build_qrels(df)
    n_unique_queries = len(qrels)
    eval_queries = choose_eval_queries(qrels)
    print(f"      {n_unique_queries:,} unique queries; "
          f"evaluating {len(eval_queries):,} of them.")

    # 4) Fit both retrievers on the corpus.
    print("[4/7] Building the TF-IDF index...")
    tfidf = TfidfRetriever().fit(product_texts, product_ids)
    print("      Building the BM25 index...")
    bm25 = BM25Retriever(tokenizer=simple_tokenize).fit(product_texts, product_ids)

    # 5) Retrieve Top-K for every evaluation query with each method.
    print(f"[5/7] Retrieving Top-{TOP_K} per query (TF-IDF, fast)...")
    tfidf_retrieved = tfidf.retrieve_batch(eval_queries, top_k=TOP_K)
    print(f"      Retrieving Top-{TOP_K} per query (BM25, this is the slow part)...")
    bm25_retrieved = bm25.retrieve_batch(eval_queries, top_k=TOP_K)

    # 6) Score both methods.
    print("[6/7] Scoring Precision@10, Recall@10, NDCG@10...")
    tfidf_metrics, tfidf_rows = evaluate_method("TF-IDF", tfidf_retrieved, eval_queries, qrels)
    bm25_metrics, bm25_rows = evaluate_method("BM25", bm25_retrieved, eval_queries, qrels)

    print("      TF-IDF:", {k: round(v, 4) for k, v in tfidf_metrics.items()})
    print("      BM25  :", {k: round(v, 4) for k, v in bm25_metrics.items()})

    # 7) Write all output files.
    print("[7/7] Writing output files...")

    # 7a) Combined retrieval results CSV (both methods).
    results_df = pd.DataFrame(tfidf_rows + bm25_rows,
                              columns=["method", "query", "product_id",
                                       "score", "rank", "relevance_score"])
    results_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"Saved retrieval results to: {RESULTS_CSV}")

    # 7b) Metrics report + comparison table.
    write_metrics_report(tfidf_metrics, bm25_metrics,
                         n_unique_queries, len(product_ids), len(eval_queries))

    # 7c) Sample query analysis.
    write_sample_query_analysis(eval_queries, tfidf_retrieved, bm25_retrieved, qrels)

    # 7d) Visualizations.
    plot_metrics(tfidf_metrics, bm25_metrics)
    plot_query_length_distribution(df)

    print("\nWeek 1 Retrieval Baselines completed successfully.")


if __name__ == "__main__":
    main()
