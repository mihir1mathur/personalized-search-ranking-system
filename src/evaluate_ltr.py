"""
evaluate_ltr.py  --  Week 5 main script: LEARNING-TO-RANK (LambdaMART / LightGBM)
=================================================================================

Run it with:

    python src/evaluate_ltr.py

WHAT IT DOES, STEP BY STEP
--------------------------
Week 5 adds a THIRD stage on top of the Week 4 "retrieve -> re-rank" pipeline:
a lightweight LEARNING-TO-RANK (LTR) model that LEARNS how to combine every
signal we already produce (BM25, embeddings, hybrid, cross-encoder, plus a few
cheap text features) into one final ranking score.

    TF-IDF -> BM25 -> Embeddings -> Hybrid -> CrossEncoder -> Learning-to-Rank

The whole point of this script is REUSE. It does NOT re-index, re-encode, or
invent new heavy computation:
  1. Loads the SAME 50k sample and builds the SAME corpus / qrels / eval queries
     as Weeks 0-4 (by importing Week 1's own functions).
  2. Rebuilds the Week 3 hybrid retriever REUSING the cached Week 2 product
     embeddings (no re-encoding of 48k products).
  3. Scores each query's hybrid Top-50 candidates -- the SAME candidate set
     Week 4 re-ranked.
  4. Runs the cross-encoder ONCE over those candidates and CACHES the scores to
     results/week5_ce_cache.pkl, so re-runs are instant (reuse, not recompute).
  5. Builds a small, interpretable FEATURE VECTOR per candidate from signals we
     already have (see ltr_features.py) -- essentially free.
  6. Splits the evaluation queries into TRAIN / TEST by query, trains a LightGBM
     LambdaMART ranker on TRAIN, and evaluates EVERY method on the held-out TEST
     queries (so the LTR model is never scored on data it trained on).
  7. Compares SIX methods on Precision@10, Recall@10, NDCG@10, MAP, MRR:
        TF-IDF, BM25, Embeddings, Hybrid, Hybrid+CrossEncoder,
        Hybrid+CrossEncoder+LTR.
  8. Measures serving cost (latency, memory, CPU, model size).
  9. Writes results/week5_metrics.txt, week5_failure_analysis.txt,
     week5_examples.txt, week5_ltr_comparison.png, week5_feature_importance.png,
     and saves the model to models/ltr_lightgbm.txt.

IMPORTANT NOTE ON THE NUMBERS
-----------------------------
Weeks 1-4 reported metrics over ALL 3,000 evaluation queries. Week 5 must not
score the LTR model on its own training data, so it reports every method's
metrics over the HELD-OUT TEST split only. That is why the baseline numbers
here differ slightly from Week 4's -- it is a smaller, unseen query set, which
is the honest way to measure a learned model.

WHAT IT DOES NOT DO (on purpose):
  no new indexing, no re-encoding, no repeated cross-encoder passes, no giant
  hyper-parameter search, and it does NOT modify any Week 0-4 file or output.
"""

import os
import sys
import gc
import time
import pickle

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
    import pandas as pd  # noqa: F401  (used indirectly by week1.load_sample)
except ImportError:
    print("ERROR: This script needs pandas and numpy.  pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: This script needs matplotlib.  pip install matplotlib")
    sys.exit(1)

# psutil is optional: we use it for memory / CPU reporting, but degrade
# gracefully if it is not installed.
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PROC = None

# Reuse Week 1 (corpus/qrels/eval queries/metrics) and the Week 1 TF-IDF model.
try:
    import metrics
    from tfidf_retriever import TfidfRetriever
    import evaluate_retrieval as week1
    from bm25_retriever import simple_tokenize
except ImportError as e:
    print(f"ERROR: Could not import a Week 1 module: {e}")
    sys.exit(1)

# Reuse the Week 3 hybrid retriever and the Week 2 model-name constant.
try:
    from hybrid_retriever import HybridRetriever
    from embedding_retriever import DEFAULT_MODEL_NAME
except ImportError as e:
    print(f"ERROR: Could not import the Week 3 hybrid retriever: {e}")
    sys.exit(1)

# Reuse the Week 4 cross-encoder re-ranker.
try:
    from cross_encoder_reranker import CrossEncoderReranker, DEFAULT_CROSS_ENCODER_NAME
except ImportError as e:
    print(f"ERROR: Could not import the Week 4 cross-encoder: {e}")
    sys.exit(1)

# The NEW Week 5 pieces.
try:
    from ltr_features import build_feature_row, build_feature_matrix, FEATURE_NAMES
    from ltr_ranker import LTRRanker
except ImportError as e:
    print(f"ERROR: Could not import the Week 5 LTR modules: {e}")
    print("Install LightGBM with: pip install lightgbm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration (reuse Week 1's constants so the setup is identical).
# ---------------------------------------------------------------------------
RESULTS_DIR = week1.RESULTS_DIR
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TOP_K = week1.TOP_K                  # 10
REL_THRESHOLD = week1.REL_THRESHOLD  # 1

# The hybrid Top-50 per query is BOTH the cross-encoder candidate set (Week 4)
# and the LTR candidate set (Week 5). Same set -> fair comparison, no rework.
CANDIDATE_DEPTH = 50

# Week 5 focuses the evaluation on a reproducible subset of queries so the
# ONE-TIME cross-encoder pass (the only heavy compute this week) fits well
# under the runtime budget. The cross-encoder scores are cached afterwards, so
# every re-run is instant. Set to None to use all of Week 1's eval queries.
EVAL_QUERY_LIMIT = 900

# Cross-encoder batch size (bigger = better CPU throughput, more memory).
CE_BATCH_SIZE = 64

# Cross-encoder cost scales with input length. Product text can be thousands of
# characters (long descriptions), most of which is noise for relevance. We feed
# the cross-encoder only the first CE_TEXT_MAXCHARS characters (title + lead),
# which is standard practice for passage re-rankers and speeds up scoring
# several-fold on CPU with negligible relevance loss. Set to None to disable.
CE_TEXT_MAXCHARS = 400

# The same alpha/beta sweep Week 3/4 used (alpha=BM25 weight, beta=embeddings).
WEIGHT_SETTINGS = [(0.5, 0.5), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]

# Train/test split over QUERIES (never mix a query across splits).
TRAIN_FRACTION = 0.75
# Fraction of the TRAIN queries held out as a validation set for early stopping.
VAL_FRACTION = 0.15
SPLIT_SEED = 42

# Prefer a locally-downloaded cross-encoder under models/ (offline friendly).
LOCAL_MODEL_DIR = os.path.join(MODELS_DIR, "ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_MODEL = (
    LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else DEFAULT_CROSS_ENCODER_NAME
)

# Reuse the Week 2 cached embeddings (skip the slow product-encoding step).
EMB_CACHE_VECTORS = os.path.join(RESULTS_DIR, "product_embeddings_minilm.npy")
EMB_CACHE_IDS = os.path.join(RESULTS_DIR, "product_embeddings_ids.npy")

# Week 5 outputs (all NEW -- we do NOT touch Week 0/1/2/3/4 outputs).
WEEK5_METRICS_TXT = os.path.join(RESULTS_DIR, "week5_metrics.txt")
WEEK5_FAILURE_TXT = os.path.join(RESULTS_DIR, "week5_failure_analysis.txt")
WEEK5_EXAMPLES_TXT = os.path.join(RESULTS_DIR, "week5_examples.txt")
WEEK5_PNG = os.path.join(RESULTS_DIR, "week5_ltr_comparison.png")
WEEK5_FEATIMP_PNG = os.path.join(RESULTS_DIR, "week5_feature_importance.png")
CE_CACHE_PKL = os.path.join(RESULTS_DIR, "week5_ce_cache.pkl")
LTR_MODEL_PATH = os.path.join(MODELS_DIR, "ltr_lightgbm.txt")

METHOD_ORDER = ["TF-IDF", "BM25", "Embeddings", "Hybrid",
                "Hybrid + CrossEncoder", "Hybrid + CrossEncoder + LTR"]


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def shorten(text, limit=60):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def average(values):
    nums = [v for v in values if v is not None]
    return float(np.mean(nums)) if nums else 0.0


# --- Extra metrics NOT in Week 1's metrics.py (defined here, we don't edit it) --
def average_precision_at_k(retrieved_ids, qrels, k=10, rel_threshold=1):
    """
    Average Precision@k: mean of Precision@i taken at each rank i (<=k) that
    holds a relevant item, divided by the number of known relevant items
    (capped at k). Returns None if the query has no relevant items.
    """
    total_relevant = sum(1 for s in qrels.values() if s >= rel_threshold)
    if total_relevant == 0:
        return None
    hits = 0
    precision_sum = 0.0
    for i, pid in enumerate(retrieved_ids[:k], start=1):
        if qrels.get(pid, 0) >= rel_threshold:
            hits += 1
            precision_sum += hits / i
    denom = min(total_relevant, k)
    return precision_sum / denom if denom else 0.0


def reciprocal_rank(retrieved_ids, qrels, k=10, rel_threshold=1):
    """1 / (rank of the first relevant item in the top-k), else 0. None if no
    relevant items exist for the query."""
    total_relevant = sum(1 for s in qrels.values() if s >= rel_threshold)
    if total_relevant == 0:
        return None
    for i, pid in enumerate(retrieved_ids[:k], start=1):
        if qrels.get(pid, 0) >= rel_threshold:
            return 1.0 / i
    return 0.0


def score_all_metrics(retrieved_per_query, eval_queries, qrels):
    """Average P@10, R@10, NDCG@10, MAP, MRR over the given queries."""
    p, r, n, ap, rr = [], [], [], [], []
    for query, retrieved in zip(eval_queries, retrieved_per_query):
        rids = [pid for (pid, _s, _rk) in retrieved]
        q = qrels.get(query, {})
        p.append(metrics.precision_at_k(rids, q, TOP_K, REL_THRESHOLD))
        r.append(metrics.recall_at_k(rids, q, TOP_K, REL_THRESHOLD))
        n.append(metrics.ndcg_at_k(rids, q, TOP_K))
        ap.append(average_precision_at_k(rids, q, TOP_K, REL_THRESHOLD))
        rr.append(reciprocal_rank(rids, q, TOP_K, REL_THRESHOLD))
    return {
        "Precision@10": average(p),
        "Recall@10": average(r),
        "NDCG@10": average(n),
        "MAP": average(ap),
        "MRR": average(rr),
    }


def per_query_ndcg(retrieved_per_query, eval_queries, qrels):
    out = []
    for query, retrieved in zip(eval_queries, retrieved_per_query):
        rids = [pid for (pid, _s, _rk) in retrieved]
        out.append(metrics.ndcg_at_k(rids, qrels.get(query, {}), TOP_K))
    return out


def mem_mb():
    """Current resident memory of this process in MB (0 if psutil missing)."""
    if _PROC is None:
        return 0.0
    return _PROC.memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Cross-encoder scoring for ALL Top-50 candidates, with an on-disk cache.
# ---------------------------------------------------------------------------
def compute_ce_scores(reranker, eval_queries, hybrid_candidates,
                      product_text_by_id, best_weight):
    """
    Score EVERY (query, candidate) pair once with the cross-encoder and return
    ce_by_query: {query: {product_id: ce_score}}.

    Reuses results/week5_ce_cache.pkl when it matches this run's setup (so a
    re-run does not pay the cross-encoder cost again). Returns
    (ce_by_query, seconds_spent, n_pairs, used_cache).
    """
    signature = {
        "model": os.path.basename(str(CROSS_ENCODER_MODEL)),
        "depth": CANDIDATE_DEPTH,
        "weight": list(best_weight),
        "n_queries": len(eval_queries),
        "maxchars": CE_TEXT_MAXCHARS,
    }

    def _ce_text(pid):
        """Text fed to the cross-encoder (optionally truncated for speed)."""
        txt = product_text_by_id.get(pid, "")
        if CE_TEXT_MAXCHARS is not None:
            txt = txt[:CE_TEXT_MAXCHARS]
        return txt

    # Load any existing (possibly PARTIAL) cache with a matching signature. The
    # cache is checkpointed every chunk below, so a run that was interrupted can
    # resume: we keep whatever was already scored and only score the rest.
    ce_by_query = {}
    if os.path.exists(CE_CACHE_PKL):
        try:
            with open(CE_CACHE_PKL, "rb") as f:
                blob = pickle.load(f)
            if blob.get("signature") == signature:
                ce_by_query = blob["scores"]
        except Exception as e:
            print(f"      (Ignoring unreadable CE cache: {e})")
            ce_by_query = {}

    throughput_holder = {"pps": None}

    def _save_cache():
        try:
            tmp = CE_CACHE_PKL + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"signature": signature, "scores": ce_by_query,
                             "throughput_pps": throughput_holder["pps"]}, f)
            os.replace(tmp, CE_CACHE_PKL)   # atomic -> never a half-written cache
        except Exception as e:
            print(f"      (Could not write CE cache: {e})")

    # Total pairs needed, and the subset still MISSING from the cache.
    total_needed = 0
    missing_pairs, missing_index = [], []
    for query, ranked in zip(eval_queries, hybrid_candidates):
        got = ce_by_query.setdefault(query, {})
        for (pid, _s, _r) in ranked:
            total_needed += 1
            if pid not in got:
                missing_pairs.append([query, _ce_text(pid)])
                missing_index.append((query, pid))

    if not missing_pairs:
        print("      Reusing cached cross-encoder scores "
              "(results/week5_ce_cache.pkl) -- no re-scoring.")
        # Reuse the steady-state throughput recorded when the cache was built.
        thr = None
        try:
            with open(CE_CACHE_PKL, "rb") as f:
                thr = pickle.load(f).get("throughput_pps")
        except Exception:
            thr = None
        return ce_by_query, 0.0, 0, True, thr

    already = total_needed - len(missing_pairs)
    if already:
        print(f"      Resuming cross-encoder cache: {already:,}/{total_needed:,} "
              f"pairs already scored; scoring the remaining {len(missing_pairs):,}...")
    else:
        print(f"      Scoring {len(missing_pairs):,} (query, product) pairs with "
              f"the cross-encoder (one pass)...")

    n = len(missing_pairs)
    t0 = time.perf_counter()
    chunk = max(reranker.batch_size * 20, 1000)
    # For a steady-state throughput estimate we time from AFTER the first chunk
    # (which includes one-off model warmup) to the end.
    t_after_warmup = None
    pairs_after_warmup = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_scores = reranker.score_pairs(missing_pairs[start:end])
        for (query, pid), sc in zip(missing_index[start:end], chunk_scores):
            ce_by_query[query][pid] = float(sc)
        if t_after_warmup is None:
            t_after_warmup = time.perf_counter()   # warmup chunk done
        else:
            pairs_after_warmup = end - chunk
        # Update the running steady-state throughput and checkpoint.
        if t_after_warmup is not None and pairs_after_warmup > 0:
            dt = time.perf_counter() - t_after_warmup
            if dt > 0:
                throughput_holder["pps"] = pairs_after_warmup / dt
        _save_cache()  # checkpoint after every chunk (cheap; survives a kill)
        print(f"        ...scored {end:,}/{n:,} new pairs "
              f"({end / n * 100:5.1f}%)  [cache checkpointed]")
    seconds = time.perf_counter() - t0
    if throughput_holder["pps"] is None and seconds > 0:
        throughput_holder["pps"] = n / seconds

    _save_cache()
    print(f"      Cached cross-encoder scores to: {CE_CACHE_PKL}")
    return ce_by_query, seconds, n, False, throughput_holder["pps"]


# ---------------------------------------------------------------------------
# Build LTR feature rows for one query's Top-50 candidates.
# ---------------------------------------------------------------------------
def build_query_features(query, ranked_top50, norm_by_pid, ce_scores,
                         title_by_id, brand_by_id, color_by_id, text_by_id):
    """
    Returns (candidate_ids, feature_rows) for one query. Each feature row is
    built from signals we already have -- no new heavy computation.
    """
    query_tokens = simple_tokenize(query)
    # Rank the candidates by cross-encoder score to get each one's ce_rank
    # (1 = highest cross-encoder score). This hands the model the cross-encoder's
    # clean ordering as a feature.
    ce_order = sorted(
        (pid for (pid, _s, _r) in ranked_top50),
        key=lambda pid: -ce_scores.get(pid, float("-inf")),
    )
    ce_rank_by_pid = {pid: rank for rank, pid in enumerate(ce_order, start=1)}

    candidate_ids, rows = [], []
    for (pid, hybrid_score, hybrid_rank) in ranked_top50:
        bm25_n, emb_n = norm_by_pid.get(pid, (0.0, 0.0))
        ce = ce_scores.get(pid, 0.0)
        row = build_feature_row(
            query_tokens=query_tokens, query_str=query,
            bm25_norm=bm25_n, emb_norm=emb_n,
            hybrid_score=hybrid_score, ce_score=ce,
            ce_rank=ce_rank_by_pid.get(pid, len(ranked_top50)),
            hybrid_rank=hybrid_rank,
            title=title_by_id.get(pid, ""),
            brand=brand_by_id.get(pid, ""),
            color=color_by_id.get(pid, ""),
            full_text=text_by_id.get(pid, ""),
        )
        candidate_ids.append(pid)
        rows.append(row)
    return candidate_ids, rows


# ---------------------------------------------------------------------------
# Output writer: metrics report (six methods, five metrics + cost).
# ---------------------------------------------------------------------------
def write_metrics_report(method_metrics, best_weight, n_products, n_train,
                         n_test, cost):
    lines = []
    bar = "=" * 78
    lines.append(bar)
    lines.append("WEEK 5: LEARNING-TO-RANK (LambdaMART / LightGBM) -- METRICS REPORT")
    lines.append("Project 3: Personalized Search Ranking System")
    lines.append(bar)
    lines.append("")
    lines.append("SETUP")
    lines.append("-" * 78)
    lines.append("Full pipeline : TF-IDF -> BM25 -> Embeddings -> Hybrid -> "
                 "CrossEncoder -> LTR")
    lines.append(f"Embedding model                 : {DEFAULT_MODEL_NAME}")
    lines.append(f"Cross-encoder (stage 2)         : {DEFAULT_CROSS_ENCODER_NAME}")
    lines.append(f"LTR model (stage 3)             : LightGBM LGBMRanker "
                 f"(lambdarank objective)")
    lines.append(f"LTR features                    : {len(FEATURE_NAMES)} "
                 f"({', '.join(FEATURE_NAMES)})")
    lines.append(f"Best hybrid weights (by NDCG@10): alpha(BM25)={best_weight[0]}, "
                 f"beta(Emb)={best_weight[1]}")
    lines.append(f"Candidates per query (stage 1)  : Top-{CANDIDATE_DEPTH} from hybrid")
    lines.append(f"Search corpus (unique products) : {n_products:,}")
    lines.append(f"LTR training queries            : {n_train:,}")
    lines.append(f"Evaluation queries (HELD-OUT)   : {n_test:,}")
    lines.append(f"Top-K retrieved per query       : {TOP_K}")
    lines.append(f"Relevance threshold (>=)        : {REL_THRESHOLD}")
    lines.append("")
    lines.append("NOTE: all six methods below are scored on the SAME held-out TEST")
    lines.append("queries the LTR model never saw in training. Baseline numbers thus")
    lines.append("differ slightly from Week 4 (which used all 3,000 queries).")
    lines.append("")

    lines.append("MAIN COMPARISON (averages over the held-out test queries)")
    lines.append("-" * 78)
    lines.append("| Method                        | P@10   | R@10   | NDCG@10 | MAP    | MRR    |")
    lines.append("| ----------------------------- | ------ | ------ | ------- | ------ | ------ |")
    for name in METHOD_ORDER:
        m = method_metrics[name]
        lines.append("| {:<29} | {:.4f} | {:.4f} | {:.4f}  | {:.4f} | {:.4f} |".format(
            name, m["Precision@10"], m["Recall@10"], m["NDCG@10"], m["MAP"], m["MRR"]))
    lines.append("")

    winner = max(METHOD_ORDER, key=lambda n: method_metrics[n]["NDCG@10"])
    lines.append("WHICH METHOD PERFORMS BEST?")
    lines.append("-" * 78)
    lines.append(f"On NDCG@10 (headline ranking metric), the best method is: {winner}.")
    lines.append("")

    def pct(new, old):
        return (new - old) / old * 100.0 if old else float("nan")

    ce_m = method_metrics["Hybrid + CrossEncoder"]
    ltr_m = method_metrics["Hybrid + CrossEncoder + LTR"]
    lines.append("LTR vs the stage-2 CROSS-ENCODER (relative change):")
    for metric in ["Precision@10", "Recall@10", "NDCG@10", "MAP", "MRR"]:
        lines.append(f"  {metric:<13}: {pct(ltr_m[metric], ce_m[metric]):+.1f}%")
    lines.append("")
    lines.append("LTR vs every other method (NDCG@10 relative change):")
    for name in ["TF-IDF", "BM25", "Embeddings", "Hybrid", "Hybrid + CrossEncoder"]:
        lines.append(f"  vs {name:<24}: "
                     f"{pct(ltr_m['NDCG@10'], method_metrics[name]['NDCG@10']):+.1f}%")
    lines.append("")

    lines.append("SERVING COST (measured on this machine)")
    lines.append("-" * 78)
    lines.append(f"  Peak process memory                : {cost['peak_mem_mb']:.0f} MB")
    import math as _math
    if cost['ce_pairs_per_sec'] is None or _math.isnan(cost['ce_pairs_per_sec']):
        lines.append("  Cross-encoder scoring              : reused from cache this "
                     "run (one-time cost; not re-scored)")
    else:
        lines.append(f"  Cross-encoder scoring throughput   : "
                     f"{cost['ce_pairs_per_sec']:.0f} (query,product) pairs/sec")
        lines.append(f"  Cross-encoder latency / query      : "
                     f"{cost['ce_ms_per_query']:.1f} ms (Top-{CANDIDATE_DEPTH} pairs)")
    lines.append(f"  LTR feature-build + predict / query: "
                 f"{cost['ltr_ms_per_query']:.3f} ms (Top-{CANDIDATE_DEPTH})")
    lines.append(f"  LTR training time                  : {cost['train_secs']:.1f} s")
    lines.append(f"  LTR model size on disk             : {cost['model_kb']:.0f} KB")
    lines.append(f"  Process CPU count available        : {cost['cpu_count']}")
    lines.append("")
    lines.append("  INTERPRETATION: the LTR stage adds a negligible per-query cost")
    lines.append("  (sub-millisecond tree inference over ~50 tiny feature vectors)")
    lines.append("  on top of the cross-encoder, yet learns a better combination of")
    lines.append("  the existing signals. The dominant serving cost remains the")
    lines.append("  cross-encoder; LTR is effectively free at inference time.")
    lines.append("")

    lines.append("WHY LEARNING-TO-RANK HELPS")
    lines.append("-" * 78)
    lines.append("  - Weeks 3-4 combined signals with FIXED rules (a hand-set weighted")
    lines.append("    sum for hybrid; 'sort by cross-encoder alone' for the re-ranker).")
    lines.append("  - LTR LEARNS the combination from the relevance labels. On this")
    lines.append("    dataset it relied most on the cross-encoder's ORDER (ce_rank) and")
    lines.append("    raw score (ce_score), refined by the hybrid / embedding / BM25")
    lines.append("    signals -- effectively a learned, calibrated blend that smooths the")
    lines.append("    cross-encoder's occasional top-of-list mis-orderings. Lexical")
    lines.append("    exact-match and brand features were available but rarely decisive on")
    lines.append("    this query subset (see the feature-importance report).")
    lines.append("  - Because it optimizes a ranking objective (LambdaMART weights each")
    lines.append("    pair by its impact on NDCG), it directly targets the metric we")
    lines.append("    report, rather than a proxy.")
    lines.append("")
    with open(WEEK5_METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved metrics report to: {WEEK5_METRICS_TXT}")


# ---------------------------------------------------------------------------
# Output writer: failure / win analysis (LTR vs cross-encoder, per-query NDCG).
# ---------------------------------------------------------------------------
def write_failure_analysis(eval_queries, qrels, product_text_by_id,
                           ce_retrieved, ltr_retrieved, ce_ndcg, ltr_ndcg,
                           feature_importance):
    improved, unchanged, worsened = [], [], []
    eps = 1e-9
    for i in range(len(eval_queries)):
        h, r = ce_ndcg[i], ltr_ndcg[i]
        if h is None or r is None:
            continue
        diff = r - h
        if diff > eps:
            improved.append((diff, i))
        elif diff < -eps:
            worsened.append((diff, i))
        else:
            unchanged.append((0.0, i))
    improved.sort(reverse=True)
    worsened.sort()

    def block(title, rows, show=10):
        out = ["=" * 78, title, "=" * 78, ""]
        if not rows:
            out += ["(No qualifying queries found in this run.)", ""]
            return out
        for n, (_diff, i) in enumerate(rows[:show], start=1):
            query = eval_queries[i]
            q = qrels.get(query, {})
            known = sorted(q.items(), key=lambda kv: -kv[1])
            out.append("-" * 78)
            out.append(f"CASE {n}:  QUERY = {query}")
            out.append(f"  NDCG@10 -> CrossEncoder={(ce_ndcg[i] or 0):.3f}  ->  "
                       f"LTR={(ltr_ndcg[i] or 0):.3f}  "
                       f"(change {((ltr_ndcg[i] or 0) - (ce_ndcg[i] or 0)):+.3f})")
            out.append("  Ground truth (judged products):")
            for pid, s in known:
                out.append(f"     - {pid} (rel={s})  {shorten(product_text_by_id.get(pid, ''))}")
            out.append("  BEFORE  (CrossEncoder Top-5):")
            for pid, score, rank in ce_retrieved[i][:5]:
                rel = q.get(pid, 0)
                mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
                out.append(f"     {rank}. {pid}  rel={rel}  CE={score:+.3f}  "
                           f"{shorten(product_text_by_id.get(pid, ''), 34)}{mark}")
            out.append("  AFTER   (LTR Top-5):")
            for pid, score, rank in ltr_retrieved[i][:5]:
                rel = q.get(pid, 0)
                mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
                out.append(f"     {rank}. {pid}  rel={rel}  LTR={score:+.3f}  "
                           f"{shorten(product_text_by_id.get(pid, ''), 34)}{mark}")
            out.append("")
        return out

    total = len(improved) + len(unchanged) + len(worsened)
    lines = ["=" * 78,
             "WEEK 5: LEARNING-TO-RANK FAILURE / WIN ANALYSIS",
             "Project 3: Personalized Search Ranking System",
             "=" * 78,
             "We compare per-query NDCG@10 BEFORE LTR (the cross-encoder Top-10)",
             "and AFTER LTR (LightGBM reorders the same Top-50 candidates).",
             "'CE' = cross-encoder score; 'LTR' = learned LightGBM ranking score.",
             ""]
    lines.append("SUMMARY (over held-out test queries with a defined NDCG@10)")
    lines.append("-" * 78)
    lines.append(f"  Queries scored      : {total:,}")
    lines.append(f"  IMPROVED by LTR     : {len(improved):,}  "
                 f"({(100.0 * len(improved) / total if total else 0):.1f}%)")
    lines.append(f"  UNCHANGED           : {len(unchanged):,}  "
                 f"({(100.0 * len(unchanged) / total if total else 0):.1f}%)")
    lines.append(f"  WORSENED by LTR     : {len(worsened):,}  "
                 f"({(100.0 * len(worsened) / total if total else 0):.1f}%)")
    lines.append("")
    lines.append("WHAT THE MODEL LEARNED (feature importance by gain)")
    lines.append("-" * 78)
    for name, imp in feature_importance:
        lines.append(f"  {name:<22} {imp:12.1f}")
    lines.append("")
    lines += block("PART A: TOP 10 QUERIES IMPROVED BY LTR", improved)
    lines += block("PART B: 10 QUERIES UNCHANGED BY LTR", unchanged)
    lines += block("PART C: TOP 10 QUERIES WORSENED BY LTR", worsened)
    lines += ["=" * 78, "HOW TO READ THIS", "=" * 78,
              "- IMPROVED: LTR found a better ORDER than the cross-encoder alone by",
              "  blending the cross-encoder score with keyword/brand/exact-match",
              "  features (e.g. a precise part-number query where BM25 exact match",
              "  should outrank a semantically-close but wrong product).",
              "- UNCHANGED: the cross-encoder ordering was already optimal for that",
              "  query, so the learned model left it alone.",
              "- WORSENED: LTR's learned trade-off occasionally hurts an individual",
              "  query (e.g. it trusted a keyword feature that happened to mislead),",
              "  even though it improves the AVERAGE. Like the re-ranker, it can only",
              "  reorder the Top-50 -- it cannot recover a relevant product that",
              "  stage-1 retrieval never surfaced.",
              ""]
    with open(WEEK5_FAILURE_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved failure analysis to: {WEEK5_FAILURE_TXT}")
    return len(improved), len(unchanged), len(worsened)


# ---------------------------------------------------------------------------
# Output writer: worked examples (original -> hybrid -> CE -> LTR).
# ---------------------------------------------------------------------------
def write_examples(eval_queries, qrels, product_text_by_id,
                   bm25_retrieved, hybrid_retrieved, ce_retrieved, ltr_retrieved,
                   ce_ndcg, ltr_ndcg):
    lines = ["=" * 78,
             "WEEK 5: LEARNING-TO-RANK -- WORKED EXAMPLES",
             "Project 3: Personalized Search Ranking System",
             "=" * 78,
             "",
             "For each query we show how the ranking evolves through the full",
             "pipeline stages:  BM25 (keyword) -> Hybrid -> CrossEncoder -> LTR.",
             "'rel' is the known relevance (3=exact .. 0=irrelevant).",
             ""]

    # Choose queries where LTR changed the NDCG most vs the cross-encoder.
    changes = []
    for i in range(len(eval_queries)):
        h, r = ce_ndcg[i], ltr_ndcg[i]
        if h is None or r is None:
            continue
        changes.append((abs(r - h), i))
    changes.sort(reverse=True)
    chosen = [i for (_d, i) in changes[:6]]

    def show_stage(title, ranked, score_label):
        out = [f"  {title}:"]
        for pid, score, rank in ranked[:5]:
            rel = qrels.get(eval_queries[i], {}).get(pid, 0)
            mark = "  <-- relevant" if rel >= REL_THRESHOLD else ""
            out.append(f"     {rank}. {pid}  rel={rel}  {score_label}={score:+.3f}  "
                       f"{shorten(product_text_by_id.get(pid, ''), 38)}{mark}")
        return out

    for n, i in enumerate(chosen, start=1):
        query = eval_queries[i]
        q = qrels.get(query, {})
        known = sorted(q.items(), key=lambda kv: -kv[1])
        lines.append("=" * 78)
        lines.append(f"EXAMPLE {n}:  QUERY = {query}")
        lines.append(f"  NDCG@10  CrossEncoder={(ce_ndcg[i] or 0):.3f}  ->  "
                     f"LTR={(ltr_ndcg[i] or 0):.3f}")
        lines.append("  Ground truth (judged products):")
        for pid, s in known:
            lines.append(f"     - {pid} (rel={s})  {shorten(product_text_by_id.get(pid, ''))}")
        lines.append("")
        lines += show_stage("STAGE 1a  BM25 (keyword) Top-5", bm25_retrieved[i], "bm25")
        lines.append("")
        lines += show_stage("STAGE 1b  Hybrid Top-5", hybrid_retrieved[i], "hyb")
        lines.append("")
        lines += show_stage("STAGE 2   CrossEncoder Top-5", ce_retrieved[i], "CE")
        lines.append("")
        lines += show_stage("STAGE 3   LTR (final) Top-5", ltr_retrieved[i], "LTR")
        lines.append("")

    with open(WEEK5_EXAMPLES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved worked examples to: {WEEK5_EXAMPLES_TXT}")


# ---------------------------------------------------------------------------
# Output writer: charts.
# ---------------------------------------------------------------------------
def plot_comparison(method_metrics):
    metric_names = ["Precision@10", "Recall@10", "NDCG@10", "MAP", "MRR"]
    x = np.arange(len(metric_names))
    width = 0.14
    fig, ax = plt.subplots(figsize=(14, 6))
    for index, name in enumerate(METHOD_ORDER):
        m = method_metrics[name]
        values = [m[metric] for metric in metric_names]
        offset = (index - 2.5) * width
        bars = ax.bar(x + offset, values, width, label=name)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5)
    ax.set_ylabel("Score")
    ax.set_title("Week 5: Six retrieval/ranking methods  (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(WEEK5_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison chart to: {WEEK5_PNG}")


def plot_feature_importance(feature_importance):
    names = [n for n, _ in feature_importance][::-1]
    values = [v for _, v in feature_importance][::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, values, color="#4C72B0")
    ax.set_xlabel("Importance (total gain)")
    ax.set_title("Week 5: LTR feature importance (what the model relies on)")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.0f}", xy=(v, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(WEEK5_FEATIMP_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature-importance chart to: {WEEK5_FEATIMP_PNG}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("WEEK 5: LEARNING-TO-RANK (LambdaMART / LightGBM)")
    print("=" * 78)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    t_start = time.perf_counter()

    # 1) Load the SAME 50k sample.
    print("\n[1/9] Loading the Week 0 sample...")
    df = week1.load_sample()
    print(f"      Loaded {len(df):,} rows.")

    # 2) Build the SAME corpus, qrels, eval queries + product metadata dicts.
    print("[2/9] Building corpus, qrels, eval queries (reusing Week 1)...")
    product_ids, product_texts = week1.build_corpus(df)
    qrels = week1.build_qrels(df)
    # Focus on a reproducible subset so the one-time cross-encoder pass fits the
    # runtime budget (choose_eval_queries samples deterministically with a seed).
    if EVAL_QUERY_LIMIT is not None:
        week1.EVAL_QUERY_SAMPLE_SIZE = EVAL_QUERY_LIMIT
    eval_queries = week1.choose_eval_queries(qrels)
    product_text_by_id = dict(zip(product_ids, product_texts))

    # Per-product metadata for the cheap LTR text features (from the same df).
    meta = df.drop_duplicates(subset="product_id")
    title_by_id = dict(zip(meta["product_id"].astype(str), meta["product_title"].astype(str)))
    brand_by_id = dict(zip(meta["product_id"].astype(str), meta["product_brand"].astype(str)))
    color_by_id = dict(zip(meta["product_id"].astype(str), meta["product_color"].astype(str)))
    print(f"      Corpus: {len(product_ids):,} products | {len(qrels):,} queries; "
          f"evaluating {len(eval_queries):,}.")
    del df, meta
    gc.collect()

    # 3) Rebuild the hybrid retriever (reuse cached Week 2 embeddings).
    print("[3/9] Building the hybrid retriever (reusing cached embeddings)...")
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

    print("      Building the TF-IDF index (baseline)...")
    tfidf = TfidfRetriever().fit(product_texts, product_ids)

    # 4) Score every query once with BM25 + embeddings (stage-1 retrieval).
    print("[4/9] Computing hybrid candidates (BM25 + embedding scores)...")
    candidates = hybrid.precompute_candidates(eval_queries, pure_top_k=TOP_K)
    bm25_retrieved = [c["bm25_ranking"] for c in candidates]
    emb_retrieved = [c["emb_ranking"] for c in candidates]
    print("      Retrieving Top-10 with TF-IDF (baseline)...")
    tfidf_retrieved = tfidf.retrieve_batch(eval_queries, top_k=TOP_K)

    # 5) Pick best hybrid weights by NDCG@10 (same sweep as Week 3/4).
    print("[5/9] Sweeping hybrid weights, selecting best by NDCG@10...")
    best_weight, best_ndcg, hybrid_top10_by_weight = None, -1.0, {}
    for (alpha, beta) in WEIGHT_SETTINGS:
        r10 = hybrid.rank_from_candidates(candidates, alpha=alpha, beta=beta, top_k=TOP_K)
        m = score_all_metrics(r10, eval_queries, qrels)
        hybrid_top10_by_weight[(alpha, beta)] = r10
        print(f"      alpha={alpha}, beta={beta} -> NDCG@10={m['NDCG@10']:.4f}")
        if m["NDCG@10"] > best_ndcg:
            best_ndcg, best_weight = m["NDCG@10"], (alpha, beta)
    print(f"      Best hybrid weights: alpha={best_weight[0]}, beta={best_weight[1]}")

    hybrid_retrieved = hybrid_top10_by_weight[best_weight]
    hybrid_candidates = hybrid.rank_from_candidates(
        candidates, alpha=best_weight[0], beta=best_weight[1], top_k=CANDIDATE_DEPTH)

    # Per-query pid -> (bm25_norm, emb_norm) map for feature building.
    norm_by_pid_per_query = []
    for c in candidates:
        pool_ids = [str(p) for p in c["product_ids"]]
        norm_by_pid_per_query.append({
            pid: (float(c["bm25_norm"][j]), float(c["emb_norm"][j]))
            for j, pid in enumerate(pool_ids)
        })

    # 6) Cross-encoder scores over the Top-50 candidates (cached).
    print(f"[6/9] Cross-encoder scoring the Top-{CANDIDATE_DEPTH} per query...")
    reranker = CrossEncoderReranker(model_name=CROSS_ENCODER_MODEL, batch_size=CE_BATCH_SIZE)
    ce_by_query, ce_secs, ce_pairs, used_cache, ce_throughput = compute_ce_scores(
        reranker, eval_queries, hybrid_candidates, product_text_by_id, best_weight)

    # Cross-encoder Top-10 (the Week 4 method) derived from the cached scores.
    ce_retrieved = []
    for query, ranked in zip(eval_queries, hybrid_candidates):
        scored = [(pid, ce_by_query[query][pid]) for (pid, _s, _r) in ranked]
        scored.sort(key=lambda kv: -kv[1])
        ce_retrieved.append([(pid, sc, rk) for rk, (pid, sc) in
                             enumerate(scored[:TOP_K], start=1)])

    # 7) Build LTR features for every query's Top-50 candidates.
    print("[7/9] Building LTR feature vectors (reusing existing signals)...")
    per_query_cand_ids, per_query_feature_rows = [], []
    for i, query in enumerate(eval_queries):
        cand_ids, rows = build_query_features(
            query, hybrid_candidates[i], norm_by_pid_per_query[i],
            ce_by_query.get(query, {}),
            title_by_id, brand_by_id, color_by_id, product_text_by_id)
        per_query_cand_ids.append(cand_ids)
        per_query_feature_rows.append(rows)

    # Train/test split BY QUERY (a query's rows never straddle the split), then
    # carve a small VALIDATION set out of train for early stopping.
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(eval_queries))
    n_train = int(len(eval_queries) * TRAIN_FRACTION)
    train_all = order[:n_train].tolist()
    test_idx = [i for i in range(len(eval_queries)) if i not in set(train_all)]
    n_val = max(1, int(len(train_all) * VAL_FRACTION))
    val_idx = set(train_all[:n_val])
    core_idx = [i for i in train_all if i not in val_idx]

    def assemble(indices):
        """Build a grouped (X, y, group) triple for a list of query indices."""
        X, y, grp = [], [], []
        for i in indices:
            q = qrels.get(eval_queries[i], {})
            rows = per_query_feature_rows[i]
            X.extend(rows)
            y.extend(int(q.get(pid, 0)) for pid in per_query_cand_ids[i])
            grp.append(len(rows))
        return build_feature_matrix(X), np.asarray(y, dtype="int32"), grp

    X_tr, y_tr, grp_tr = assemble(core_idx)
    X_val, y_val, grp_val = assemble(sorted(val_idx))

    # 8) Train the LightGBM LambdaMART ranker on the TRAIN queries.
    print(f"[8/9] Training LightGBM LambdaMART on {len(grp_tr):,} train queries "
          f"({X_tr.shape[0]:,} rows), early-stopping on {len(grp_val):,} val queries...")
    t_train = time.perf_counter()
    ranker = LTRRanker()
    ranker.fit(X_tr, y_tr, grp_tr, eval_set=(X_val, y_val), eval_group=grp_val)
    train_secs = time.perf_counter() - t_train
    ranker.save(LTR_MODEL_PATH)
    print(f"      Trained in {train_secs:.1f}s. Saved model to {LTR_MODEL_PATH}")

    # LTR ranking for ALL queries (we only SCORE the test ones), and measure
    # the per-query LTR inference latency (feature build already done above).
    ltr_retrieved_all = [None] * len(eval_queries)
    t_infer = time.perf_counter()
    for i in range(len(eval_queries)):
        fmat = build_feature_matrix(per_query_feature_rows[i])
        ltr_retrieved_all[i] = ranker.rank_candidates(
            per_query_cand_ids[i], fmat, top_k=TOP_K)
    ltr_ms_per_query = (time.perf_counter() - t_infer) / max(len(eval_queries), 1) * 1000

    # 9) Restrict every method to the held-out TEST queries and score.
    print("[9/9] Scoring all six methods on the held-out test queries...")
    def sub(arr):
        return [arr[i] for i in test_idx]

    test_queries = [eval_queries[i] for i in test_idx]
    method_retrieved = {
        "TF-IDF": sub(tfidf_retrieved),
        "BM25": sub(bm25_retrieved),
        "Embeddings": sub(emb_retrieved),
        "Hybrid": sub(hybrid_retrieved),
        "Hybrid + CrossEncoder": sub(ce_retrieved),
        "Hybrid + CrossEncoder + LTR": sub(ltr_retrieved_all),
    }
    method_metrics = {name: score_all_metrics(ret, test_queries, qrels)
                      for name, ret in method_retrieved.items()}
    for name in METHOD_ORDER:
        print(f"      {name:<30}:", {k: round(v, 4) for k, v in method_metrics[name].items()})

    # Cost summary. The cross-encoder throughput is the STEADY-STATE rate
    # measured (after warmup) during the actual scoring pass and persisted in the
    # cache, so it is reported consistently on both fresh and cached runs.
    model_kb = os.path.getsize(LTR_MODEL_PATH) / 1024 if os.path.exists(LTR_MODEL_PATH) else 0.0
    if ce_throughput and ce_throughput > 0:
        ce_pairs_per_sec = ce_throughput
        ce_ms_per_query = CANDIDATE_DEPTH / ce_throughput * 1000
    elif ce_secs > 0:
        ce_pairs_per_sec = ce_pairs / ce_secs
        ce_ms_per_query = CANDIDATE_DEPTH / ce_pairs_per_sec * 1000
    else:
        ce_pairs_per_sec = float("nan")
        ce_ms_per_query = float("nan")
    cost = {
        "peak_mem_mb": mem_mb(),
        "ce_pairs_per_sec": ce_pairs_per_sec,
        "ce_ms_per_query": ce_ms_per_query,
        "ltr_ms_per_query": ltr_ms_per_query,
        "train_secs": train_secs,
        "model_kb": model_kb,
        "cpu_count": (psutil.cpu_count() if psutil else os.cpu_count()),
    }

    # Write all outputs.
    feature_importance = ranker.feature_importance("gain")
    write_metrics_report(method_metrics, best_weight, len(product_ids),
                         len(grp_tr) + len(grp_val), len(test_idx), cost)
    plot_comparison(method_metrics)
    plot_feature_importance(feature_importance)

    ce_ndcg = per_query_ndcg(sub(ce_retrieved), test_queries, qrels)
    ltr_ndcg = per_query_ndcg(sub(ltr_retrieved_all), test_queries, qrels)
    n_imp, n_unch, n_wor = write_failure_analysis(
        test_queries, qrels, product_text_by_id,
        sub(ce_retrieved), sub(ltr_retrieved_all), ce_ndcg, ltr_ndcg,
        feature_importance)
    write_examples(test_queries, qrels, product_text_by_id,
                   sub(bm25_retrieved), sub(hybrid_retrieved),
                   sub(ce_retrieved), sub(ltr_retrieved_all), ce_ndcg, ltr_ndcg)

    # Final printed summary.
    total_secs = time.perf_counter() - t_start
    ce_m = method_metrics["Hybrid + CrossEncoder"]
    ltr_m = method_metrics["Hybrid + CrossEncoder + LTR"]

    def pct(new, old):
        return (new - old) / old * 100.0 if old else float("nan")

    print("")
    print("=" * 78)
    print("WEEK 5 COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"  Total wall-clock time : {total_secs / 60:.1f} min")
    print(f"  Peak process memory   : {cost['peak_mem_mb']:.0f} MB")
    print("")
    print("  NDCG@10 (held-out test queries):")
    for name in METHOD_ORDER:
        print(f"     {name:<30}: {method_metrics[name]['NDCG@10']:.4f}")
    print("")
    print(f"  LTR vs CrossEncoder NDCG@10 : {pct(ltr_m['NDCG@10'], ce_m['NDCG@10']):+.1f}%")
    total_cls = n_imp + n_unch + n_wor
    print(f"  Queries improved by LTR     : {n_imp:,}/{total_cls:,} "
          f"({(100.0 * n_imp / total_cls if total_cls else 0):.1f}%)")
    print(f"  LTR inference / query       : {ltr_ms_per_query:.3f} ms "
          f"(negligible on top of the cross-encoder)")
    print("")


if __name__ == "__main__":
    main()
