"""
search_service.py  --  the production search service (reuses Week 0-5).
=======================================================================

WHAT THIS IS
------------
``SearchService`` is the single seam between the HTTP layer and the existing
retrieval/ranking pipeline. It loads every Week 0-5 artifact ONCE at startup
and then answers queries. It does NOT reimplement any algorithm and does NOT
retrain or regenerate anything -- it imports and calls the exact same classes
the Week 5 evaluation used:

    TfidfRetriever          (src/tfidf_retriever.py)      -- Week 1
    BM25Retriever           (src/bm25_retriever.py)       -- Week 1
    EmbeddingRetriever      (src/embedding_retriever.py)  -- Week 2 (+ FAISS)
    HybridRetriever         (src/hybrid_retriever.py)     -- Week 3
    CrossEncoderReranker    (src/cross_encoder_reranker)  -- Week 4
    LTRRanker + features    (src/ltr_ranker.py, ltr_features.py) -- Week 5

REUSE, NOT RECOMPUTE
--------------------
* The product embeddings are loaded from the cached Week 2 ``.npy`` files, so
  the 48k-product catalog is NEVER re-encoded at startup.
* The trained LambdaMART model is loaded from ``models/ltr_lightgbm.txt``.
* The cross-encoder is loaded from the local ``models/`` dir (offline).
* The Week 5 cross-encoder score cache is loaded READ-ONLY as a warm lookup;
  the service never writes to that file.

The heavy imports happen inside :meth:`load` (lazy), so importing this module
in a unit test is cheap and does not require the ML stack.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api.config.settings import Settings
from api.services.cache import QueryCache
from api.services.exceptions import (
    EmptyQueryError,
    InvalidParameterError,
    ModelNotReadyError,
    QueryTooLongError,
    UnsupportedMethodError,
)
from api.utils.logging_config import get_logger
from api.utils.timing import Timer

logger = get_logger("service")

# Single-stage retrievers vs multi-stage pipelines (used by dispatch/validation).
_SINGLE_STAGE = {"tfidf", "bm25", "embedding"}
_PIPELINES = {"hybrid", "rerank", "ltr"}


class SearchService:
    """Loads the Week 0-5 pipeline once and serves ranked search over it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = QueryCache(
            max_size=settings.cache_max_size,
            ttl_seconds=settings.cache_ttl_seconds,
            enabled=settings.cache_enabled,
        )

        # Populated by load().
        self._ready = False
        self._load_error: Optional[str] = None
        self._started_at = time.time()
        self._component_status: Dict[str, Tuple[bool, str]] = {}

        # Pipeline components (built in load()).
        self.tfidf = None
        self.hybrid = None
        self.reranker = None
        self.ltr = None

        # Corpus + metadata (product_id -> text/title/brand/color).
        self.product_ids: List[str] = []
        self.text_by_id: Dict[str, str] = {}
        self.title_by_id: Dict[str, str] = {}
        self.brand_by_id: Dict[str, str] = {}
        self.color_by_id: Dict[str, str] = {}

        # Reusable helpers imported from src at load() time.
        self._simple_tokenize = None
        self._build_feature_row = None
        self._build_feature_matrix = None

        # Warm cross-encoder cache {query: {product_id: score}} (read-only).
        self._warm_ce: Dict[str, Dict[str, float]] = {}

    # ======================================================================
    # Startup / loading
    # ======================================================================
    def _bootstrap_paths_and_env(self) -> None:
        """Put src/ + evaluation/ on sys.path and set Hugging Face offline."""
        for path in (self.settings.src_dir, self.settings.evaluation_dir):
            p = str(path)
            if p not in sys.path:
                sys.path.insert(0, p)
        if self.settings.hf_hub_offline:
            # The Week notes document that this machine's HF download CDN is
            # blocked; force offline so the libs use the local/cached models.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    @staticmethod
    @contextlib.contextmanager
    def _suppress_stdout():
        """Silence the reusable modules' per-call ``print`` progress lines."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            yield

    def load(self) -> None:
        """
        Build the full pipeline from the existing artifacts. Never raises: a
        failure is recorded so the app can start in a 'degraded' state and
        /health can report exactly what is missing, while search endpoints
        return a clean 503.
        """
        t0 = time.time()
        self._started_at = t0
        try:
            self._bootstrap_paths_and_env()
            self._load_impl()
            self._ready = all(ok for ok, _ in self._component_status.values())
            if self._ready:
                logger.info("SearchService ready in %.1fs (corpus=%d products)",
                            time.time() - t0, len(self.product_ids))
            else:
                missing = [n for n, (ok, _) in self._component_status.items() if not ok]
                logger.error("SearchService started DEGRADED; missing: %s", missing)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            self._ready = False
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("SearchService failed to load: %s", self._load_error)

    def _mark(self, name: str, ok: bool, detail: str = "") -> None:
        self._component_status[name] = (ok, detail)

    def _load_impl(self) -> None:
        import numpy as np
        import pandas as pd

        s = self.settings

        # --- Reusable modules from src/ (heavy imports live here) -----------
        from tfidf_retriever import TfidfRetriever
        from hybrid_retriever import HybridRetriever
        from cross_encoder_reranker import CrossEncoderReranker
        from ltr_ranker import LTRRanker
        from ltr_features import build_feature_row, build_feature_matrix
        from bm25_retriever import simple_tokenize

        self._simple_tokenize = simple_tokenize
        self._build_feature_row = build_feature_row
        self._build_feature_matrix = build_feature_matrix

        # --- 1) Load the Week 0 sample and build the corpus + metadata ------
        if not s.data_file.exists():
            self._mark("data", False, f"missing {s.data_file}")
            raise ModelNotReadyError(f"Dataset not found: {s.data_file}")
        logger.info("Loading dataset: %s", s.data_file)
        df = pd.read_parquet(s.data_file)
        corpus_df = df.drop_duplicates(subset="product_id")[["product_id", "product_text"]]
        self.product_ids = corpus_df["product_id"].astype(str).tolist()
        product_texts = corpus_df["product_text"].astype(str).tolist()
        self.text_by_id = dict(zip(self.product_ids, product_texts))

        meta = df.drop_duplicates(subset="product_id")
        ids = meta["product_id"].astype(str)
        self.title_by_id = dict(zip(ids, meta["product_title"].astype(str)))
        self.brand_by_id = dict(zip(ids, meta["product_brand"].astype(str)))
        self.color_by_id = dict(zip(ids, meta["product_color"].astype(str)))
        self._mark("data", True, f"{len(self.product_ids):,} unique products")
        del df, meta, corpus_df

        # --- 2) TF-IDF index (fast) -----------------------------------------
        with self._suppress_stdout():
            self.tfidf = TfidfRetriever().fit(product_texts, self.product_ids)
        self._mark("tfidf", True, "fitted")

        # --- 3) Hybrid retriever (BM25 + embeddings + FAISS) ----------------
        cached_vectors = None
        if s.embeddings_vectors_file.exists() and s.embeddings_ids_file.exists():
            cached_ids = np.load(s.embeddings_ids_file, allow_pickle=True)
            if list(map(str, cached_ids)) == list(map(str, self.product_ids)):
                cached_vectors = np.load(s.embeddings_vectors_file)
                logger.info("Reusing cached Week 2 embeddings (no re-encoding).")
            else:
                logger.warning("Embedding cache mismatch; products would be re-encoded.")
        self.hybrid = HybridRetriever(
            alpha=s.hybrid_alpha, beta=s.hybrid_beta,
            candidate_pool_size=s.candidate_pool_size,
            embedding_model_name=s.embedding_model_name,
        )
        self.hybrid.embedder.batch_size = s.embedding_batch_size
        with self._suppress_stdout():
            self.hybrid.fit(product_texts, self.product_ids,
                            precomputed_embeddings=cached_vectors)
        self._mark("hybrid", True,
                   "reused cached embeddings" if cached_vectors is not None
                   else "encoded embeddings")

        # --- 4) Cross-encoder re-ranker (offline local model) ---------------
        ce_model = s.resolved_cross_encoder()
        with self._suppress_stdout():
            self.reranker = CrossEncoderReranker(
                model_name=ce_model, batch_size=s.cross_encoder_batch_size)
        self._mark("cross_encoder", True, f"loaded {os.path.basename(str(ce_model))}")

        # --- 5) Learning-to-Rank model (trained LambdaMART) -----------------
        if not s.ltr_model_file.exists():
            self._mark("ltr", False, f"missing {s.ltr_model_file}")
            raise ModelNotReadyError(f"LTR model not found: {s.ltr_model_file}")
        self.ltr = LTRRanker().load(str(s.ltr_model_file))
        self._mark("ltr", True, f"loaded {s.ltr_model_file.name}")

        # --- 6) Warm cross-encoder cache (read-only) ------------------------
        self._load_warm_ce_cache()

    def _load_warm_ce_cache(self) -> None:
        """Load the Week 5 CE score cache as a warm lookup (never written)."""
        path = self.settings.ce_cache_file
        if not path.exists():
            self._mark("ce_warm_cache", True, "absent (optional)")
            return
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            scores = blob.get("scores", {}) if isinstance(blob, dict) else {}
            self._warm_ce = {str(q): {str(p): float(v) for p, v in d.items()}
                             for q, d in scores.items()}
            self._mark("ce_warm_cache", True,
                       f"{len(self._warm_ce):,} queries pre-scored")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read warm CE cache (%s): %s",
                           type(exc).__name__, exc)
            self._mark("ce_warm_cache", True, "unreadable (ignored)")

    # ======================================================================
    # Readiness / introspection
    # ======================================================================
    @property
    def ready(self) -> bool:
        return self._ready

    def _require_ready(self) -> None:
        if not self._ready:
            detail = self._load_error or "pipeline still loading or unavailable"
            raise ModelNotReadyError(detail)

    def health(self) -> Dict[str, Any]:
        components = [
            {"name": n, "ready": ok, "detail": d or None}
            for n, (ok, d) in sorted(self._component_status.items())
        ]
        return {
            "status": "ok" if self._ready else "degraded",
            "ready": self._ready,
            "environment": self.settings.environment,
            "version": self.settings.app_version,
            "uptime_seconds": round(time.time() - self._started_at, 3),
            "corpus_size": len(self.product_ids),
            "components": components,
        }

    def version_info(self) -> Dict[str, Any]:
        return {
            "name": self.settings.app_name,
            "version": self.settings.app_version,
            "environment": self.settings.environment,
            "pipeline": ["TF-IDF", "BM25", "Embeddings", "FAISS", "Hybrid",
                         "CrossEncoder", "Learning-to-Rank"],
            "embedding_model": self.settings.embedding_model_name,
            "cross_encoder_model": self.settings.resolved_cross_encoder(),
            "ltr_model": str(self.settings.ltr_model_file.name),
            "supported_methods": list(self.settings.supported_methods),
        }

    def cache_stats(self) -> Dict[str, Any]:
        return self.cache.stats()

    # ======================================================================
    # Validation
    # ======================================================================
    def _validate_query(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            raise EmptyQueryError("query must not be empty or whitespace only")
        if len(q) > self.settings.max_query_length:
            raise QueryTooLongError(
                f"query length {len(q)} exceeds the maximum of "
                f"{self.settings.max_query_length} characters")
        return q

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        k = self.settings.default_top_k if top_k is None else int(top_k)
        if k < 1:
            raise InvalidParameterError("top_k must be a positive integer")
        if k > self.settings.max_top_k:
            raise InvalidParameterError(
                f"top_k {k} exceeds the maximum of {self.settings.max_top_k}")
        return k

    def _resolve_method(self, method: str) -> str:
        m = (method or "").strip().lower()
        if m not in self.settings.supported_methods:
            raise UnsupportedMethodError(
                f"method '{m}' is not supported; choose one of "
                f"{sorted(self.settings.supported_methods)}")
        return m

    def _resolve_depth(self, depth: Optional[int]) -> int:
        d = self.settings.candidate_depth if depth is None else int(depth)
        if d < 1:
            raise InvalidParameterError("candidate_depth must be positive")
        if d > self.settings.candidate_pool_size:
            raise InvalidParameterError(
                f"candidate_depth {d} exceeds candidate_pool_size "
                f"{self.settings.candidate_pool_size}")
        return d

    @staticmethod
    def _resolve_weight(value: Optional[float], default: float) -> float:
        if value is None:
            return default
        if not (0.0 <= value <= 1.0):
            raise InvalidParameterError("alpha/beta must be within [0, 1]")
        return float(value)

    # ======================================================================
    # Confidence transform (documented in schemas/responses.py)
    # ======================================================================
    def _confidence(self, score: float, method: str) -> float:
        if method in ("rerank", "ltr"):
            # Raw margin/logit -> logistic probability (clamped to avoid overflow).
            x = max(-30.0, min(30.0, float(score)))
            return round(1.0 / (1.0 + math.exp(-x)), 6)
        if method == "bm25":
            k = self.settings.bm25_confidence_saturation
            s = max(0.0, float(score))
            return round(s / (s + k), 6)
        # tfidf / embedding / hybrid are cosine/normalized ~ [0, 1].
        return round(min(1.0, max(0.0, float(score))), 6)

    def _build_items(self, ranked: List[Tuple[str, float, int]],
                     method: str) -> List[Dict[str, Any]]:
        items = []
        for pid, score, rank in ranked:
            items.append({
                "rank": int(rank),
                "product_id": str(pid),
                "title": self.title_by_id.get(str(pid), ""),
                "score": round(float(score), 6),
                "confidence": self._confidence(score, method),
            })
        return items

    def _envelope(self, method: str, query: str, top_k: int,
                  items: List[Dict[str, Any]], ms: float, cached: bool) -> Dict[str, Any]:
        return {
            "query": query,
            "ranking_method": method,
            "top_k": top_k,
            "count": len(items),
            "processing_time_ms": round(ms, 3),
            "cached": cached,
            "results": items,
        }

    # ======================================================================
    # Stage-1 hybrid candidate computation (reused by hybrid/rerank/ltr)
    # ======================================================================
    def _hybrid_candidates(self, query: str, depth: int, alpha: float, beta: float):
        """
        Return (ranked, norm_by_pid) for one query:
          ranked      : Top-`depth` [(product_id, hybrid_score, hybrid_rank)]
          norm_by_pid : {product_id: (bm25_norm, emb_norm)} over the pool
        Uses the SAME HybridRetriever methods Week 3/5 used.
        """
        with self._suppress_stdout():
            candidates = self.hybrid.precompute_candidates([query], pure_top_k=depth)
            ranked = self.hybrid.rank_from_candidates(
                candidates, alpha=alpha, beta=beta, top_k=depth)[0]
        cand = candidates[0]
        pool_ids = [str(p) for p in cand["product_ids"]]
        norm_by_pid = {
            pid: (float(cand["bm25_norm"][j]), float(cand["emb_norm"][j]))
            for j, pid in enumerate(pool_ids)
        }
        return ranked, norm_by_pid

    def _cross_encoder_scores(self, query: str,
                              cand_ids: List[str]) -> Dict[str, float]:
        """
        Cross-encoder score for each candidate. Reuses the warm Week 5 cache for
        any (query, product) already scored there; only genuinely NEW pairs hit
        the model. Truncates product text exactly as Week 5 did (for speed).
        """
        warm = self._warm_ce.get(query, {})
        scores: Dict[str, float] = {}
        missing_ids, missing_pairs = [], []
        maxchars = self.settings.cross_encoder_max_chars
        for pid in cand_ids:
            if pid in warm:
                scores[pid] = warm[pid]
            else:
                text = self.text_by_id.get(pid, "")
                if maxchars is not None:
                    text = text[:maxchars]
                missing_ids.append(pid)
                missing_pairs.append([query, text])
        if missing_pairs:
            with self._suppress_stdout():
                predicted = self.reranker.score_pairs(missing_pairs)
            for pid, sc in zip(missing_ids, predicted):
                scores[pid] = float(sc)
        return scores

    # ======================================================================
    # Public search methods (each returns the response envelope dict)
    # ======================================================================
    def search(self, query: str, top_k: Optional[int] = None,
               method: str = "hybrid") -> Dict[str, Any]:
        """Generic entrypoint: dispatch to the retriever named by ``method``."""
        self._require_ready()
        m = self._resolve_method(method)
        if m == "hybrid":
            return self.hybrid_search(query, top_k)
        if m == "rerank":
            return self.rerank(query, top_k)
        if m == "ltr":
            return self.ltr_search(query, top_k)
        # Single-stage retrievers.
        q = self._validate_query(query)
        k = self._resolve_top_k(top_k)
        key = (m, q, k)
        cached_items = self.cache.get(key)
        with Timer() as t:
            if cached_items is not None:
                return self._envelope(m, q, k, cached_items, t.elapsed_ms, True)
            with self._suppress_stdout():
                if m == "tfidf":
                    ranked = self.tfidf.retrieve(q, top_k=k)
                elif m == "bm25":
                    ranked = self.hybrid.bm25.retrieve(q, top_k=k)
                else:  # embedding
                    ranked = self.hybrid.embedder.retrieve(q, top_k=k)
            items = self._build_items(ranked, m)
            self.cache.set(key, items)
        return self._envelope(m, q, k, items, t.ms, False)

    def hybrid_search(self, query: str, top_k: Optional[int] = None,
                      alpha: Optional[float] = None,
                      beta: Optional[float] = None) -> Dict[str, Any]:
        """Week 3 hybrid retrieval (weighted BM25 + embedding fusion)."""
        self._require_ready()
        q = self._validate_query(query)
        k = self._resolve_top_k(top_k)
        a = self._resolve_weight(alpha, self.settings.hybrid_alpha)
        b = self._resolve_weight(beta, self.settings.hybrid_beta)
        key = ("hybrid", q, k, a, b)
        cached_items = self.cache.get(key)
        with Timer() as t:
            if cached_items is not None:
                return self._envelope("hybrid", q, k, cached_items, t.elapsed_ms, True)
            ranked, _ = self._hybrid_candidates(q, k, a, b)
            items = self._build_items(ranked, "hybrid")
            self.cache.set(key, items)
        return self._envelope("hybrid", q, k, items, t.ms, False)

    def rerank(self, query: str, top_k: Optional[int] = None,
               candidate_depth: Optional[int] = None) -> Dict[str, Any]:
        """Week 4 pipeline: hybrid Top-N -> cross-encoder -> Top-K."""
        self._require_ready()
        q = self._validate_query(query)
        k = self._resolve_top_k(top_k)
        depth = self._resolve_depth(candidate_depth)
        a, b = self.settings.hybrid_alpha, self.settings.hybrid_beta
        key = ("rerank", q, k, depth)
        cached_items = self.cache.get(key)
        with Timer() as t:
            if cached_items is not None:
                return self._envelope("rerank", q, k, cached_items, t.elapsed_ms, True)
            ranked, _ = self._hybrid_candidates(q, depth, a, b)
            cand_ids = [pid for (pid, _s, _r) in ranked]
            ce_scores = self._cross_encoder_scores(q, cand_ids)
            ordered = sorted(cand_ids, key=lambda pid: -ce_scores.get(pid, float("-inf")))
            reranked = [(pid, ce_scores.get(pid, 0.0), rank)
                        for rank, pid in enumerate(ordered[:k], start=1)]
            items = self._build_items(reranked, "rerank")
            self.cache.set(key, items)
        return self._envelope("rerank", q, k, items, t.ms, False)

    def ltr_search(self, query: str, top_k: Optional[int] = None,
                   candidate_depth: Optional[int] = None) -> Dict[str, Any]:
        """Week 5 full pipeline: hybrid -> cross-encoder -> LTR -> Top-K."""
        self._require_ready()
        q = self._validate_query(query)
        k = self._resolve_top_k(top_k)
        depth = self._resolve_depth(candidate_depth)
        a, b = self.settings.hybrid_alpha, self.settings.hybrid_beta
        key = ("ltr", q, k, depth)
        cached_items = self.cache.get(key)
        with Timer() as t:
            if cached_items is not None:
                return self._envelope("ltr", q, k, cached_items, t.elapsed_ms, True)
            ranked, norm_by_pid = self._hybrid_candidates(q, depth, a, b)
            cand_ids = [pid for (pid, _s, _r) in ranked]
            ce_scores = self._cross_encoder_scores(q, cand_ids)

            # ce_rank feature: candidates ordered by cross-encoder score (Week 5).
            ce_order = sorted(cand_ids, key=lambda pid: -ce_scores.get(pid, float("-inf")))
            ce_rank_by_pid = {pid: rank for rank, pid in enumerate(ce_order, start=1)}

            query_tokens = self._simple_tokenize(q)
            rows = []
            for (pid, hybrid_score, hybrid_rank) in ranked:
                bm25_n, emb_n = norm_by_pid.get(pid, (0.0, 0.0))
                rows.append(self._build_feature_row(
                    query_tokens=query_tokens, query_str=q,
                    bm25_norm=bm25_n, emb_norm=emb_n,
                    hybrid_score=hybrid_score,
                    ce_score=ce_scores.get(pid, 0.0),
                    ce_rank=ce_rank_by_pid.get(pid, len(cand_ids)),
                    hybrid_rank=hybrid_rank,
                    title=self.title_by_id.get(pid, ""),
                    brand=self.brand_by_id.get(pid, ""),
                    color=self.color_by_id.get(pid, ""),
                    full_text=self.text_by_id.get(pid, ""),
                ))
            fmat = self._build_feature_matrix(rows)
            ltr_ranked = self.ltr.rank_candidates(cand_ids, fmat, top_k=k)
            items = self._build_items(ltr_ranked, "ltr")
            self.cache.set(key, items)
        return self._envelope("ltr", q, k, items, t.ms, False)
