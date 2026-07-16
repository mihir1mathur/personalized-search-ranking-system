"""
settings.py  --  Central configuration for the Week 6 search API.
=================================================================

WHY THIS FILE EXISTS
--------------------
A production service must never scatter "magic numbers" and file paths through
its code. Every value that an operator might want to change per environment
(dev / staging / prod) lives HERE, in one typed, validated place:

    * default / maximum Top-K
    * hybrid weights (alpha / beta)
    * candidate depth for re-ranking and LTR
    * batch sizes and inference device
    * all model / cache / embedding / data paths
    * query-cache size and TTL
    * validation limits (query length, allowed methods)
    * logging configuration (level, file, rotation)

Values are read from environment variables (prefix ``SEARCH_``) or an optional
``.env`` file, then validated by Pydantic. Because it is a ``BaseSettings``
model, an operator can override anything without touching code, e.g.::

    SEARCH_DEFAULT_TOP_K=20  SEARCH_ENVIRONMENT=production  uvicorn api.main:app

The default paths point at the SAME Week 0-5 artifacts that already exist in
the repo, so the API reuses every model, cache, embedding, and index. Nothing
here regenerates or modifies those artifacts.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Repository layout. This file lives at <root>/api/config/settings.py, so the
# project root is three parents up. Everything else is derived from it, which
# keeps the config portable across machines (no absolute paths committed).
# --------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]


class Settings(BaseSettings):
    """All configurable values for the API, validated once at startup."""

    model_config = SettingsConfigDict(
        env_prefix="SEARCH_",
        env_file=os.environ.get("SEARCH_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # we intentionally use `model_*` field names
    )

    # ---- Service identity / environment -----------------------------------
    app_name: str = "Personalized Search Ranking API"
    app_version: str = "6.0.0"
    environment: str = Field(
        default="development",
        description="Deployment environment: development | staging | production",
    )
    api_prefix: str = Field(default="", description="Optional path prefix, e.g. /api/v1")

    # ---- Server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # ---- Retrieval / ranking knobs (reuse Week 0-5 defaults) ---------------
    default_top_k: int = Field(default=10, ge=1)
    max_top_k: int = Field(default=100, ge=1, description="Hard ceiling on top_k")
    # Stage-1 candidate pool taken from BM25 + embeddings (Week 3 default).
    candidate_pool_size: int = Field(default=100, ge=10)
    # Depth of candidates re-scored by the cross-encoder / LTR (Week 4-5 = 50).
    candidate_depth: int = Field(default=50, ge=1)

    # Best hybrid weights selected in Week 3/5 (alpha = BM25, beta = embeddings).
    hybrid_alpha: float = Field(default=0.4, ge=0.0, le=1.0)
    hybrid_beta: float = Field(default=0.6, ge=0.0, le=1.0)

    # Inference batch sizes (mirror the Week 2/4 defaults).
    embedding_batch_size: int = Field(default=64, ge=1)
    cross_encoder_batch_size: int = Field(default=32, ge=1)
    # Cross-encoder input is truncated to this many chars for speed (Week 5).
    cross_encoder_max_chars: int = Field(default=400, ge=1)

    device: str = Field(default="cpu", description="Inference device: cpu | cuda")

    # Confidence mapping: BM25 scores are unbounded, so a saturation constant
    # turns them into a 0..1 confidence via score/(score+k). Larger k => the
    # same raw score maps to a lower confidence. Cross-encoder/LTR use a
    # logistic; cosine-based methods (tfidf/embedding/hybrid) are clamped to 0..1.
    bm25_confidence_saturation: float = Field(default=8.0, gt=0.0)

    # ---- Validation limits -------------------------------------------------
    min_query_length: int = Field(default=1, ge=1)
    max_query_length: int = Field(default=256, ge=1)
    supported_methods: List[str] = Field(
        default=["tfidf", "bm25", "embedding", "hybrid", "rerank", "ltr"]
    )

    # ---- Query cache -------------------------------------------------------
    cache_enabled: bool = True
    cache_max_size: int = Field(default=512, ge=1)
    cache_ttl_seconds: int = Field(default=300, ge=0)

    # ---- Model / data / cache paths (relative to the repo root) ------------
    # Stored as strings so they are easy to override via env; resolved to
    # absolute Paths by the properties below.
    data_path: str = "data/processed/sample_esci_50k.parquet"
    models_dir: str = "models"
    results_dir: str = "results"
    ltr_model_path: str = "models/ltr_lightgbm.txt"
    cross_encoder_dir: str = "models/ms-marco-MiniLM-L-6-v2"
    cross_encoder_hub_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_vectors_path: str = "results/product_embeddings_minilm.npy"
    embeddings_ids_path: str = "results/product_embeddings_ids.npy"
    ce_cache_path: str = "results/week5_ce_cache.pkl"

    # ---- Hugging Face offline (this machine's CDN is blocked; see memory) ---
    hf_hub_offline: bool = True

    # ---- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/api.log"
    log_max_bytes: int = Field(default=5_000_000, ge=1024)   # ~5 MB per file
    log_backup_count: int = Field(default=5, ge=0)            # rotate, keep 5
    log_json: bool = False

    # ---- Testing -----------------------------------------------------------
    # When true, the app lifespan does NOT build the heavy service (models are
    # not loaded); tests inject a fake service instead. Keeps unit tests fast.
    testing: bool = False

    # ----------------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------------
    @field_validator("environment")
    @classmethod
    def _valid_environment(cls, value: str) -> str:
        allowed = {"development", "staging", "production"}
        if value not in allowed:
            raise ValueError(f"environment must be one of {sorted(allowed)}")
        return value

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        value = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return value

    @field_validator("supported_methods")
    @classmethod
    def _lower_methods(cls, value: List[str]) -> List[str]:
        return [m.lower() for m in value]

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.max_top_k < self.default_top_k:
            raise ValueError("max_top_k must be >= default_top_k")
        if self.max_query_length < self.min_query_length:
            raise ValueError("max_query_length must be >= min_query_length")
        # alpha + beta need not sum to 1, but warn-worthy configs are rejected.
        if self.candidate_depth > self.candidate_pool_size:
            # candidate_depth draws from the stage-1 pool, so it cannot exceed it.
            raise ValueError("candidate_depth must be <= candidate_pool_size")
        return self

    # ----------------------------------------------------------------------
    # Absolute-path helpers (resolve every configured path against the root).
    # ----------------------------------------------------------------------
    def _abs(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def src_dir(self) -> Path:
        return PROJECT_ROOT / "src"

    @property
    def evaluation_dir(self) -> Path:
        return PROJECT_ROOT / "evaluation"

    @property
    def data_file(self) -> Path:
        return self._abs(self.data_path)

    @property
    def ltr_model_file(self) -> Path:
        return self._abs(self.ltr_model_path)

    @property
    def cross_encoder_path(self) -> Path:
        return self._abs(self.cross_encoder_dir)

    @property
    def embeddings_vectors_file(self) -> Path:
        return self._abs(self.embeddings_vectors_path)

    @property
    def embeddings_ids_file(self) -> Path:
        return self._abs(self.embeddings_ids_path)

    @property
    def ce_cache_file(self) -> Path:
        return self._abs(self.ce_cache_path)

    @property
    def log_file_path(self) -> Path:
        return self._abs(self.log_file)

    @property
    def hybrid_weights(self) -> Tuple[float, float]:
        return (self.hybrid_alpha, self.hybrid_beta)

    def resolved_cross_encoder(self) -> str:
        """Prefer the local model dir (offline friendly), else the hub name."""
        local = self.cross_encoder_path
        return str(local) if local.is_dir() else self.cross_encoder_hub_name


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (built once per process)."""
    return Settings()
