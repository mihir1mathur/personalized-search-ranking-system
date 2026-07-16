"""
requests.py  --  Pydantic request models (the input side of the contract).
==========================================================================

These models define exactly what a client may send. Pydantic validates and
coerces the JSON body BEFORE it ever reaches our service code, so malformed
requests are rejected at the edge with a clear 422 error instead of blowing up
deep inside the pipeline.

We keep two layers of validation on purpose:

  * STRUCTURAL / TYPE validation lives here (a query must be a non-empty
    string; top_k, if given, must be a positive integer). This is static and
    always enforced.
  * CONFIG-BOUND validation (max query length, max top_k, which methods are
    enabled) lives in the service, because those limits come from live
    :class:`Settings` and an operator may tighten them per environment.

``query`` is stripped and rejected if blank so " " never reaches retrieval.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Structural bounds. These mirror the Settings defaults and act as a first,
# always-on guard rail; the service re-checks the live (possibly stricter)
# limits from Settings so operators can tighten them without a code change.
_QUERY_MIN = 1
_QUERY_MAX = 256


class _BaseQueryRequest(BaseModel):
    """Fields shared by every search request."""

    query: str = Field(
        ...,
        min_length=_QUERY_MIN,
        max_length=_QUERY_MAX,
        description="The shopper's free-text search query.",
        examples=["wireless noise cancelling headphones"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="How many ranked results to return. Defaults to the "
        "server's configured default_top_k when omitted.",
        examples=[10],
    )

    @field_validator("query")
    @classmethod
    def _strip_and_check(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty or whitespace only")
        return stripped


class SearchRequest(_BaseQueryRequest):
    """
    Request for the generic ``/search`` endpoint.

    ``method`` chooses a single-stage retriever. It is validated against the
    server's ``supported_methods`` in the service layer so the enabled set can
    be configured per environment.
    """

    method: str = Field(
        default="hybrid",
        description="Ranking method: tfidf | bm25 | embedding | hybrid | "
        "rerank | ltr.",
        examples=["hybrid"],
    )

    @field_validator("method")
    @classmethod
    def _normalize_method(cls, value: str) -> str:
        return value.strip().lower()


class HybridSearchRequest(_BaseQueryRequest):
    """
    Request for ``/hybrid-search``. Optional per-request ``alpha`` (BM25 weight)
    and ``beta`` (embedding weight) override the server defaults; when omitted
    the Week 3/5 best weights are used.
    """

    alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Weight on the BM25 (keyword) signal. Defaults to the "
        "configured hybrid_alpha.",
    )
    beta: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Weight on the embedding (semantic) signal. Defaults to the "
        "configured hybrid_beta.",
    )


class RerankRequest(_BaseQueryRequest):
    """
    Request for ``/rerank``: hybrid retrieval followed by cross-encoder
    re-ranking (the Week 4 pipeline). ``candidate_depth`` optionally overrides
    how many stage-1 candidates the cross-encoder re-scores.
    """

    candidate_depth: Optional[int] = Field(
        default=None, ge=1,
        description="How many stage-1 hybrid candidates to re-score with the "
        "cross-encoder. Defaults to the configured candidate_depth.",
    )


class LTRSearchRequest(_BaseQueryRequest):
    """
    Request for ``/ltr-search``: the full Week 5 pipeline
    (hybrid -> cross-encoder -> Learning-to-Rank), the best-performing method.
    """

    candidate_depth: Optional[int] = Field(
        default=None, ge=1,
        description="How many stage-1 hybrid candidates flow into the "
        "cross-encoder + LTR stages. Defaults to the configured candidate_depth.",
    )
