"""Unit tests for the query cache and settings validation (pure logic)."""

from __future__ import annotations

import time

import pytest

from api.config.settings import Settings
from api.services.cache import QueryCache


def test_cache_hit_and_miss():
    cache = QueryCache(max_size=10, ttl_seconds=0)
    assert cache.get(("a",)) is None
    cache.set(("a",), [1, 2, 3])
    assert cache.get(("a",)) == [1, 2, 3]
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_cache_lru_eviction():
    cache = QueryCache(max_size=2, ttl_seconds=0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")            # touch 'a' so 'b' is now least-recent
    cache.set("c", 3)         # evicts 'b'
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None


def test_cache_ttl_expiry():
    cache = QueryCache(max_size=10, ttl_seconds=1)
    cache.set("a", 1)
    assert cache.get("a") == 1
    time.sleep(1.1)
    assert cache.get("a") is None


def test_cache_disabled_is_noop():
    cache = QueryCache(enabled=False)
    cache.set("a", 1)
    assert cache.get("a") is None


def test_settings_reject_bad_environment():
    with pytest.raises(Exception):
        Settings(environment="prod-typo")


def test_settings_reject_depth_over_pool():
    with pytest.raises(Exception):
        Settings(candidate_depth=200, candidate_pool_size=100)


def test_settings_paths_resolve_absolute():
    s = Settings()
    assert s.data_file.is_absolute()
    assert s.ltr_model_file.name == "ltr_lightgbm.txt"
    assert s.hybrid_weights == (s.hybrid_alpha, s.hybrid_beta)
