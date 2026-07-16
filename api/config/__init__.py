"""Configuration package: every tunable value lives here (no hardcoded constants)."""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
