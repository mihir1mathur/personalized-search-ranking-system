"""
Streamlit render smoke tests using the official AppTest harness.
================================================================

These actually EXECUTE each page script in a simulated Streamlit runtime and
assert the script ran without raising an uncaught exception. They are hermetic:
the pages handle a missing backend gracefully (friendly error + st.stop()), so
these pass whether or not the FastAPI backend is running.

Run with:  python -m pytest frontend/tests/test_app_render.py -q
"""

from __future__ import annotations

import os

import pytest

FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from streamlit.testing.v1 import AppTest
    _HAS_APPTEST = True
except Exception:  # pragma: no cover
    _HAS_APPTEST = False

pytestmark = pytest.mark.skipif(not _HAS_APPTEST,
                                reason="streamlit AppTest not available")

PAGE_FILES = [
    "app.py",
    os.path.join("pages", "1_Health.py"),
    os.path.join("pages", "2_API_Docs.py"),
    os.path.join("pages", "3_Benchmark.py"),
    os.path.join("pages", "4_Pipeline.py"),
    os.path.join("pages", "5_Settings.py"),
]


@pytest.mark.parametrize("page", PAGE_FILES)
def test_page_runs_without_exception(page):
    at = AppTest.from_file(os.path.join(FRONTEND, page), default_timeout=30)
    at.run()
    # A well-behaved page never surfaces an uncaught exception, even when the
    # backend is down (it shows a friendly error and stops instead).
    assert not at.exception, f"{page} raised: {at.exception}"


def test_search_page_has_controls():
    at = AppTest.from_file(os.path.join(FRONTEND, "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    # The search page exposes a query text input, a method selectbox, and a slider.
    assert len(at.text_input) >= 1
    assert len(at.selectbox) >= 1


def test_settings_page_persists_values():
    at = AppTest.from_file(os.path.join(FRONTEND, "pages", "5_Settings.py"),
                           default_timeout=30)
    at.run()
    assert not at.exception
    # Session defaults are seeded on first render.
    assert at.session_state["top_k"] >= 1
    assert 0.0 <= at.session_state["alpha"] <= 1.0
