"""Make the frontend modules importable in tests (add frontend/ to sys.path)."""

import os
import sys

FRONTEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)
