"""
run.py  --  convenience entrypoint to launch the API locally.
=============================================================

Equivalent to ``uvicorn api.main:app`` but reads host/port from the same
Settings object, so one config source drives everything. Usage::

    python -m api.run
    # or, with overrides:
    SEARCH_PORT=9000 python -m api.run
"""

from __future__ import annotations

import uvicorn

from api.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
