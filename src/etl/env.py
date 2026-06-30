"""Load API keys from the project .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_LOADED = False


def load_project_env() -> None:
    """Load .env from project root into os.environ (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    _ENV_LOADED = True
