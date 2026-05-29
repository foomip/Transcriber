"""
conftest.py — Project-wide pytest configuration.

Makes the hyphen-named youtube-summarize.py importable as youtube_summarize
by loading it via importlib and registering it in sys.modules before any
test module that imports it is collected.
"""

import importlib.util
import sys
from pathlib import Path


def _register_hyphenated_module(stem: str, filename: str) -> None:
    """Load *filename* from the project root and register it as *stem*."""
    if stem in sys.modules:
        return
    source = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(stem, source)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]


_register_hyphenated_module("youtube_summarize", "youtube-summarize.py")
