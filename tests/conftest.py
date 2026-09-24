"""Shared test configuration.

Some tests in this directory load an experiment script from `extras/scripts/`
by path. Those scripts are part of the research repo, not the package, so
those particular tests are skipped there rather than erroring. Archived
library modules in `extras/*.py` are always imported and tested.

When `extras/scripts/` is present this file does nothing.
"""
import inspect
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "extras" / "scripts"
_MARKER = '"extras" / "scripts"'


def _source(obj):
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def _needs_scripts(item):
    """True if this test loads a script, directly or via a module helper."""
    func = getattr(item, "function", None)
    if func is None:
        return False
    src = _source(func)
    if _MARKER in src:
        return True

    module = getattr(item, "module", None)
    if module is None:
        return False
    for name, obj in vars(module).items():
        if not name.startswith("_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        # a helper that loads a script, and this test calls it
        if name in src and _MARKER in _source(obj):
            return True
    return False


def pytest_collection_modifyitems(config, items):
    if SCRIPTS_DIR.is_dir():
        return
    skip = pytest.mark.skip(
        reason="needs extras/scripts, which is not in this checkout "
               "(package-only distribution)"
    )
    for item in items:
        if _needs_scripts(item):
            item.add_marker(skip)
