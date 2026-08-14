"""`import swebench` must stay cheap and keep its public names (#525).

Eager imports in `swebench/__init__.py` pulled in bs4, docker, datasets and modal,
which made a bare import take seconds and fail outright when an optional dependency
was missing.
"""

import subprocess
import sys

import pytest

import swebench


def test_import_does_not_pull_heavy_dependencies():
    # a fresh interpreter, so nothing else in the test run can have imported these
    code = (
        "import sys; import swebench; "
        "print(','.join(m for m in ('bs4','docker','datasets','modal') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_every_public_name_resolves():
    missing = [name for name in swebench.__all__ if not hasattr(swebench, name)]
    assert missing == []


def test_all_matches_the_lazy_import_table():
    # __all__ is written out literally for editors and type checkers; keep them in step
    assert sorted(swebench.__all__) == sorted([*swebench._LAZY_IMPORTS, "__version__"])


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        _ = swebench.does_not_exist
