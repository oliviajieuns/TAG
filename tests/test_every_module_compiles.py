"""Every module and script must at least compile.

A syntax error shipped to main in tag/eval.py — an `elif` after an `except`
— and the whole test suite stayed green, because nothing imports the eval
entry point. It was found by a human running it on the cluster, which is
the most expensive place to find it.

This is not a substitute for testing behaviour. It is the floor: no file in
the repo may be un-parseable, and the check costs milliseconds.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DIRS = ("tag", "scripts", "tests")


def _python_files():
    for d in _DIRS:
        for p in sorted((_ROOT / d).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


@pytest.mark.parametrize(
    "path", list(_python_files()), ids=lambda p: str(p.relative_to(_ROOT)),
)
def test_module_parses(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src, filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path.relative_to(_ROOT)}:{e.lineno}: {e.msg}\n"
                    f"  {(e.text or '').rstrip()}")


def test_eval_entry_point_imports() -> None:
    """tag.eval specifically: it is the module the syntax error shipped in,
    and it is only ever exercised by `python -m tag.eval` on a GPU box."""
    import importlib
    importlib.import_module("tag.eval")
