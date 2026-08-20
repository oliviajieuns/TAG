"""Every ``selection.tag`` key in a config must reach the gate.

``tag/train.py`` assembles ``tag_ctx["params"]`` by naming each key it
forwards, and ``_build_gate_config`` reads that dict with its own defaults.
A key present in the YAML but absent from the forwarding dict therefore does
not fail — it silently reverts to the default, and the run reports a G that
does not match its config.

That is not hypothetical. ``prefix_tokens`` was never forwarded:
``main_7b/llama2/tag_10.yaml`` set 32, the gate ran at 0, and the Table 2 TAG
row trained for three seeds on a gate that zeroed 49% of a clean pool against
a configured ``target_zero_rate`` of 0.05.

The first two tests read the source with ``ast`` rather than importing it, so
they run on a box with no torch — the same box where someone is most likely
to be editing a config and wondering why the gate ignored them.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_PY = _REPO_ROOT / "tag" / "train.py"
_SELECTION_PY = _REPO_ROOT / "tag" / "pipelines" / "selection.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _forwarded_keys() -> set[str]:
    """The literal keys of ``tag_ctx["params"]`` in tag/train.py.

    Read statically: the dict is built inside ``main()`` and only exists once
    a model and tokenizer are loaded.
    """
    for node in ast.walk(_module(_TRAIN_PY)):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if "span_tokens" in keys and "gate_cache_file" in keys:
            return keys
    raise AssertionError("could not locate tag_ctx['params'] in tag/train.py")


def _consumed_elsewhere() -> set[str]:
    """``TAG_PARAMS_CONSUMED_ELSEWHERE``, read without importing torch."""
    for node in ast.walk(_module(_TRAIN_PY)):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name)
                    and t.id == "TAG_PARAMS_CONSUMED_ELSEWHERE"
                    for t in node.targets)
        ):
            return set(ast.literal_eval(node.value.args[0]))
    raise AssertionError("TAG_PARAMS_CONSUMED_ELSEWHERE not found in tag/train.py")


def _gate_config_keys() -> set[str]:
    """Keys read via ``params.get(...)`` by the two functions that build the
    gate config from the YAML subtree."""
    wanted = {"_build_gate_config", "_resolve_gate_calibration"}
    found: set[str] = set()
    for node in ast.walk(_module(_SELECTION_PY)):
        if not (isinstance(node, ast.FunctionDef) and node.name in wanted):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "params"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
            ):
                found.add(sub.args[0].value)
    assert found, "no params.get(...) calls found — the AST probe is stale"
    return found


def test_every_gate_config_key_is_forwarded():
    """The regression: a key the gate reads but train.py does not forward."""
    missing = _gate_config_keys() - _forwarded_keys()
    assert not missing, (
        f"tag/train.py does not forward {sorted(missing)} into "
        f"tag_ctx['params'], so the gate falls back to its own defaults and "
        f"the run's G silently disagrees with its config."
    )


def test_prefix_tokens_is_forwarded():
    """Pinned by name: this is the key whose absence cost the Table 2 row."""
    assert "prefix_tokens" in _forwarded_keys()


def _tag_configs() -> list[Path]:
    return sorted(
        p for p in (_REPO_ROOT / "configs").rglob("*.yaml")
        if "tag:" in p.read_text()
    )


@pytest.mark.parametrize(
    "cfg_path", _tag_configs(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_shipped_configs_set_no_dropped_key(cfg_path):
    """No config in the repo sets a selection.tag key that never arrives.

    Needs the real loader (inheritance + env interpolation), hence torch.
    """
    pytest.importorskip("torch")
    from tag.core.utils import load_config

    tag_cfg = ((load_config(str(cfg_path)).get("selection") or {}).get("tag")) or {}
    dropped = set(tag_cfg) - _forwarded_keys() - _consumed_elsewhere()
    assert not dropped, (
        f"{cfg_path.relative_to(_REPO_ROOT)} sets {sorted(dropped)}, which "
        f"tag/train.py drops on the floor."
    )
