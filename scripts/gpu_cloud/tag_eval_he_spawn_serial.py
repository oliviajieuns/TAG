#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import human_eval.evaluation as _he

PATCH_ID = "tag-he-fresh-spawn-serial-v1"
_ORIGINAL = _he.evaluate_functional_correctness
_SELF = Path(__file__).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in tuple(env):
        upper = key.upper()
        if any(word in upper for word in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
            env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    return env


def _spawn_and_wait(argv: list[str], env: dict[str, str]) -> int:
    if not hasattr(os, "posix_spawn"):
        raise RuntimeError("This wrapper requires Linux/POSIX os.posix_spawn")
    pid = os.posix_spawn(sys.executable, argv, env)
    while True:
        try:
            _, status = os.waitpid(pid, 0)
            return os.waitstatus_to_exitcode(status)
        except InterruptedError:
            continue


def _fresh_spawn_serial(*args: Any, **kwargs: Any) -> dict[str, float]:
    bound = inspect.signature(_ORIGINAL).bind(*args, **kwargs)
    bound.apply_defaults()
    call = dict(bound.arguments)
    call["sample_file"] = os.fspath(call["sample_file"])
    call["problem_file"] = os.fspath(call["problem_file"])
    call["k"] = [int(value) for value in call["k"]]
    call["timeout"] = float(call["timeout"])
    call["n_workers"] = 1

    with tempfile.TemporaryDirectory(prefix="tag-he-request-") as tmpdir:
        root = Path(tmpdir)
        request = root / "request.json"
        result = root / "result.json"
        request.write_text(
            json.dumps({"patch_id": PATCH_ID, "parent_pid": os.getpid(), "kwargs": call}),
            encoding="utf-8",
        )
        argv = [sys.executable, str(_SELF), "--tag-he-score", str(request), str(result)]
        returncode = _spawn_and_wait(argv, _clean_env())
        payload = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
        if returncode != 0 or not payload.get("ok"):
            raise RuntimeError(
                f"fresh HumanEval scorer failed: rc={returncode}, "
                f"error={payload.get('error', 'no result.json')}"
            )
        meta = payload["meta"]
        if meta["scorer_pid"] == os.getpid() or meta["n_workers"] != 1 or meta["start_method"] != "spawn":
            raise RuntimeError(f"HumanEval scorer isolation check failed: {meta}")
        print(
            f"[tag-safe-he] patch={PATCH_ID} parent_pid={os.getpid()} "
            f"scorer_pid={meta['scorer_pid']} start_method={meta['start_method']} "
            f"n_workers={meta['n_workers']} filelock_loaded={meta['filelock_loaded']}",
            flush=True,
        )
        return {str(key): float(value) for key, value in payload["scores"].items()}


_fresh_spawn_serial.__name__ = _ORIGINAL.__name__
_fresh_spawn_serial.__doc__ = _ORIGINAL.__doc__
_fresh_spawn_serial._tag_patch_id = PATCH_ID
_he.evaluate_functional_correctness = _fresh_spawn_serial


def _score_mode(request: Path, result: Path) -> int:
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    try:
        payload = json.loads(request.read_text(encoding="utf-8"))
        if payload.get("patch_id") != PATCH_ID:
            raise ValueError("patch ID mismatch")
        call = dict(payload["kwargs"])
        call["n_workers"] = 1
        meta = {
            "parent_pid": int(payload["parent_pid"]),
            "scorer_pid": os.getpid(),
            "n_workers": 1,
            "start_method": mp.get_start_method(),
            "filelock_loaded": "filelock" in sys.modules,
        }
        scores = {str(key): float(value) for key, value in _ORIGINAL(**call).items()}
        _atomic_json(result, {"ok": True, "scores": scores, "meta": meta})
        return 0
    except BaseException as exc:
        traceback.print_exc()
        _atomic_json(
            result,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def _smoke() -> int:
    with tempfile.TemporaryDirectory(prefix="tag-he-smoke-") as tmpdir:
        root = Path(tmpdir)
        problems = root / "problems.jsonl"
        samples = root / "samples.jsonl"
        problem = {
            "task_id": "smoke/0",
            "prompt": 'def answer():\n    """Return 42."""\n',
            "entry_point": "answer",
            "canonical_solution": "    return 42\n",
            "test": "def check(candidate):\n    assert candidate() == 42\n",
        }
        problems.write_text(json.dumps(problem) + "\n", encoding="utf-8")
        samples.write_text(
            json.dumps({"task_id": "smoke/0", "completion": "    return 42\n"}) + "\n",
            encoding="utf-8",
        )
        scores = _he.evaluate_functional_correctness(
            sample_file=str(samples),
            problem_file=str(problems),
            k=[1],
            timeout=5,
            n_workers=99,
        )
        if abs(scores.get("pass@1", -1.0) - 1.0) > 1e-12:
            raise RuntimeError(f"HumanEval smoke score mismatch: {scores}")
        digest = hashlib.sha256(_SELF.read_bytes()).hexdigest()
        print(f"[tag-safe-he] SMOKE_OK pass@1=1.0 wrapper_sha256={digest}", flush=True)
        return 0


def _main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--tag-he-score":
        return _score_mode(Path(sys.argv[2]), Path(sys.argv[3]))
    if sys.argv[1:] == ["--smoke-human-eval-scorer"]:
        return _smoke()
    print(f"[tag-safe-he] patch={PATCH_ID} wrapper={_SELF}", flush=True)
    from tag.eval import main as tag_eval_main

    tag_eval_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
