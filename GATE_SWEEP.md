# Table-2 clean-pool gate sweep

This branch adds a matched gate-strength experiment without recomputing the
LLaMA-2 raw gate cache.

| arm | effective gate | exact zero? | role |
|---|---:|:---:|---|
| strong (existing `run.sh`) | `G` | yes | canonical TAG |
| `weak.sh` | `sqrt(G)` | yes | primary weaker TAG |
| `soft.sh` | `0.5 + 0.5 G` | no | soft-gate diagnostic |
| `ctl.sh` | none | n/a | matched R x A control |

Every new arm uses effective batch 64, cosine LR floor 0.10, fp32 AdamW with
`foreach=false`, the same 52,002-row clean pool, and the same three seeds.
The soft arm is an ablation only: it removes TAG's non-compensation guarantee
and must not be labelled canonical TAG.

## Fast 10-GPU launch

Start the primary weak arm on the three allocated nodes:

```bash
# 2-GPU node
S=1 bash weak.sh

# first 4-GPU node
S=7 bash weak.sh

# second 4-GPU node
S=42 bash weak.sh
```

Training automatically starts a full eight-benchmark evaluation on that node.
The same three commands with `soft.sh` or `ctl.sh` launch the diagnostic arm or
matched control.  They can be run in the next wave after `weak.sh` completes.

Status and manual evaluation restart stay short:

```bash
S=42 bash weak.sh status
S=42 bash weak.sh eval
```

Replace `weak.sh` with `soft.sh` or `ctl.sh` for those arms.  Existing strong
TAG and historical R x A entry points (`run.sh`, `ra.sh`) are unchanged.

When evaluations finish, one command prints per-seed scores, 3-seed means,
run-level SD, and the two-sided 95% Student-t CI:

```bash
bash sweep.sh
```

## Recorded audit fields

Every new weak/soft run (and any future strong run on this branch) records
`gate_raw_mean`, `gate_raw_zero_frac`, `gate_mean`, `gate_zero_frac`,
`gate_power`, and `gate_strength`.  This proves that each new arm read the same
raw cache while applying the intended score-time transform.  Each arm and seed
receives a private cache copy (no cross-node write race), and the launcher
verifies its raw gate/config tensors against the validated historical artifact
after training.  Older completed strong runs predate the extra `gate_raw_*`
fields and retain their original gate diagnostics.
