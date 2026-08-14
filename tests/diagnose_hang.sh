#!/usr/bin/env bash
# Diagnose a hung tag.train / tag.eval run by dumping every Python stack
# plus a GPU and process snapshot. Output goes to a single timestamped file
# under logs/ so it's easy to share / attach.
#
# Usage:
#   bash tests/diagnose_hang.sh
#   bash tests/diagnose_hang.sh --pattern tag.eval     # different entrypoint
#   bash tests/diagnose_hang.sh --out /tmp/hang.txt     # custom output path
#
# Notes:
#   - Requires py-spy. Install with `pip install py-spy` (or `--user`,
#     or `--break-system-packages` on system Python). The script will
#     try a `pip install py-spy --user` once if py-spy is missing.
#   - `py-spy dump` needs ptrace permission. Run with sudo, or run once:
#       sudo setcap cap_sys_ptrace,cap_dac_read_search=ep $(which py-spy)
#     to make subsequent runs work without sudo.
set -u

PATTERN="tag.train"
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --pattern)  PATTERN="$2"; shift 2 ;;
        --pattern=*) PATTERN="${1#*=}"; shift ;;
        --out)      OUT="$2"; shift 2 ;;
        --out=*)    OUT="${1#*=}"; shift ;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "[warn] unknown arg: $1" >&2; shift ;;
    esac
done

cd "$(dirname "$0")/.."
mkdir -p logs

TS=$(date +%Y%m%d-%H%M%S)
if [ -z "$OUT" ]; then
    OUT="logs/hang_diagnose_${TS}.txt"
fi

# --- py-spy availability ------------------------------------------------------
if ! command -v py-spy >/dev/null 2>&1; then
    echo "[diag] py-spy not found, attempting install ..."
    pip install py-spy --user 2>&1 | tail -5
    if ! command -v py-spy >/dev/null 2>&1; then
        # Fall back to user-site bin if it's not on PATH
        USERBIN="$(python -c 'import site; print(site.USER_BASE)')/bin"
        if [ -x "$USERBIN/py-spy" ]; then
            export PATH="$USERBIN:$PATH"
        else
            echo "[diag] could not install py-spy. Install manually:"
            echo "       pip install py-spy --user  OR  pip install py-spy --break-system-packages"
            exit 1
        fi
    fi
fi

PIDS=$(pgrep -f "$PATTERN")
if [ -z "$PIDS" ]; then
    echo "[diag] no live process matching '$PATTERN' — nothing to dump." | tee "$OUT"
    exit 0
fi

{
    echo "============================================================"
    echo "tag hang diagnostic — $(date)"
    echo "pattern: $PATTERN"
    echo "============================================================"
    echo ""
    echo "--- live processes matching pattern ---"
    ps -o pid,ppid,etime,stat,cmd -p $PIDS 2>&1
    echo ""
    echo "--- nvidia-smi snapshot ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv 2>&1
    echo ""
    echo "--- nvidia-smi compute apps (orphan GPU processes?) ---"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
    echo ""
    echo "--- NCCL / torch env in shell ---"
    env | grep -E "^(NCCL_|TORCH_NCCL_|RANK|LOCAL_RANK|WORLD_SIZE|MASTER_)" | sort 2>&1 || true
    echo ""

    for pid in $PIDS; do
        echo "============================================================"
        echo "[py-spy dump] PID $pid"
        echo "============================================================"
        # `sudo` only if not already root; --no-stderr if py-spy supports it
        if [ "$(id -u)" -eq 0 ]; then
            py-spy dump --pid "$pid" 2>&1
        else
            sudo -n py-spy dump --pid "$pid" 2>&1 \
                || py-spy dump --pid "$pid" 2>&1
        fi
        echo ""
    done

    echo "============================================================"
    echo "Hints for reading this dump:"
    echo "  - Stuck in dist.all_reduce / ncclAllReduce → NCCL collective hang"
    echo "  - Stuck in bnb.optim.optimizer8bit._step → bnb 8-bit + DDP"
    echo "  - Stuck in torch.utils.checkpoint.backward → gradient ckpt"
    echo "  - Stuck in model.forward → first forward (CUDA JIT compile?)"
    echo "  - Stuck in time.sleep on workers → still polling, rank-0 work pending"
    echo "============================================================"
} | tee "$OUT"

echo ""
echo "[diag] saved to $OUT"
echo "[diag] share this file when reporting the hang."
