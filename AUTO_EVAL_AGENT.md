# Auto-Eval Agent Guide (for Hermes)

이 문서는 **체크포인트가 생기면 자동으로 eval을 돌리는 원격 AI 에이전트**가 따라야 할 지침이다.
사람용 설명서가 아니라 LLM이 그대로 실행할 수 있도록 경로/명령을 명시적으로 적어둔다.

---

## 0. Mission

**학습은 사용자가 직접 돌린다. 에이전트는 어떤 경우에도 학습을 실행하지 않는다.**
에이전트의 유일한 자동 실행 권한은 **eval**이다.

에이전트가 해야 할 일:

1. 사용자가 학습으로 떨군 새 체크포인트를 감지하고,
2. 해당 체크포인트에 대해 멀티 벤치마크 evaluation을 돌리고,
3. 결과 디렉터리(`${EVAL_RESULTS_ROOT}`) 및 score board(`experiments.md`)에 기록하고,
4. **체크포인트가 아예 없는 셀(= 학습이 아직 안 된 셀)을 별도로 표기해서 보고**한다 — "이 셀은 사용자가 학습을 돌려야 한다"는 신호. 직접 학습 명령(`run_main_7b.sh`, `tads.train`, `torchrun` 등) 실행은 **금지**.

이미 평가된 체크포인트는 건너뛴다.

### 0-1. 최종 결과 매트릭스 (목표 = 80개 셀)

| 축 | 개수 | 값 |
|---|---|---|
| 모델 | **4** | `llama2`, `qwen25`, `mistral`, `deepseek` |
| 메서드(실험) | **4** | `full_100`, `random_10`, `data_agent_10`, `tads_10` |
| 벤치마크 | **5** | `mmlu`, `gsm8k`, `humaneval`, `tydiqa`, `bbh` |

→ 4 × 4 × 5 = **80개** (모델, 메서드, 벤치마크) 결과 셀.

에이전트 단위는 (모델, 메서드) = **16개 셀**. 각 셀에서 한 번의 eval 호출이 벤치마크 5개를 한꺼번에 처리한다 (`--benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh`).

진행 상황은 "16개 중 N개 완료 / 80개 중 M개 점수 산출"로 보고할 것.

### 0-2. Baselines — 비교 기준 (외워둘 것)

| 약칭 | 무엇 | 역할 |
|---|---|---|
| **BASE-FULL** | `full_100` — Alpaca-GPT4 **전체** 데이터로 학습 (= 01 alpaca-gpt4 full data) | **upper-bound baseline**. 10%-셀렉션 메서드(treatment)는 이 점수에 **얼마나 근접하는지**가 핵심. tads_10 / data_agent_10 / random_10 모두 이걸 향해 간다. |
| **BASE-NAIT** | `random_10` — NAIT 논문의 10% 랜덤 샘플링 baseline | **lower-bound / naive baseline**. tads_10 / data_agent_10가 **반드시 이겨야 하는** 기준선. |

즉, 매 모델마다 비교 라인은 다음 두 줄:

```
BASE-FULL (full_100)     ← treatment가 도달하려는 천장
BASE-NAIT (random_10)    ← treatment가 넘어야 하는 바닥
treatment (data_agent_10, tads_10)  ← 위 둘 사이 어디쯤이어야 정상
```

비정상 신호 (에이전트가 보고 시 우측 컬럼에 플래그할 것):

- `tads_10 < random_10` (BASE-NAIT보다 낮음) → **빨간불**. 셀렉션이 오히려 해롭다는 뜻.
- `tads_10 << full_100` (BASE-FULL보다 5%p 이상 처짐) → **노란불**. 10% 데이터로 100% 격차 회수 실패.
- `data_agent_10 > tads_10 + 1%p` → **파란불**. 경쟁 method가 더 잘함 — TADS 하이퍼파라미터 재검토 필요.

### 0-3. 전체 매트릭스 — 로컬 파일 기준 표

아래 표는 **이 repo (`/home/jieun/kms/tads`)에 실제로 존재하는 config**와, 그것이 만들어낼 체크포인트/결과 경로, 그리고 각 셀의 **비교 대상**을 정리한 것이다. 각 모델 블록 안에서 **baseline 두 줄이 위, treatment 두 줄이 아래** 순서.

`experiment_label`은 `tads/eval.py` 규칙(`<parent_dir>_<config_stem>`)으로 자동 생성되며, 결과 JSON 파일명에 그대로 박힌다.

#### (a) 학습 config × 체크포인트 위치 × 결과 디렉터리 (16개 셀)

| # | 모델 | 메서드 | Role | Config (학습/eval 공용) | 체크포인트 루트 | 결과 디렉터리 | `experiment_label` | 비교 대상 / 발산 알람 |
|---|---|---|---|---|---|---|---|---|
| 1 | llama2 | full_100 | **BASE-FULL** | `configs/experiments/main_7b/llama2/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/full_100/` | `llama2_full_100` | 기준선 (천장). 절대값 자체가 sanity 체크 — 평균 정확도가 동급 reference 대비 5%p 이상 낮으면 학습 자체 의심 |
| 2 | llama2 | random_10 | **BASE-NAIT** | `configs/experiments/main_7b/llama2/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/random_10/` | `llama2_random_10` | 기준선 (바닥). 보통 BASE-FULL 대비 3~8%p 처짐이 정상 |
| 3 | llama2 | data_agent_10 | treat | `configs/experiments/main_7b/llama2/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/data_agent_10/` | `llama2_data_agent_10` | vs `llama2_random_10` (반드시 ≥), vs `llama2_full_100` (≤이지만 -2%p 이내 권장) |
| 4 | llama2 | tads_10 | **treat (proposed)** | `configs/experiments/main_7b/llama2/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/tads_10/` | `llama2_tads_10` | vs `llama2_random_10` (반드시 ≥, -1%p 처지면 **빨간불**), vs `llama2_data_agent_10` (≥ 권장), vs `llama2_full_100` (-5%p 이상 처지면 **노란불**) |
| 5 | qwen25 | full_100 | **BASE-FULL** | `configs/experiments/main_7b/qwen25/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/full_100/` | `qwen25_full_100` | 기준선 (천장) |
| 6 | qwen25 | random_10 | **BASE-NAIT** | `configs/experiments/main_7b/qwen25/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/random_10/` | `qwen25_random_10` | 기준선 (바닥) |
| 7 | qwen25 | data_agent_10 | treat | `configs/experiments/main_7b/qwen25/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/data_agent_10/` | `qwen25_data_agent_10` | vs `qwen25_random_10` / `qwen25_full_100` (위 #3 규칙) |
| 8 | qwen25 | tads_10 | **treat (proposed)** | `configs/experiments/main_7b/qwen25/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/tads_10/` | `qwen25_tads_10` | vs `qwen25_random_10` / `qwen25_data_agent_10` / `qwen25_full_100` (위 #4 규칙) |
| 9 | mistral | full_100 | **BASE-FULL** | `configs/experiments/main_7b/mistral/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/full_100/` | `mistral_full_100` | 기준선 (천장) |
| 10 | mistral | random_10 | **BASE-NAIT** | `configs/experiments/main_7b/mistral/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/random_10/` | `mistral_random_10` | 기준선 (바닥) |
| 11 | mistral | data_agent_10 | treat | `configs/experiments/main_7b/mistral/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/data_agent_10/` | `mistral_data_agent_10` | vs `mistral_random_10` / `mistral_full_100` (위 #3 규칙) |
| 12 | mistral | tads_10 | **treat (proposed)** | `configs/experiments/main_7b/mistral/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/tads_10/` | `mistral_tads_10` | vs `mistral_random_10` / `mistral_data_agent_10` / `mistral_full_100` (위 #4 규칙) |
| 13 | deepseek | full_100 | **BASE-FULL** | `configs/experiments/main_7b/deepseek/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/full_100/` | `deepseek_full_100` | 기준선 (천장) |
| 14 | deepseek | random_10 | **BASE-NAIT** | `configs/experiments/main_7b/deepseek/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/random_10/` | `deepseek_random_10` | 기준선 (바닥) |
| 15 | deepseek | data_agent_10 | treat | `configs/experiments/main_7b/deepseek/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/data_agent_10/` | `deepseek_data_agent_10` | vs `deepseek_random_10` / `deepseek_full_100` (위 #3 규칙) |
| 16 | deepseek | tads_10 | **treat (proposed)** | `configs/experiments/main_7b/deepseek/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/tads_10/` | `deepseek_tads_10` | vs `deepseek_random_10` / `deepseek_data_agent_10` / `deepseek_full_100` (위 #4 규칙) |

> 모든 config 파일은 로컬에 실제로 존재함을 확인함. `random_50 / data_agent_50 / tads_50`도 디스크엔 있지만 **이번 매트릭스 범위 밖**이므로 자동 eval에서 제외.

#### (b) 셀 하나당 떨어지는 결과 JSON (16개 셀 × 5 벤치 = 80개 점수 파일 + 16개 summary)

`tads/eval.py`는 셀 `${EVAL_RESULTS_ROOT}/<model>/<method>/` 안에 다음 파일들을 만든다:

```
<experiment_label>-mmlu.json
<experiment_label>-gsm8k.json
<experiment_label>-humaneval.json
<experiment_label>-tydiqa.json
<experiment_label>-bbh.json
<experiment_label>-eval_summary.json    # 5개 벤치 합산 요약
logs/                                    # 벤치별 로그
```

예시 (llama2 + tads_10 셀):

```
${EVAL_RESULTS_ROOT}/llama2/tads_10/llama2_tads_10-mmlu.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/llama2_tads_10-gsm8k.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/llama2_tads_10-humaneval.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/llama2_tads_10-tydiqa.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/llama2_tads_10-bbh.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/llama2_tads_10-eval_summary.json
```

전체 16개 셀에 대해 같은 파일 6종이 만들어지면 **점수 파일 80개 + summary 16개 = 96개** 산출물이 최종 상태다.

#### (c) "셀이 완료됐다"의 정의

한 셀 `(model, method)`가 완료됐다고 보려면 해당 디렉터리 안에 **`<experiment_label>-eval_summary.json`이 존재**하고, 그 mtime이 가장 큰 `epoch_*`의 mtime보다 나중이어야 한다. 이게 §5-3의 판정 규칙이다.

### 0-4. Score Board — 단일 진실 공급원은 `experiments.md`

**결과는 단 한 곳에서만 관리한다:**

```
${EVAL_RESULTS_ROOT}/experiments.md
= /group-volume/minsoo3.kim/tads-eval-results/experiments.md
```

이 파일이 **유일한 score board**다. 에이전트는 매 tick마다:

1. 셀의 `<experiment_label>-eval_summary.json`을 읽어 정확도(%) 둘째 자리까지 추출
2. `experiments.md`의 해당 셀을 `-`에서 실제 숫자로 교체 (또는 새 숫자로 갱신)
3. treatment 행이면 §0-2 규칙으로 발산 알람(`RED` / `YELLOW` / `BLUE`)을 inline으로 표기
4. 파일을 atomic하게 저장 (`experiments.md.tmp` 작성 후 `mv`)
5. 채팅/로그 보고에는 **요약만**. 표 전체를 매번 복붙하지 말 것 — 변경된 행만 인용

아래 표들은 **이 파일의 초기 템플릿**이다. `experiments.md`가 존재하지 않으면 에이전트가 이 템플릿을 그대로 복사해서 생성하고, 이후 셀 단위로 갱신만 한다. **이 가이드 문서(AUTO_EVAL_AGENT.md) 자체는 절대 수정하지 말 것** — 가이드는 spec이고, `experiments.md`가 live document.

값은 `<experiment_label>-eval_summary.json` 또는 벤치별 JSON에서 직접 파싱한 정확도 (%, 소수점 둘째자리까지). **아직 안 돌아간 셀은 `-` 그대로**. 표 형식은 monospace 가정. `experiments.md`에 쓸 때 코드 펜스(```) 안에 넣어 정렬 깨지지 않게.

#### (1) llama2

```
Method          Role          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ============= ======= ======= ========== ======== ======= =======
full_100        BASE-FULL     -       -       -          -        -       -
random_10       BASE-NAIT     -       -       -          -        -       -
data_agent_10   treat         -       -       -          -        -       -
tads_10         proposed *    -       -       -          -        -       -
```

#### (2) qwen25

```
Method          Role          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ============= ======= ======= ========== ======== ======= =======
full_100        BASE-FULL     -       -       -          -        -       -
random_10       BASE-NAIT     -       -       -          -        -       -
data_agent_10   treat         -       -       -          -        -       -
tads_10         proposed *    -       -       -          -        -       -
```

#### (3) mistral

```
Method          Role          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ============= ======= ======= ========== ======== ======= =======
full_100        BASE-FULL     -       -       -          -        -       -
random_10       BASE-NAIT     -       -       -          -        -       -
data_agent_10   treat         -       -       -          -        -       -
tads_10         proposed *    -       -       -          -        -       -
```

#### (4) deepseek

```
Method          Role          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ============= ======= ======= ========== ======== ======= =======
full_100        BASE-FULL     -       -       -          -        -       -
random_10       BASE-NAIT     -       -       -          -        -       -
data_agent_10   treat         -       -       -          -        -       -
tads_10         proposed *    -       -       -          -        -       -
```

#### (5) Cross-model summary (avg of 5 benchmarks)

```
                          llama2     qwen25     mistral    deepseek
========================= ========== ========== ========== ==========
full_100      BASE-FULL   -          -          -          -
random_10     BASE-NAIT   -          -          -          -
data_agent_10 treat       -          -          -          -
tads_10       proposed *  -          -          -          -
```

채우는 규칙 (§5-4의 4-state 분류를 그대로 반영):

| 상태 | 셀 표기 | 비고 |
|---|---|---|
| **NEED-TRAIN** | `학습필요` | 한 행 전체를 이 마커로 채움 (벤치 컬럼 5개 + avg). 사용자에게 별도 "학습이 필요한 셀 목록" 보고. |
| **NEED-EVAL** | `eval대기` | 체크포인트는 있음. 곧 자동 eval됨. |
| **LEGACY** | `legacy(숫자)` 예: `legacy(41.20)` | 옛 포맷에서 추출. 다음 tick에 새 포맷으로 덮어써 DONE 전환. |
| **DONE** | 정확도(%) 둘째 자리까지. 예: `42.13`. | humaneval은 pass@1. |
| 진행 중 (eval 실행 중) | `…` 또는 `eval중` | 보고 본문에 "running" 라인 별도. |
| 최근 실패 | `실패` | 보고 본문에 "failed: <원인>" 라인. fail_count(§10-3) 동시 갱신. |

추가 규칙:
- treatment 행이 DONE이면 옆에 §0-2의 발산 알람을 인라인으로 붙일 것. 예: `tads_10  proposed * 41.20 (RED < random_10)`
- NEED-TRAIN 셀은 score board 갱신 외에 **별도 섹션 "필요한 학습 목록"을 `experiments.md` 상단에 자동 추가**해서 사용자가 한눈에 보게 할 것. 형식:
  ```
  ## 사용자 액션 필요 — 아직 학습되지 않은 셀 (NEED-TRAIN)
  - llama2 / data_agent_10
  - llama2 / tads_10
  - mistral / random_10
  ...
  ```
- 셀의 LEGACY 점수가 발산 알람을 트리거하더라도, 재실행으로 DONE이 될 때까지는 알람을 빨간/노란색이 아니라 회색(`(provisional)`)으로 표기.

---

## 1. Repo / Working Directory

- Repo root: `/home/jieun/kms/tads` (서버에서도 동일 경로라고 가정. 다르면 `cd $(git rev-parse --show-toplevel)`로 이동.)
- 모든 명령은 **repo root에서 실행**해야 한다. 새 권장 형식(`python -m tads.eval`)은 cwd가 repo root여야 모듈 경로가 풀린다.

---

## 2. Environment Setup (반드시 가장 먼저)

쉘마다 한 번씩:

```bash
source scripts/setup_env.sh
```

이 스크립트가 export하는 핵심 환경 변수:

| 변수 | 기본값 | 의미 |
|---|---|---|
| `OUTPUT_ROOT` | `/group-volume/minsoo3.kim/tads-checkpoints` | 학습 체크포인트 루트 |
| `EVAL_RESULTS_ROOT` | `/group-volume/minsoo3.kim/tads-eval-results` | eval 결과 루트 |
| `HF_HOME`, `HF_*_OFFLINE` | offline=1 | 오프라인 모드 강제. 절대 끄지 말 것 |
| `MODEL_PATH_*`, `*_DATA_DIR` | `/group-volume/...` | 모델/벤치마크 경로 |

`setup_env.sh`는 누락된 경로를 **경고만** 하고 종료하지 않는다. 경고가 있어도 사용하지 않는 모델/벤치마크면 무시 가능.

---

## 3. Checkpoint Layout (어디서 새 체크포인트를 찾을지)

### 3-1. 현재 main matrix — **history-preserving run-layout** (메인 타깃)

`feature/run-layout` (커밋 `358cf11`) 이후로 학습은 매 호출마다 별도의 `runs/<tag>/`
서브디렉터리에 떨어진다. 이전 체크포인트는 절대 덮어쓰지 않으며, 가장 최근 run은
`_latest` 포인터가 가리킨다.

```
${OUTPUT_ROOT}/main_7b/<model>/<method>/
├── runs/
│   ├── 20260515_180000/                ← 첫 학습 (auto-timestamp tag)
│   │   ├── cfg.yaml + cfg.json         ← 사용된 모든 하이퍼파라미터 스냅샷
│   │   ├── epoch_1/, epoch_2/, epoch_3/
│   │   │   ├── _complete               ← 저장 성공 시에만 생기는 sentinel
│   │   │   ├── env_meta.json           ← torch / bitsandbytes 버전
│   │   │   ├── optimizer.pt, scheduler.pt
│   │   │   └── (model.safetensors 등 HF 표준)
│   │   ├── metrics.json
│   │   └── selected_indices_epoch{N}.json
│   ├── 20260515_193000_lr5e5/          ← 같은 셀에 파라미터 튜닝 재학습
│   └── 20260516_080000_anchor_all/
└── _latest -> runs/20260516_080000_anchor_all/   ← 항상 최신 가리킴 (symlink)
```

- `<model>` ∈ {`llama2`, `qwen25`, `mistral`, `deepseek`}
- `<method>` ∈ {`full_100`, `random_10`, `data_agent_10`, `tads_10`} (또는 `_50` 변형)
- 매 sealed epoch 저장 후 `_latest` 포인터가 atomic하게 갱신됨 (학습 중 eval 가능).
- symlink 불가 FS는 `_latest.txt` (run_tag 한 줄)로 fallback.
- **eval은 항상 `_latest` 기준** — 사용자가 새 튜닝 잡을 띄우면 자동으로 새 결과가 평가됨.

대응 config: `configs/experiments/main_7b/<model>/<method>.yaml`

**최신 epoch 찾는 정식 방법**:
```bash
ckpt_root="${OUTPUT_ROOT}/main_7b/<model>/<method>"
latest_run="$(readlink -f "$ckpt_root/_latest" 2>/dev/null || \
              { [ -f "$ckpt_root/_latest.txt" ] && echo "$ckpt_root/runs/$(cat "$ckpt_root/_latest.txt")"; })"
latest_epoch="$(ls -1d "$latest_run"/epoch_* 2>/dev/null | sort -V | tail -n 1)"
# latest_epoch에 _complete 파일이 있어야 정상적으로 sealed 됐다는 뜻.
[ -f "$latest_epoch/_complete" ] || { echo "WARN: $latest_epoch is not sealed"; }
```

### 3-2. Legacy flat layout (`feature/run-layout` 이전 학습)

```
${OUTPUT_ROOT}/main_7b/<model>/<method>/epoch_<N>/   ← runs/ 와 _latest 없음
```

`feature/run-layout` 머지 이전에 학습된 셀은 평탄 구조로 떨어져 있다. eval 측은
`_latest`가 없으면 평탄 구조를 fallback으로 자동 인식 (§4-4 참고). 이전 체크포인트를
새 구조로 마이그레이션하려면 `runs/<tag>/`로 이동 후 `_latest`를 수동 설정:

```bash
mv ${OUTPUT_ROOT}/main_7b/llama2/tads_10/epoch_* ${OUTPUT_ROOT}/main_7b/llama2/tads_10/runs/legacy/
ln -s runs/legacy ${OUTPUT_ROOT}/main_7b/llama2/tads_10/_latest
```

### 3-3. Legacy 7b_fullft (더 옛날 구버전, 보통은 건드릴 일 없음)

```
${OUTPUT_ROOT}/7b_fullft/<run>/epoch_3/
```

대응 config: `configs/experiments/7b_fullft_<run>.yaml`. 이건 별도 스크립트(`auto_eval_7b_fullft.sh`)에 위임한다 — §6 참고.

---

## 4. Eval 실행 — **per-GPU per-experiment** (의무 형식)

**원칙**: 셀 1개(`<model>/<method>`) = GPU 1장 = 백그라운드 프로세스 1개.
에이전트는 매 tick에서 `nvidia-smi`로 **사용 중이지 않은 GPU**를 발견하면 NEED-EVAL
큐에서 한 셀을 골라 그 GPU에 곧바로 launch한다. 다음 tick에서 다시 빈 GPU를
찾고 큐의 다음 셀을 또 launch — 큐가 빌 때까지 반복.

### 4-1. 한 셀 launch — 정식 명령 형태

bash 스크립트 wrapper(`run_eval_main_7b.sh`)는 더 이상 사용하지 않는다.
**`python -m tads.eval`을 직접 호출**하고 `CUDA_VISIBLE_DEVICES`로 GPU를 핀한다.

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> nohup python -m tads.eval \
    --config configs/experiments/main_7b/<model>/<method>.yaml \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
    >> logs/eval_<model>_<method>.log 2>&1 &
```

`--ckpt`는 **생략**한다 — `tads.eval`이 `<output_dir>/_latest`에서 마지막 sealed
epoch을 자동 resolve하므로 (§3-1 참고). `--out_dir`도 생략하면 자동으로
`<ckpt>/eval/` 옆에 결과가 떨어진다.

> **`CUDA_VISIBLE_DEVICES`는 1개 GPU만 지정**할 것. 2개 이상 주면 그 잡이 두 GPU를
> 모두 점유해 다음 셀이 launch될 자리가 사라진다. `--cuda_device 0`은 잡 내부
> 인덱스이므로 항상 0 (CUDA_VISIBLE_DEVICES로 이미 1개로 좁혀진 상태).

### 4-2. 빈 GPU 찾기 — 결정 규칙

```bash
# 사용 가능한 GPU 목록 (memory.used < 2 GB AND compute-process 없음 = 비어있음)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', *' '$2 < 2000 {print $1}'
```

추가 안전장치:
1. **학습 중인 잡의 GPU는 절대 건드리지 말 것**:
   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader \
     | grep -E 'python.*tads\.(train|eval)' || true
   ```
2. **우리 평가 잡 자체가 이미 그 GPU에 떠있는지** 체크:
   ```bash
   pgrep -af "python -m tads.eval.*<model>/<method>" >/dev/null && echo "already running"
   ```
3. 같은 셀이 이미 큐 안에 있으면 새로 enqueue 금지 (중복 lauch 방지).

### 4-3. 한 tick의 알고리즘 (실행 순서)

```
1. NEED-EVAL 큐 = §5-4의 classify_cell()로 결정 (DONE/LEGACY/NEED-TRAIN 제외)
2. 빈 GPU 리스트 = nvidia-smi 기반 §4-2
3. for gpu in 빈 GPU:
       if NEED-EVAL 큐 비었으면 break
       cell = NEED-EVAL 큐.pop_front()
       if 그 cell이 이미 어떤 GPU에서 돌고 있으면 skip (다른 cell pop)
       launch §4-1 명령으로 (CUDA_VISIBLE_DEVICES=gpu)
       0.5 sec sleep (CUDA init race 회피)
4. 큐가 남아있어도 빈 GPU 없으면 다음 tick까지 대기
5. 매 tick 시작에서 끝난 잡(=프로세스 종료) 확인 → DONE/FAIL 분류 → 보고
```

이 알고리즘은 §9 cron tick 스크립트 안에 그대로 들어있다 (§9-3 참고).

### 4-4. 필터 / 제한

```bash
# 한 셀만 강제 실행 (테스트용)
CUDA_VISIBLE_DEVICES=0 python -m tads.eval \
    --config configs/experiments/main_7b/llama2/tads_10.yaml \
    --benchmarks mmlu \
    --limit 16    # 디버그용 sample cap

# 과거 특정 run을 다시 평가
CUDA_VISIBLE_DEVICES=0 python -m tads.eval \
    --config configs/experiments/main_7b/llama2/tads_10.yaml \
    --run_tag 20260515_180000 \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh
```

- `--benchmarks` 기본 `mmlu`. 정식 매트릭스에선 `mmlu,gsm8k,humaneval,tydiqa,bbh` 명시.
- `--limit N`: 벤치별 샘플 N개로 제한 (디버그 전용).
- `--run_tag <tag>`: `_latest` 대신 특정 과거 run 평가.

---

## 5. 결과 / "이미 평가됨" 판정 규칙

### 5-1. 결과 저장 위치

```
${EVAL_RESULTS_ROOT}/<model>/<method>/
```

(주의: `main_7b/` 접두어 **없음**. `OUTPUT_ROOT`와 레이아웃이 다름.)

### 5-2. 로그

```
logs/eval_main_7b_<model>_<method>.log
```

### 5-3. 중복 평가 방지 로직 (에이전트가 직접 판정해야 함)

`python -m tads.eval`은 자체적인 "이미 함" 체크가 **없다** (기존 bash wrapper도 마찬가지였음). 호출 전에 에이전트가 판단해야 한다.

새 run-layout (§3-1) 기준 — 셀 한 개에 대해 **다음이 모두 만족**되면 eval 재실행 불필요:

1. `${OUTPUT_ROOT}/main_7b/<model>/<method>/_latest`가 가리키는 run의 가장 큰
   sealed `epoch_N/`(= `_complete` 파일이 있는 것)을 찾는다.
2. `${EVAL_RESULTS_ROOT}/<model>/<method>/<experiment_label>-eval_summary.json`가 있고,
   그 mtime이 위 sealed epoch 의 mtime보다 **나중**.

`_latest` 포인터가 갱신됐다는 건 사용자가 새 학습을 마쳤다는 뜻이므로,
포인터 변경 후 mtime이 새 → 자동으로 재평가 대상이 된다. **`runs/<tag>/`의 다른 과거
run 들은 평가 대상이 아니다** (그것들은 사용자가 명시적으로 `--run_tag`로 평가).

(Legacy 스크립트 `auto_eval_7b_fullft.sh`는 `.eval_done` 센티넬을 쓰지만, main 스크립트는 그걸 안 만든다. 일관성 유지하려면 에이전트가 평가 성공 후 `${EVAL_RESULTS_ROOT}/<model>/<method>/.eval_done`를 직접 `touch`해줘도 된다. 단, eval **실패 시 절대 touch 금지** — 옛날 버그가 이거였다.)

### 5-4. 셀 상태 분류 (4-state) — 매 tick 시작 시 모든 셀에 대해 판정

매트릭스 16개 셀 각각을 다음 **4가지 중 하나**로 분류한다. score board의 마커, 에이전트의 행동, 사용자 보고가 모두 이 분류에 따라 갈린다.

| 상태 | 판정 조건 | 에이전트 행동 | Score board 마커 |
|---|---|---|---|
| **NEED-TRAIN** | `${OUTPUT_ROOT}/main_7b/<model>/<method>/`이 없거나, `_latest` 포인터(또는 `_latest.txt`)가 없고 평탄 layout의 `epoch_*`도 없음 | **아무것도 자동 실행하지 말 것**. 사용자에게 "이 셀은 학습이 아직 안 됐다" 보고만. | `학습필요` |
| **NEED-EVAL** | sealed epoch이 존재하지만, 결과 디렉터리에 §5-5의 최신 포맷 파일(`<experiment_label>-eval_summary.json`)이 없거나, 있어도 mtime ≤ latest sealed epoch mtime | 다음 tick에 eval 실행 (단일 셀에 `MODELS=<m> METHODS=<x>` 필터로 호출) | `eval대기` |
| **LEGACY** | 결과 디렉터리에 옛 포맷 파일만 있음 (예: 접두어 없는 `eval_summary.json`, 또는 벤치별 `mmlu.json` / `gsm8k.json` 만 있고 `*-eval_summary.json` 없음) | 점수는 옛 파일에서 추출해 표에 잠정 기재하되, **재실행 권장 (NEED-EVAL과 동일하게 큐잉)**. 새 포맷으로 덮어쓰면 LEGACY → DONE 자동 전환. | `legacy(점수)` 예: `legacy(41.20)` |
| **DONE** | 결과 디렉터리에 `<experiment_label>-eval_summary.json`이 있고, mtime > latest sealed epoch mtime | 건너뜀. 점수 표에 숫자 반영. | 실제 숫자 (예: `42.13`) |

판정 의사 코드 (run-layout + flat-layout fallback):

```python
def _resolve_latest_run(ckpt_root):
    """Return path of <ckpt_root>/_latest target, or None.
    Also handles _latest.txt fallback for symlink-less filesystems."""
    link = f"{ckpt_root}/_latest"
    if os.path.islink(link):
        target = os.path.realpath(link)
        return target if os.path.isdir(target) else None
    if os.path.isdir(link) and glob(f"{link}/epoch_*"):
        return link
    txtfile = f"{ckpt_root}/_latest.txt"
    if os.path.isfile(txtfile):
        tag = open(txtfile).read().strip()
        run = f"{ckpt_root}/runs/{tag}"
        return run if os.path.isdir(run) else None
    return None

def _largest_sealed_epoch(run_dir):
    """epoch_N/_complete 가 있는 것만 카운트. 없으면 None."""
    sealed = []
    for p in glob(f"{run_dir}/epoch_*"):
        try: n = int(os.path.basename(p).replace("epoch_", ""))
        except ValueError: continue
        if not os.path.exists(f"{p}/_complete"): continue
        sealed.append((n, p))
    return max(sealed)[1] if sealed else None

def classify_cell(model, method):
    ckpt_root = f"{OUTPUT_ROOT}/main_7b/{model}/{method}"
    # 1) 새 run-layout 우선
    latest_run = _resolve_latest_run(ckpt_root)
    if latest_run:
        latest_epoch = _largest_sealed_epoch(latest_run)
    else:
        # 2) Legacy flat layout fallback
        flat = sorted(glob(f"{ckpt_root}/epoch_*"))
        latest_epoch = flat[-1] if flat else None
    if latest_epoch is None:
        return "NEED-TRAIN", None
    out_dir = f"{EVAL_RESULTS_ROOT}/{model}/{method}"
    label = f"{model}_{method}"     # experiment_label 규칙 (parent_stem)
    new_summary = f"{out_dir}/{label}-eval_summary.json"
    if exists(new_summary) and mtime(new_summary) > mtime(latest_epoch):
        return "DONE", new_summary
    # 최신 포맷이 없거나 stale — 옛 포맷이라도 있는지 확인
    legacy_candidates = (
        glob(f"{out_dir}/eval_summary.json")              # 옛 접두어 없는 이름
        + glob(f"{out_dir}/{label}-*.json")               # 새 포맷 벤치별
        + glob(f"{out_dir}/mmlu.json")                     # 옛 벤치별
        + glob(f"{out_dir}/gsm8k.json")
        + glob(f"{out_dir}/humaneval.json")
        + glob(f"{out_dir}/tydiqa.json")
        + glob(f"{out_dir}/bbh.json")
    )
    if legacy_candidates:
        return "LEGACY", legacy_candidates
    return "NEED-EVAL", None
```

매 tick 시작 시 16개 셀의 분류 결과를 한 줄 요약 로그로 남길 것:

```
[tick 2026-05-15T12:00] DONE=3 NEED-EVAL=2 LEGACY=4 NEED-TRAIN=7
```

### 5-5. 결과 JSON 포맷 — 옛것/새것 혼재 대응

**옛날 코드와 최신 코드가 같은 트리에 섞여 있어 JSON 스키마가 일관적이지 않다.** 에이전트는 키 이름을 가정하지 말고 **반드시 파일을 열어 구조를 직접 확인**한 후 점수를 추출한다.

#### (1) 식별 우선순위 (filename 기준)

| 우선 | 파일 패턴 | 비고 |
|---|---|---|
| 1 (최신) | `<experiment_label>-eval_summary.json` <br/> 예: `llama2_tads_10-eval_summary.json` | 현재 `tads/eval.py`가 만드는 표준. 모든 벤치 점수와 메타데이터를 포함. 이 파일이 있으면 무조건 이걸 먼저 본다. |
| 2 (최신, 벤치별) | `<experiment_label>-<bench>.json` <br/> 예: `llama2_tads_10-mmlu.json` | 벤치별 상세. summary가 없으면 이것들로 합산 (단, 5개 벤치 다 있을 때만 DONE 처리). |
| 3 (legacy) | `eval_summary.json` (접두어 없음) | 옛 코드 산출물. label 충돌 위험(다른 셀의 결과를 덮어썼을 수 있음). LEGACY로 분류. |
| 4 (legacy) | `mmlu.json` / `gsm8k.json` / `humaneval.json` / `tydiqa.json` / `bbh.json` (접두어 없음) | 옛 벤치별. LEGACY로 분류. |

#### (2) 파싱 절차 (스키마 가정 금지)

```python
import json
with open(path) as f:
    data = json.load(f)

# 1. 최상위 키를 먼저 출력해서 구조 파악
print(list(data.keys()))

# 2. 흔히 보이는 키 후보들 (어느 게 있을지 모름):
#    - "metrics" (dict of bench -> {accuracy/exact_match/pass@1/...})
#    - "results"
#    - 벤치 이름이 최상위 키 (예: data["mmlu"]["acc"])
#    - "summary" / "score" / "accuracy"
#
# 3. 점수 키 후보:
#    - "accuracy", "acc", "acc_norm"
#    - "exact_match", "em"
#    - "pass@1", "pass_at_1"
#    - "f1"
#
# 4. 값이 0~1 범위면 ×100 해서 %로, 이미 1~100이면 그대로.
```

**절대 금지**: 키 이름을 하드코드해서 KeyError 나면 셀을 통째로 skip하는 패턴. 키가 다르면 일단 파일 전체 dict를 dump해서 로그에 남기고, 가능한 후보를 시도한 후 그래도 못 찾으면 해당 벤치만 `-`로 두고 나머지는 진행.

#### (3) 새 포맷으로 통일 정책

LEGACY로 분류된 셀은 가능하면 **다음 tick에 새 eval을 돌려 최신 포맷으로 덮어쓴다**. 단, 덮어쓰기 전에 옛 파일을 같은 디렉터리 안 `legacy/` 하위로 옮겨 백업할 것:

```bash
mkdir -p "${out_dir}/legacy"
mv "${out_dir}/eval_summary.json"   "${out_dir}/legacy/" 2>/dev/null || true
mv "${out_dir}"/{mmlu,gsm8k,humaneval,tydiqa,bbh}.json "${out_dir}/legacy/" 2>/dev/null || true
```

(`<experiment_label>-*.json`는 새 포맷이므로 옮기지 말 것 — 그건 새 eval이 자체적으로 덮어쓴다.)

---

## 6. Legacy 7b_fullft 자동 감시 모드 (참고만)

쓸 일이 있다면:

```bash
bash scripts/auto_eval_7b_fullft.sh <gpu_id> [run1 run2 ...]
# 예: bash scripts/auto_eval_7b_fullft.sh 0 tads_50 random_50
```

이건 무한 루프(60초 sleep)로 `epoch_3`를 polling하므로 **tmux/screen 안에서 띄워야 한다**. 메인 매트릭스에는 이 스크립트를 쓰지 말 것.

---

## 7. 에이전트가 한 번의 tick에서 해야 할 일 (의사 코드)

**원칙 재강조**: 셀 1개 = GPU 1장 = `python -m tads.eval` 백그라운드 프로세스 1개.
bash wrapper 호출 금지. 빈 GPU가 있는 만큼만 한꺼번에 launch하고, 큐에 남은 셀은
다음 tick에서 또 빈 GPU가 생기면 launch.

```
1. cd /home/jieun/kms/tads
2. source scripts/setup_env.sh   # 매 쉘마다 한 번
3. classify pass — NEED-EVAL 큐 만들기
   for model in {llama2, qwen25, mistral, deepseek}:
     for method in {full_100, random_10, data_agent_10, tads_10}:
         ckpt_root = ${OUTPUT_ROOT}/main_7b/${model}/${method}
         latest_run = resolve_latest_run(ckpt_root)        # §4-2의 함수
         if not latest_run: continue                       # NEED-TRAIN
         latest = largest_sealed_epoch(latest_run)         # _complete 있는 max N
         if not latest: continue
         out_dir = ${EVAL_RESULTS_ROOT}/${model}/${method}
         done_marker = ${out_dir}/.eval_done
         if done_marker exists AND mtime(done_marker) > mtime(latest): continue
         # 같은 (model, method)가 이미 어디 GPU에서 돌고 있으면 skip
         if pgrep -af "python -m tads.eval.*${model}/${method}\.yaml" > /dev/null: continue
         queue.append((model, method))

4. dispatch pass — 빈 GPU만큼 launch
   free_gpus = nvidia-smi 기반, memory.used < 2GB AND tads.train 프로세스 없음
   while queue and free_gpus:
     gpu  = free_gpus.pop()
     cell = queue.pop(0)
     launch:
       CUDA_VISIBLE_DEVICES=${gpu} nohup python -m tads.eval \
         --config configs/experiments/main_7b/${cell.model}/${cell.method}.yaml \
         --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
         >> logs/eval_${cell.model}_${cell.method}.log 2>&1 &
     log "launched ${cell} on GPU ${gpu} (pid=$!)"
     sleep 0.5   # CUDA init race buffer

5. monitor pass — 끝난 잡 회수
   for proc in 우리가 launch한 프로세스들 (pidfile / pgrep로 추적):
     if exited 0:  touch ${EVAL_RESULTS_ROOT}/${model}/${method}/.eval_done
     if exited !=0: log 끝 30줄 캡처 → 보고; .fail_count 증가

6. 큐가 비지 않았으면 다음 tick에서 4번 부터 재시도 (epoch당 한 번 dispatch가 정상)
```

---

## 8. 금지 사항 / 함정

- **HF offline 모드를 끄지 말 것**. `setup_env.sh`가 `HF_DATASETS_OFFLINE=1` 등을 강제하는 이유는 클러스터 노드가 outbound HTTPS가 없어서 켜면 캐시락이 깨지기 때문.
- **체크포인트를 절대 자동 삭제하지 말 것**. legacy 스크립트의 `CLEANUP_EARLY_EPOCHS` 옵션은 **opt-in이고 기본값 0**. 에이전트는 건드리지 말 것.
- **`OUTPUT_ROOT`와 `EVAL_RESULTS_ROOT`를 혼동하지 말 것**. checkpoint는 `OUTPUT_ROOT/main_7b/...`, 결과는 `EVAL_RESULTS_ROOT/<model>/<method>` (접두어 없음).
- **eval 실패 시 done 마커를 만들지 말 것**. 옛날 버그가 이거여서 실패한 run을 영영 재시도 안 했다.
- **user-volume (`~`)에 절대 쓰지 말 것**. HF cache는 이미 `DATA_CACHE=${OUTPUT_ROOT}/cache` 아래로 redirect되어 있다. 새 파일을 만들 거면 `OUTPUT_ROOT` 또는 `EVAL_RESULTS_ROOT` 아래에만 만들 것 (user-volume 50GB는 금방 찬다).
- **core dump 켜지 말 것** (`TADS_ENABLE_COREDUMPS=0` 유지). 7B DDP rank 한 개가 죽으면 ~240GB 코어 파일을 떨군다.
- **GPU 충돌 주의**. 학습 중인 GPU에 eval을 같이 쏘면 OOM. 학습 잡(`python -m tads.train`)이 점유한 GPU는 §9-3의 `training_gpus` 필터로 자동 제외 — 그 필터를 끄지 말 것.
- **`CUDA_VISIBLE_DEVICES`에 GPU 1개만 지정**. 여러 개 지정하면 그 잡이 둘 다 점유해 다음 셀이 launch될 자리가 사라진다. 셀 1개 = GPU 1장이 §4의 의무 형식.
- **`scripts/run_eval_main_7b.sh` (bash wrapper) 호출 금지**. `--parallel` 모드의 sequential dispatch 로직이 우리 per-tick 큐 알고리즘과 충돌. `python -m tads.eval`을 직접 호출할 것.

---

## 9. Cron 스케줄 — 주기적 자동 체크

에이전트는 **무한 루프로 상주하지 않는다**. cron이 정해진 간격으로 호출하고 매번 idempotent하게 §7 의사코드를 수행한다.

### 9-1. 권장 간격

- **30분마다** (`*/30 * * * *`) — 학습 한 epoch에 통상 1시간 이상 걸리므로 30분 폴링이면 새 체크포인트를 늦지 않게 잡는다. GPU/디스크 부담도 최소.
- 학습 끝물(체크포인트가 줄줄이 떨어지는 구간)에 빠른 따라잡기가 필요하면 일시적으로 10분(`*/10`)으로 조이고, 매트릭스 16셀이 다 차면 다시 30분 또는 1시간으로 늘릴 것.

### 9-2. crontab 예시

서버에 `crontab -e`로 등록:

```cron
# TADS auto-eval — 매 30분, 새 체크포인트 감지 후 eval
*/30 * * * * /home/jieun/kms/tads/scripts/auto_eval_tick.sh >> /home/jieun/kms/tads/logs/auto_eval_cron.log 2>&1
```

### 9-3. tick 스크립트 (없으면 에이전트가 생성)

`scripts/auto_eval_tick.sh` — 다음 골격으로 작성. cron은 비대화형 쉘이라 PATH/env가 사실상 비어있으므로 **반드시 setup_env.sh를 명시적으로 source**하고, 동시 실행 방지용 flock을 건다.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=/home/jieun/kms/tads
LOCK=/tmp/tads_auto_eval.lock
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

# 이전 tick이 아직 안 끝났으면 그대로 종료 (중복 방지)
exec 9>"$LOCK"
flock -n 9 || { echo "[tick $(date -Is)] previous tick still running, skip"; exit 0; }

cd "$REPO"
# cron은 PATH/env 거의 비어있음 — 명시적 source 필수
source scripts/setup_env.sh >/dev/null

# §7 의사코드의 enqueue 로직을 그대로 수행
# (run-layout: _latest 포인터 + _complete sealed sentinel 인식)

resolve_latest_run() {
  local ckpt_root=$1
  local link="${ckpt_root}/_latest"
  if [ -L "$link" ] || [ -d "$link" ]; then
    readlink -f "$link"
    return
  fi
  if [ -f "${ckpt_root}/_latest.txt" ]; then
    local tag
    tag=$(cat "${ckpt_root}/_latest.txt")
    [ -d "${ckpt_root}/runs/${tag}" ] && echo "${ckpt_root}/runs/${tag}"
    return
  fi
  # Legacy flat layout fallback
  if compgen -G "${ckpt_root}/epoch_*" >/dev/null; then
    echo "${ckpt_root}"
  fi
}

largest_sealed_epoch() {
  local run_dir=$1
  ls -1d "${run_dir}"/epoch_* 2>/dev/null | sort -V | while read -r p; do
    [ -f "${p}/_complete" ] && echo "${p}"
  done | tail -n 1
}

need=()
for model in llama2 qwen25 mistral deepseek; do
  for method in full_100 random_10 data_agent_10 tads_10; do
    ckpt_root="${OUTPUT_ROOT}/main_7b/${model}/${method}"
    latest_run=$(resolve_latest_run "$ckpt_root")
    [ -z "$latest_run" ] && continue
    latest=$(largest_sealed_epoch "$latest_run")
    [ -z "$latest" ] && continue
    out_dir="${EVAL_RESULTS_ROOT}/${model}/${method}"
    done_marker="${out_dir}/.eval_done"
    if [ -f "$done_marker" ] && [ "$done_marker" -nt "$latest" ]; then
      continue
    fi
    need+=("${model}/${method}")
  done
done

if [ ${#need[@]} -eq 0 ]; then
  echo "[tick $(date -Is)] nothing to do"
  exit 0
fi

echo "[tick $(date -Is)] need: ${need[*]}"

# 같은 셀이 이미 어디 GPU에서 돌고 있으면 큐에서 제외 (중복 launch 방지)
filtered=()
for cell in "${need[@]}"; do
  model="${cell%/*}"; method="${cell#*/}"
  if pgrep -af "python -m tads.eval.*main_7b/${model}/${method}\.yaml" >/dev/null; then
    echo "[tick $(date -Is)] already running: ${cell} — skip enqueue"
    continue
  fi
  filtered+=("$cell")
done
need=("${filtered[@]}")

# 빈 GPU 목록 (메모리 < 2GB, 그리고 tads.train 프로세스가 점유하지 않은 것).
# 학습 잡이 도는 GPU는 절대 건드리지 말 것 — OOM의 가장 흔한 원인.
training_gpus=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name \
                --format=csv,noheader 2>/dev/null \
                | grep -E 'python.*tads\.train' | awk -F',' '{print $1}' | sort -u)
free_gpus=()
while IFS=, read -r idx mem; do
  mem="${mem# }"
  [ "${mem:-9999}" -ge 2048 ] && continue
  # 이 GPU에 tads.train이 떠있으면 제외
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$idx" | tr -d ' ')
  if echo "$training_gpus" | grep -q "$uuid"; then continue; fi
  free_gpus+=("$idx")
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

if [ ${#free_gpus[@]} -eq 0 ]; then
  echo "[tick $(date -Is)] no free GPU — ${#need[@]} cells stay queued for next tick"
  exit 0
fi
echo "[tick $(date -Is)] free GPUs: ${free_gpus[*]}  |  queue: ${need[*]}"

# Dispatch pass: 빈 GPU 1장에 셀 1개를 nohup 백그라운드로 launch.
# bash wrapper(run_eval_main_7b.sh)는 의도적으로 사용하지 않음 — 한 번에 여러 셀을
# 병렬 dispatch하려면 wrapper가 가진 sequential vs --parallel 모드와 우리 큐
# 알고리즘이 충돌. 그냥 `python -m tads.eval`을 직접 부르는 게 가장 단순.
launched_pids=()
for gpu in "${free_gpus[@]}"; do
  [ ${#need[@]} -eq 0 ] && break
  cell="${need[0]}"; need=("${need[@]:1}")
  model="${cell%/*}"; method="${cell#*/}"
  cfg="configs/experiments/main_7b/${model}/${method}.yaml"
  log="${LOG_DIR}/eval_${model}_${method}.log"
  echo "[tick $(date -Is)] dispatch ${cell} -> GPU ${gpu}" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tads.eval \
      --config "$cfg" \
      --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
      >> "$log" 2>&1 &
  launched_pids+=("$!:${cell}")
  sleep 0.5
done

# 이 tick에서 launch만 하고 종료 — 잡들은 백그라운드에서 계속 돈다.
# 다음 tick에서 (a) 끝난 잡은 .eval_done 자동 갱신 되어있을 것 (eval 본체가
# 마지막 atomic write), (b) 새 done_marker로 큐가 자동 줄어듦.
echo "[tick $(date -Is)] launched: ${launched_pids[*]}"

# OPTIONAL: 직전 tick에서 launch했던 잡의 종료 회수 — exit code 0이면 done_marker
# touch, 0 아니면 .fail_count 증가. 이걸 모니터링하려면 tick 자체가 길게 살아야
# 해서 cron 모델과 안 맞음. 권장: launch만 하고 .eval_done은 eval 자체에서 마지막
# 단계로 atomic touch (이미 tads.eval이 출력 디렉토리에 결과 JSON 쓰는 시점이
# 자연스러운 sentinel — 별도 .eval_done 없이도 §5-3의 mtime 비교로 판단 가능).
```

### 9-4. cron 디버깅 체크리스트

- cron이 안 돌면 → `systemctl status cron`, `/var/log/syslog | grep CRON`
- 돌긴 하는데 환경 변수 없음 에러 → setup_env.sh가 정말 source됐는지 (cron은 `$HOME` 빼고 거의 비어있음. 절대경로 사용 필수)
- 동시 실행으로 GPU 충돌 → flock 라인이 살아있는지 확인
- 새 체크포인트 감지 못 함 → 다음 두 줄을 tick 스크립트 안에 임시 추가해 stdout 확인:
  ```bash
  ls -ld ${OUTPUT_ROOT}/main_7b/*/*/_latest 2>/dev/null    # 새 run-layout
  ls -d  ${OUTPUT_ROOT}/main_7b/*/*/epoch_* 2>/dev/null    # legacy flat layout
  ```

---

## 10. 프로세스 / 캐시 위생 (실패가 누적되면 청소)

학습 잡이 죽었거나 eval이 OOM/CUDA error로 죽으면 **다음 tick이 같은 자리에서 또 죽는다**. 에이전트는 다음 신호 중 하나라도 보이면 즉시 §10-2의 cleanup 루틴을 돌리고 다시 시도한다. **자동 cleanup을 매 tick마다 무조건 돌리지는 말 것** — 학습 잡 GPU/캐시를 잘못 건드릴 수 있다.

### 10-1. Cleanup 트리거 조건

다음 중 하나라도 만족:

1. 같은 `(model, method)` 셀이 **연속 2회 eval 실패** (로그 끝부분에 `CUDA error` / `OOM` / `NCCL` / `RuntimeError`).
2. `nvidia-smi`에서 **PID가 매핑되지 않은 phantom 메모리**가 보임 (사용 메모리 > 0인데 `nvidia-smi --query-compute-apps=pid,used_memory` 결과는 비어있음).
3. HF dataset 캐시락 에러: 로그에 `Cannot acquire lock` / `.lock` 파일 stale (`find $HF_DATASETS_CACHE -name "*.lock" -mmin +60`로 1시간+ 묵은 락이 있음).
4. eval 프로세스가 30분 이상 진행 없이 멈춰 있음 (로그 mtime 정체).

### 10-2. Cleanup 절차 (안전한 것부터 위에서 아래로)

```bash
# (a) 우리 eval 프로세스 중 좀비/멈춤만 정리. 학습 프로세스는 절대 건드리지 않음.
#     "tads.eval"로 시작하는 python만 타깃. 학습은 "tads.train" — 패턴이 다르므로 안전.
pgrep -af 'python.*-m tads\.eval' \
  | awk '{print $1}' \
  | xargs -r -I{} sh -c 'echo "killing PID {}"; kill -TERM {}; sleep 2; kill -KILL {} 2>/dev/null || true'

# (b) HF dataset cache 중 stale lock 제거 (60분 이상 묵은 것만)
find "${HF_DATASETS_CACHE:-${DATA_CACHE}/huggingface/datasets}" -name "*.lock" -mmin +60 -print -delete 2>/dev/null || true
find "${HF_HUB_CACHE:-${DATA_CACHE}/huggingface/hub}"        -name "*.lock" -mmin +60 -print -delete 2>/dev/null || true

# (c) 학습/eval 둘 다 안 돌고 있는데 GPU phantom 메모리가 있을 때만 GPU 리셋 시도.
#     일반 사용자 권한이면 `nvidia-smi --gpu-reset`는 실패함 — 이 경우 그냥 다음 tick에 재시도.
if ! pgrep -f 'python.*-m tads\.(train|eval)' >/dev/null; then
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    | awk -F',' '{print "still attached:", $0}' || true
  # sudo 권한이 있으면 phantom 장에만 reset (없으면 그냥 스킵):
  # sudo nvidia-smi --gpu-reset -i <idx>
fi

# (d) 우리 잡이 만든 stale lock/tmp 파일만 정리 (다른 사람 파일은 절대 건드리지 말 것)
find "${OUTPUT_ROOT}/cache" -maxdepth 3 -name "*.lock" -mmin +60 -user "$USER" -print -delete 2>/dev/null || true

# (e) Python bytecode 캐시 — eval 코드 갱신 후 stale .pyc 의심될 때만
find tads -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

### 10-3. Cleanup 후 재시도 정책

- 실패 셀을 즉시 다시 enqueue하되, **3번째 실패면 더 이상 자동 재시도하지 말고 보고**. 무한 retry는 GPU 낭비.
- 실패 누적 카운트는 `${EVAL_RESULTS_ROOT}/<model>/<method>/.fail_count` 파일에 정수로 기록. 성공하면 파일 삭제.

### 10-4. 절대로 손대지 말 것

- 학습 중인 프로세스 (`python -m tads.train ...`).
- `OUTPUT_ROOT/main_7b/**` 아래 어떤 파일이든 (체크포인트는 학습 결과물).
- 다른 사용자가 만든 락/캐시 파일 (`-user "$USER"` 필터 반드시 유지).
- `nvidia-smi --gpu-reset`를 학습 GPU에 절대 쏘지 말 것.

---

## 11. 파라미터 튜닝 가이드

§0-2 발산 알람(RED/YELLOW/BLUE)이 떴을 때 **에이전트가 어디를 봐야 할지** 가이드. 에이전트는 **튜닝을 직접 실행하지 말고**, 어떤 노브를 어떻게 돌릴지 사용자에게 제안만 한다 (학습은 사용자 담당, §0).

### 11-1. 알람별 1차 진단

| 알람 | 의미 | 1차로 의심할 것 |
|---|---|---|
| **RED** `tads_10 < random_10` | 셀렉션이 오히려 해로움 | (a) 셀렉터(PPO agent)가 발산. agent.lr·entropy_coef·clip_eps 검토. (b) Anchor가 noisy: `anchor.max_samples_for_pca`가 너무 작음. (c) `selection_ratio`가 너무 낮아 학습량 부족. |
| **YELLOW** `tads_10 << full_100` (>5%p 처짐) | 10% 데이터로 100% 격차 회수 실패 | (a) `train_epochs` 부족 (보통 3 → 4~5로 늘림). (b) 학습률이 너무 작음 (BASE-FULL이 충분히 학습되어 있으면 treatment는 더 빨리 수렴해야 함). (c) `selection_ratio`를 0.1 → 0.15로 살짝 올려보기. |
| **BLUE** `data_agent_10 > tads_10 + 1%p` | 경쟁 method가 앞섬 | (a) `tads.lam` 조정 (현재 1.0). λ=0이면 사실상 data_agent와 같음 — 둘 사이 sweep. (b) `anchor.layer_indices`를 `all` ↔ `middle_to_last` 비교. (c) `anchor.use_anchor: false`로 ablation. |

### 11-2. 주요 튜닝 노브 (config 경로 포함)

#### 학습 일반 (`configs/base.yaml`)

| 노브 | 기본값 | 보통 시도 범위 | 의미 |
|---|---|---|---|
| `learning_rate` | `2.0e-5` | 5e-6 ~ 5e-5 | 안 수렴하면 ×2, 발산하면 ÷2 |
| `train_epochs` | 3 | 2 ~ 5 | YELLOW일 때 +1~2 |
| `batch_size` × `grad_accum` | 2 × 4 = 8 | effective 8 ~ 32 | 노이즈 크면 effective batch 키우기 |
| `warmup_ratio` | 0.03 | 0.01 ~ 0.1 | 초반 loss 폭주 시 키우기 |
| `weight_decay` | 0.1 | 0.0 ~ 0.1 | over-fit 의심되면 유지/증가 |
| `gradient_clip` | 1.0 | 0.5 ~ 1.0 | grad-norm spike 시 줄이기 |
| `max_seq_len` | 512 | 512 / 1024 / 2048 | tydiqa/bbh가 잘리면 늘리기 (단 OOM 주의) |
| `use_8bit_optimizer` | false | true | A100 80GB 풀FT에서 OOM 시 켜기 |
| `attn_implementation` | null | `flash_attention_2` | 속도 30% ↑ (flash-attn 설치되어 있어야 함) |

#### Selection (TADS / data_agent 공통)

| 노브 | 기본값 | 보통 시도 범위 | 의미 |
|---|---|---|---|
| `selection_ratio` | exp별 다름 (0.1) | 0.05 ~ 0.3 | 10%-셀렉션 매트릭스에선 0.1 고정이 원칙이지만, RED 진단용으로 잠시 0.15로 sweep 가능 |
| `episode_batch_size` | 16 (tads/data_agent) | 1 ~ 32 | collect_episode 속도. NCCL hang 의심되면 1까지 내리기 |

#### PPO Agent (`configs/base.yaml: agent.*`)

| 노브 | 기본값 | 보통 시도 범위 | 의미 |
|---|---|---|---|
| `agent.lr` | 3.0e-4 | 1e-4 ~ 1e-3 | RED일 때 1차 의심 |
| `agent.clip_eps` | 0.2 | 0.1 ~ 0.3 | PPO 업데이트가 너무 공격적/소극적일 때 |
| `agent.entropy_coef` | 0.01 | 0.001 ~ 0.05 | 셀렉터가 한 쪽으로 수축하면 키우기 |
| `agent.ppo_epochs` | 4 | 2 ~ 8 | over-fit ↔ 학습 부족 |
| `agent.advantage_mode` | `group_relative` | (변경 안 권장) | 변경 전 ablation 필요 |

#### Anchor (TADS 전용, `configs/base.yaml: anchor.*` + `configs/methods/tads.yaml`)

| 노브 | 기본값 | 보통 시도 범위 | 의미 |
|---|---|---|---|
| `tads.lam` | 1.0 | 0.0 ~ 1.0 | 0.0이면 data_agent와 동등. λ sweep으로 anchor 효과 측정. |
| `tads.use_anchor` | true | true/false | false로 두면 anchor ablation |
| `anchor.layer_indices` | `all` | `all` / `middle_to_last` / 명시 리스트 | `all`이 paper-faithful. memory 부족 시 middle_to_last |
| `anchor.max_samples_for_pca` | 2000 | 1024 ~ 4096 | RED일 때 노이즈 의심되면 늘리기 |
| `anchor.pca_batch_size` | 4 | 1 ~ 16 | OOM 시 줄이기 |

### 11-3. 튜닝 워크플로 (에이전트가 제안해야 할 순서)

새 run-layout 덕분에 **튜닝은 같은 셀(`main_7b/<model>/<method>/`) 안에서 자동 격리**된다.
사용자가 `--run_suffix=<param>` 또는 `--run_tag=<tag>` 만 다르게 줘서 학습하면 매번
`runs/<timestamp>_<suffix>/` 가 새로 생기므로 **이전 결과는 절대 손상되지 않는다**.
`_latest` 포인터는 가장 최근 학습을 가리키므로 **에이전트의 자동 eval은 자동으로 새 튜닝 결과를 평가**한다.

1. **먼저 ablation 두 개로 원인 격리**:
   - `tads.use_anchor: false` (= data_agent 동등) → 점수가 data_agent_10에 수렴하는가? (yes면 셀렉터는 정상, anchor가 문제)
   - `tads.lam: 0.0` → λ만 끄고 anchor는 유지. 차이를 보면 λ vs anchor 기여 분리 가능.
2. **§11-1 표대로 1차 노브 1~2개만 sweep** (한 번에 여러 개 동시 변경 금지 — 원인 추적 불가).
3. **튜닝 학습 명령 권장 형태** (사용자에게 제안):
   ```bash
   torchrun -m tads.train \
       --config configs/experiments/main_7b/llama2/tads_10.yaml \
       --run_suffix=lr5e5 \
       --override learning_rate=5e-5
   ```
   →  `runs/20260516_140000_lr5e5/` 에 떨어지고, 끝나면 `_latest`가 그쪽으로 옮겨감.
   기존 baseline run (`runs/20260515_180000/`)은 그대로 보존됨.
4. **이전 baseline run으로 다시 비교 평가**가 필요하면 사용자에게 다음 명령 제안:
   ```bash
   python -m tads.eval --config <cfg> --run_tag=20260515_180000 --benchmarks ...
   ```
   결과는 `runs/<해당 tag>/epoch_<N>/eval/` 에 저장됨 (같은 셀이지만 다른 run).
5. 튜닝 결과 비교는 `experiments.md`의 **별도 섹션 ("Tuning sweeps")**에 따로 기록.
   메인 16셀 표는 `_latest` 결과만 반영. 사용자가 튜닝 실패 후 baseline로 되돌리려면
   `_latest`를 옛 tag로 다시 가리키면 됨 (또는 그 tag로 한 번 더 학습 — `--run_tag=<old>`로 resume).

### 11-4. 에이전트가 결정하지 않는 것 (사용자 컨펌 필요)

- 새 학습 잡 실행 (학습은 사용자 담당).
- config 파일 자체 수정 (제안만 하고 사용자가 직접 편집).
- 메인 매트릭스 셀의 체크포인트 삭제/덮어쓰기.

---

## 12. 빠른 sanity check

에이전트가 처음 떴을 때 (또는 cron 첫 tick에서) 이걸 먼저 돌려서 환경이 살아있는지 확인:

```bash
cd /home/jieun/kms/tads
source scripts/setup_env.sh
# 새 run-layout: <ckpt_root>/_latest -> runs/<tag>/epoch_<N>/
ls -ld ${OUTPUT_ROOT}/main_7b/*/*/_latest 2>/dev/null | head
# 그 안의 sealed epoch들
for r in ${OUTPUT_ROOT}/main_7b/*/*/_latest; do
  for e in "$r"/epoch_*; do
    [ -f "$e/_complete" ] && echo "OK   $e"
  done
done | head
# Legacy flat layout (옛날 학습용)도 확인
ls -d ${OUTPUT_ROOT}/main_7b/*/*/epoch_* 2>/dev/null | head
ls -d ${EVAL_RESULTS_ROOT}/*/* 2>/dev/null | head
ls -l ${EVAL_RESULTS_ROOT}/experiments.md 2>/dev/null || echo "experiments.md 아직 없음 — §0-4 템플릿으로 생성"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

네 줄 다 합리적인 출력이 나오면 §7 루프(또는 §9 cron tick)를 시작해도 된다.
