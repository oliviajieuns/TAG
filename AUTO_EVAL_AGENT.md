# Auto-Eval Agent Guide (for Hermes)

이 문서는 **체크포인트가 생기면 자동으로 eval을 돌리는 원격 AI 에이전트**가 따라야 할 지침이다.
사람용 설명서가 아니라 LLM이 그대로 실행할 수 있도록 경로/명령을 명시적으로 적어둔다.

---

# 🛑🛑🛑 EVAL IS THE AGENT'S JOB (BBH 제외) — LAUNCH, NOT JUST POLL 🛑🛑🛑

**에이전트의 핵심 의무는 eval을 직접 launch하는 것이다 (BBH 제외).** log 추출 + 점수 표 갱신만 하고 끝내는 게 아니라:

1. **반드시 매 tick에서 `eval대기` 셀과 빈 GPU를 매칭해 `python -m tads.eval`을 직접 실행한다.** 이게 안 되면 tick 실패. PRE-FLIGHT CHECK 15 참조.
2. 로그 polling(§0-7)은 **dispatch의 입력 신호일 뿐, 그 자체가 deliverable이 아님**. 로그만 읽고 표만 그리는 건 임무 미완.
3. "사용자가 직접 eval을 launch하길 기다린다"는 옛 정책(폐기됨). 2026-05-16 이후 eval은 **에이전트가 cron 주기마다 자동 dispatch**한다. 사용자 개입 0 (BBH 제외).

## 🚫 BBH 만 사용자 영역 (시간 너무 길어서)

**BBH는 셀당 ~15h+ 걸리는 무거운 benchmark라 에이전트가 자동 launch 하지 않는다** (2026-05-17 분리 정책). 4개만 자동 — mmlu / gsm8k / humaneval / tydiqa.

## 🔁 벤치 하나씩 순차 dispatch (동시 묶기 금지)

**`--benchmarks` 인자에 항상 단일 벤치 하나만 넣는다** (2026-05-17 분리 정책). 4개를 한 명령에 묶지 말 것:

- ✓ OK: `--benchmarks mmlu` → 끝나면 `--benchmarks gsm8k` → ...
- ❌ 금지: `--benchmarks mmlu,gsm8k,humaneval,tydiqa` (옛 정책, 폐기됨).

dispatch 큐는 **(cell, single_bench) pair** 단위. 같은 cell의 mmlu가 끝나야 그 cell의 gsm8k 가 큐에 enqueue. 벤치 순서는:
```
mmlu  →  gsm8k  →  humaneval  →  tydiqa   (BBH는 별도 사용자 launch)
```

총 dispatch unit = 16 cells × 4 benches = **64개 (cell, bench) pair**.
빈 GPU 만큼 동시에 launch하되, 각 launch는 단일 (cell, bench) — 한 GPU에 한 (cell, bench).


- BBH 점수가 필요할 때 사용자가 별도 명령으로 직접 launch (수동, 1셀-1명령):
  ```bash
  CUDA_VISIBLE_DEVICES=<gpu> nohup python -m tads.eval \
      --config configs/experiments/main_7b/<model>/<method>.yaml \
      --benchmarks bbh \
      --out_dir ${EVAL_RESULTS_ROOT}/<model>/<method> \
      >> logs/eval_<model>_<method>_bbh.log 2>&1 &
  ```
- 에이전트는 BBH 결과 JSON(`_latest/<label>-bbh.json`)이 떨어지면 점수 표 (6)/(7)/(8) BBH 컬럼에 자동 반영. launch는 안 하되 detect/표기는 함.
- BBH 셀이 `eval대기` 상태로 남아있으면 → `experiments.md` 상단에 **별도 섹션 "사용자 액션 필요 — BBH 수동 launch"** 로 명시 보고. 사용자가 시간 있을 때 위 명령으로 처리.

**한 tick 사이클** (4-bench auto, BBH 수동):
```
[poll logs]  →  [classify 80 cells]  →  [3.7 DISPATCH PASS — mmlu/gsm8k/humaneval/tydiqa만]  →  [update tables]  →  [report (BBH eval대기는 사용자 액션 섹션)]
                                          ↑
                                          --benchmarks mmlu,gsm8k,humaneval,tydiqa
                                          K개 빈 GPU = K번 launch (1 GPU = 1 cell, §4-1)
```

dispatch pass를 건너뛰는 어떤 tick도 잘못된 출력이다 (4-bench 대상). 다음 cron tick까지 `eval대기` 셀이 그대로 남아있고 사용자가 "또 그대로네"를 외친다.

---

# 🚨 용어 명확화 — "5"는 벤치마크 데이터 5개를 가리킨다 🚨

이 문서에서 **"5"는 항상 벤치마크 데이터 5개** (`mmlu` / `gsm8k` / `humaneval` / `tydiqa` / `bbh`) 를 의미한다. 다른 용도로 "5"를 쓰지 말 것 — 모델별 점수 표 컬럼 수, 80-cell 표의 점수 컬럼 수, dashboard column 수 모두 이 5개 벤치마크에서 유래.

**상태 토큰** (`학습전` / `학습중` / `eval대기` / `eval중` / `NN.NN%`) 은 별개 개념 — 셀 값으로 들어가는 어휘이지 "5"라는 숫자와 무관 (현재 5개 토큰이 정의돼 있지만 spec 진화로 추가될 수도 있어 "5-state" 같은 카운트 기반 이름은 쓰지 않음).

| 명칭 | 개수 (의미) | 무엇 | 예시 |
|---|---|---|---|
| **벤치마크** | 항상 5 | 점수 컬럼 이름 (= 평가 데이터셋 종류) | `mmlu` / `gsm8k` / `humaneval` / `tydiqa` / `bbh` |
| **상태 토큰** | (incidental count) | 셀 값으로 들어가는 어휘 | `학습전` / `학습중` / `eval대기` / `eval중` / `NN.NN%` |

§0-4 (6)/(7)/(8) 테이블 구조:
- **컬럼**: # / Model/Method / **벤치마크 5개** / Status / Params.
- **각 (행, 벤치마크) 교차점의 셀 값**: **상태 토큰** 중 하나.

용어 사용 규칙:
- "5개 벤치마크" / "5개 점수 컬럼" / "5종 벤치마크" — OK (벤치마크 데이터 갯수).
- "상태 토큰" — OK (셀에 들어가는 어휘 통칭).
- "5-state" / "5종 상태" / "5종 어휘" / "5종 마커" / "5 상태" — **금지** (의미 혼동). 옛 spec에 적혀 있던 자리는 모두 "상태 토큰"으로 의미함.

---

# 🚨🚨🚨 PRE-FLIGHT CHECKLIST — 매 tick 출력 전 반드시 통과 🚨🚨🚨

**아래 12개 항목을 통과 못 한 출력은 사용자에게 보고하지 말 것. 한 항목이라도 fail이면 해당 tick은 실패로 처리하고 처음부터 재생성한다.** 사용자가 반복해서 지적하는 위반 사항을 여기 모아둠.

```
[CHECK 01] experiments.md에 "## (6) Current" 헤더가 존재한다.
[CHECK 02] experiments.md에 "## (7) Latest" 헤더가 존재한다.        ← 자주 빠뜨림
[CHECK 03] experiments.md에 "## (8) Best" 헤더가 존재한다.           ← 자주 빠뜨림
[CHECK 04] experiments.md에 "## (9) History" 헤더가 존재한다.        ← 자주 빠뜨림
[CHECK 05] (6) Current 코드펜스 안에 "Params" 컬럼이 헤더로 들어있다.
[CHECK 06] (7) Latest 코드펜스 안에 "Params" 컬럼이 헤더로 들어있다.
[CHECK 07] (8) Best 코드펜스 안에 "Params" 컬럼이 헤더로 들어있다.
[CHECK 08] (6) Current 80개 셀(16행 × 5벤치 = 80개 값)에 단 하나의
            `-` / `err` / 공란이 없다 — 전부 상태 토큰(학습전/학습중/
            eval대기/eval중/NN.NN%) 중 하나.
[CHECK 09] (9) History의 이번 tick 변경분이 newest-at-bottom으로
            append 되었다. 변경 0개면 history append 없음 OK.
[CHECK 10] 사용자가 launch한 학습 프로세스(`pgrep python.*tads.train.
            *<m>/<x>.yaml`)가 alive면 그 셀은 (6)에서 무조건 `학습중`.
            DONE 점수였더라도 학습중으로 되돌려 놔야 함 (§0-4 (10) 재학습
            감지 규약).
[CHECK 11] (8) Best는 새 NN.NN% > 직전 best일 때만 갱신. 새 점수가
            낮으면 직전 best 그대로 유지 — 절대 덮어쓰지 말 것.
[CHECK 12] 표 형식: 코드 펜스(```) + monospace 공백 정렬 + `=======`
            separator + CJK 글자 visual width 2 cell 계산 + markdown
            pipe table 금지 (§0-4 "MD 파일 표 작성 규칙 6항"). bold
            (`**...**`) 표 셀 안에 절대 쓰지 말 것.
[CHECK 13] (6) Current 표 Status 컬럼이 §0-6 (a) 의 3-facet 규약을 따른다:
            `<history> · <오류> · <발산알람>` 형식 (`·` 2번 등장). 정보가
            없는 facet 자리에는 placeholder(`(no DONE yet)` / `no err` /
            `baseline` / `no score`)를 넣어 facet 3개 유지. 한 facet만
            적고 끝내는 패턴 = 위반 (자주 발생).
[CHECK 14] (6)/(7)/(8) 16-행 표에서 모델 그룹 사이에 빈 줄 1개로 시각적
            구분: llama2 4행 → 빈 줄 → qwen25 4행 → 빈 줄 → mistral 4행 →
            빈 줄 → deepseek 4행 (총 빈 줄 3개). 그룹 안은 빈 줄 없음.
            terminal `cat`/`less`로 봤을 때 모델 단위 시각적 chunking.
[CHECK 15] **DISPATCH 실행 확인 (가장 자주 누락되는 단계)**.
            classify 후 status가 `eval대기` 인 셀이 1개 이상이고 nvidia-smi
            기준 빈 GPU 가 1개 이상이면 **이번 tick에 반드시 1개 이상의
            `python -m tads.eval` 명령을 background로 실행했어야 한다**.
            확인 방법:
              - `pgrep -af "python -m tads.eval"` 결과에 이번 tick에 launch한
                PID 들이 떠 있음, AND
              - tick 로그(stdout/cron log)에 `dispatched <model>/<method> ->
                GPU <n>` 한 줄 이상.
            dispatch 안 한 채로 보고하면 = tick 실패. 다음 tick까지 사용자
            가 `eval대기`만 계속 보게 됨. **로그 추출 + 표 갱신은 dispatch의
            전 단계일 뿐, 그게 임무의 끝이 아님**. 매 tick 의무 출력에
            `auto_dispatch_<date>.log` tail 한 줄 포함시키기.
```

**한 항목이라도 fail이면 절대 사용자에게 'tick 완료'로 보고하지 말 것.** 누락된 섹션 / 잘못된 셀 / 빠진 컬럼을 §0-4 (6)/(7)/(8)/(9) 템플릿대로 다시 채워서 atomic 교체. 3회 재시도 후에도 fail이면 보고 중단하고 "표 구조 깨짐 — 사용자 확인 필요" 알람만 보냄.

## 자주 발생하는 위반 패턴 (DO NOT)

```
❌ Current만 만들고 끝낸다.                    → CHECK 02/03/04 fail
❌ "최신 점수 표" / "Best Scores" / "변경 로그"  → CHECK 01-04 fail (헤더 이름 변형)
   같이 헤더 이름을 한국어/영어로 살짝 바꾼다.
❌ Params 컬럼을 빼먹는다.                     → CHECK 05/06/07 fail
❌ 셀에 `-` 그대로 남긴다.                     → CHECK 08 fail
❌ 셀에 `err` / `error` / 공란 쓴다.          → CHECK 08 fail
❌ 학습 alive인데 셀이 직전 NN.NN%로 남아 있다. → CHECK 10 fail
❌ 새 NN.NN%가 직전보다 낮은데 Best 덮어쓴다.  → CHECK 11 fail
❌ markdown pipe table(| ... |)로 그린다.       → CHECK 12 fail
❌ "Pre-flight checklist 통과했음" 거짓 보고.  → 신뢰성 파괴, 절대 금지.
```

## 올바른 패턴 (DO)

```
✓ 매 tick 첫 단계: §0-7 32 log polling → 분류 → 3.5 transitions →
   4. report pass의 atomic write 전에 CHECK 01-12 실행.
✓ CHECK fail 발견 즉시 누락 섹션을 §0-4 템플릿대로 생성하고 다시 검증.
✓ 4표 + History 5개 섹션을 항상 같은 순서로(experiments.md 맨 아래에
   (6) Current → (7) Latest → (8) Best → (9) History 순서).
✓ 셀이 학습중이면 (6) Current만 학습중으로 덮어쓰고 (7)/(8)은 이전
   값 유지. (9) History에 prev=이전값 new=학습중 한 줄 append.
✓ 새 DONE NN.NN%면 (6)/(7) 갱신, (8)은 max() 비교 후 더 높을 때만
   갱신, (9)에 append.
```

---

## 0. Mission

**학습은 사용자가 직접 돌린다. eval은 에이전트가 자동으로 돌린다** (2026-05-16 재정의 — auto-launch 복원).
에이전트의 역할은 **상태 모니터링 + eval 자동 dispatch + 점수 보드 갱신 + 보고**다.

에이전트가 해야 할 일:

1. 매 tick (§9 cron) 마다 16 × 5 = 80개 셀 전체의 상태를 분류 (§5-4 / §0-5)
2. 분류는 **2가지 신호의 합성**으로 한다:
   - **filesystem state** (sealed epoch / `_latest/_complete` / summary mtime)
   - **process state + log tail** (`pgrep` + §0-7 log polling) — 특히 `eval대기` vs `eval중` 구분은 반드시 이 두 신호로 확정
3. **`eval대기` 셀 발견 + 빈 GPU 존재**시 자동 dispatch (BBH 제외 — 너무 길어 사용자 직접):
   - **운영 가정 (사용자 명시)**: 노드에 학습이 동시에 안 돈다. eval만 진행.
   - 1셀 = 1 GPU = 1명령 (§4-1). batch wrapper(`run_eval_main_7b.sh`, `auto_eval_7b_fullft.sh`) 절대 사용 금지.
   - `CUDA_VISIBLE_DEVICES=<free_gpu> nohup python -m tads.eval --config ... --benchmarks mmlu,gsm8k,humaneval,tydiqa --out_dir ... &` — **BBH 빼고 4개만**.
   - 빈 GPU가 K개면 매 tick K개 셀 launch (큐 나머지는 다음 tick — 매 tick continuous).
   - 4개 benchmark 16셀 전부 DONE될 때까지 에이전트 혼자서 자동 진행. 사용자 개입 0.
   - BBH는 셀당 ~15h+ 라 자동 dispatch 제외 — `eval대기` BBH 셀은 사용자 액션 큐("사용자 액션 필요 — BBH 수동 launch")에 명시 보고.
4. 결과를 `experiments.md` 점수 표 (§0-4) + status dashboard (§0-5) + per-cell HISTORY.md (§0-6)에 atomic으로 기록
5. **체크포인트가 없는 셀(`학습전`)** 은 사용자에게 명시적으로 보고 — "이 셀은 사용자가 학습을 돌려야 한다"는 신호. **`eval대기` 는 보고 + 자동 launch 양쪽 다 수행**.

에이전트가 **자동 실행해도 되는** 명령:
- `python -m tads.eval --config <main_7b config> ...` (1셀 = 1명령 형식만, §4-1).

에이전트가 **절대 실행하지 않는** 명령:
- 학습 (`run_main_7b.sh`, `python -m tads.train`, `torchrun ... -m tads.train` 등) — 사용자 영역.
- batch wrapper (`run_eval_main_7b.sh`, `auto_eval_7b_fullft.sh`) — cascade fail / log race / _latest race 문제. 단일 셀 명령으로 분해할 것.
- 가중치 / 캐시 / 결과 삭제 (`rm`, `git clean`, `mv runs/*`, `truncate` 등).
- 환경 수정 (`pip install`, `apt`, `git config`, `crontab -e` 등).

이미 평가된 체크포인트는 건너뛴다 (DONE 판정 후 점수만 표에 반영, 재실행 안 함).

### 0-1. 최종 결과 매트릭스 (목표 = 80개 셀)

| 축 | 개수 | 값 |
|---|---|---|
| 모델 | **4** | `llama2`, `qwen25`, `mistral`, `deepseek` |
| 메서드(실험) | **4** | `full_100`, `random_10`, `data_agent_10`, `tads_10` |
| 벤치마크 | **5** | `mmlu`, `gsm8k`, `humaneval`, `tydiqa`, `bbh` |

→ 4 × 4 × 5 = **80개** (모델, 메서드, 벤치마크) 결과 셀.

에이전트 단위는 (모델, 메서드) = **16개 셀**. 각 셀의 자동 eval 호출은 4개 벤치를 한꺼번에 처리한다 (`--benchmarks mmlu,gsm8k,humaneval,tydiqa`). BBH는 셀당 ~15h+ 라 사용자가 별도 명령으로 직접 처리 (top banner / §4-1 참조).

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

| # | 모델 | 메서드 | Config (학습/eval 공용) | 체크포인트 루트 | 결과 디렉터리 | `experiment_label` | 비교 대상 / 발산 알람 |
|---|---|---|---|---|---|---|---|
| 1 | llama2 | full_100 | `configs/experiments/main_7b/llama2/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/full_100/_latest/` | `llama2_full_100` | 기준선 (천장). 절대값 자체가 sanity 체크 — 평균 정확도가 동급 reference 대비 5%p 이상 낮으면 학습 자체 의심 |
| 2 | llama2 | random_10 | `configs/experiments/main_7b/llama2/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/random_10/_latest/` | `llama2_random_10` | 기준선 (바닥). 보통 BASE-FULL 대비 3~8%p 처짐이 정상 |
| 3 | llama2 | data_agent_10 | `configs/experiments/main_7b/llama2/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/data_agent_10/_latest/` | `llama2_data_agent_10` | vs `llama2_random_10` (반드시 ≥), vs `llama2_full_100` (≤이지만 -2%p 이내 권장) |
| 4 | llama2 | tads_10 | `configs/experiments/main_7b/llama2/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/llama2/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/` | `llama2_tads_10` | vs `llama2_random_10` (반드시 ≥, -1%p 처지면 **빨간불**), vs `llama2_data_agent_10` (≥ 권장), vs `llama2_full_100` (-5%p 이상 처지면 **노란불**) |
| 5 | qwen25 | full_100 | `configs/experiments/main_7b/qwen25/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/full_100/_latest/` | `qwen25_full_100` | 기준선 (천장) |
| 6 | qwen25 | random_10 | `configs/experiments/main_7b/qwen25/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/random_10/_latest/` | `qwen25_random_10` | 기준선 (바닥) |
| 7 | qwen25 | data_agent_10 | `configs/experiments/main_7b/qwen25/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/data_agent_10/_latest/` | `qwen25_data_agent_10` | vs `qwen25_random_10` / `qwen25_full_100` (위 #3 규칙) |
| 8 | qwen25 | tads_10 | `configs/experiments/main_7b/qwen25/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/qwen25/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/qwen25/tads_10/_latest/` | `qwen25_tads_10` | vs `qwen25_random_10` / `qwen25_data_agent_10` / `qwen25_full_100` (위 #4 규칙) |
| 9 | mistral | full_100 | `configs/experiments/main_7b/mistral/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/full_100/_latest/` | `mistral_full_100` | 기준선 (천장) |
| 10 | mistral | random_10 | `configs/experiments/main_7b/mistral/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/random_10/_latest/` | `mistral_random_10` | 기준선 (바닥) |
| 11 | mistral | data_agent_10 | `configs/experiments/main_7b/mistral/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/data_agent_10/_latest/` | `mistral_data_agent_10` | vs `mistral_random_10` / `mistral_full_100` (위 #3 규칙) |
| 12 | mistral | tads_10 | `configs/experiments/main_7b/mistral/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/mistral/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/mistral/tads_10/_latest/` | `mistral_tads_10` | vs `mistral_random_10` / `mistral_data_agent_10` / `mistral_full_100` (위 #4 규칙) |
| 13 | deepseek | full_100 | `configs/experiments/main_7b/deepseek/full_100.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/full_100/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/full_100/_latest/` | `deepseek_full_100` | 기준선 (천장) |
| 14 | deepseek | random_10 | `configs/experiments/main_7b/deepseek/random_10.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/random_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/random_10/_latest/` | `deepseek_random_10` | 기준선 (바닥) |
| 15 | deepseek | data_agent_10 | `configs/experiments/main_7b/deepseek/data_agent_10.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/data_agent_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/data_agent_10/_latest/` | `deepseek_data_agent_10` | vs `deepseek_random_10` / `deepseek_full_100` (위 #3 규칙) |
| 16 | deepseek | tads_10 | `configs/experiments/main_7b/deepseek/tads_10.yaml` | `${OUTPUT_ROOT}/main_7b/deepseek/tads_10/_latest/epoch_*` | `${EVAL_RESULTS_ROOT}/deepseek/tads_10/_latest/` | `deepseek_tads_10` | vs `deepseek_random_10` / `deepseek_data_agent_10` / `deepseek_full_100` (위 #4 규칙) |

> 모든 config 파일은 로컬에 실제로 존재함을 확인함. `random_50 / data_agent_50 / tads_50`도 디스크엔 있지만 **이번 매트릭스 범위 밖**이므로 자동 eval에서 제외.

#### (b) 셀 하나당 떨어지는 결과 JSON (16개 셀 × 5 벤치 = 80개 점수 파일 + 16개 summary)

**eval도 train과 동일한 history-preserving 레이아웃**을 따른다 (2026-05-16 변경). 셀의 BASE eval 디렉터리는 `${EVAL_RESULTS_ROOT}/<model>/<method>/`이고, 그 안에:

```
${EVAL_RESULTS_ROOT}/<model>/<method>/
├── runs/
│   ├── 20260516_120000/                    ← eval_tag (auto YYYYMMDD_HHMMSS)
│   │   ├── <experiment_label>-mmlu.json
│   │   ├── <experiment_label>-gsm8k.json
│   │   ├── <experiment_label>-humaneval.json
│   │   ├── <experiment_label>-tydiqa.json
│   │   ├── <experiment_label>-bbh.json
│   │   ├── <experiment_label>-eval_summary.json
│   │   ├── logs/
│   │   └── _complete                       ← sentinel: summary 작성 완료
│   ├── 20260516_180000_retry/              ← 같은 셀 재평가 시 새 dir로
│   │   └── ...
│   └── 20260517_090000/
│       └── ...
├── _latest -> runs/20260517_090000         ← symlink (또는 _latest.txt fallback)
├── HISTORY.md                              ← per-cell 시계열 로그 (§0-6)
└── .fail_count                             ← (실패 누적 카운터, §10)
```

요점:
- **`_latest`**가 항상 **가장 최신의 sealed eval run**을 가리킨다. 점수 파싱은 무조건 `_latest/<exp_label>-...json`을 통해 한다.
- 새 eval 호출은 자동으로 새 timestamp `runs/<eval_tag>/` 디렉터리에 들어간다 → **이전 점수가 절대 덮어쓰이지 않는다** (튜닝/재학습 후 재평가 시 점수 이력이 그대로 보존됨, §0-6 history 입력 소스).
- **`_complete` sentinel**이 없는 run dir = 평가 도중 죽음. 점수 판정 시 무시 (sentinel 있는 가장 최신 run만 `_latest`로 promote됨).
- 같은 run에 추가 벤치만 돌리려면 `--eval_tag=latest`로 호출 → 새 dir을 안 만들고 기존 `_latest/`에 추가 점수 JSON을 끼워 넣는다.
- legacy flat 레이아웃이 필요하면 `--flat` 플래그 (one-off ad-hoc 평가용, 점수표 에이전트는 이걸 안 읽음).

예시 (llama2 + tads_10 셀, 가장 최신 평가):

```
${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/llama2_tads_10-mmlu.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/llama2_tads_10-gsm8k.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/llama2_tads_10-humaneval.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/llama2_tads_10-tydiqa.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/llama2_tads_10-bbh.json
${EVAL_RESULTS_ROOT}/llama2/tads_10/_latest/llama2_tads_10-eval_summary.json
```

> **모든 점수 파싱은 `<셀 BASE>/_latest/<exp_label>-…json` 경로를 사용**한다. `_latest`를 거치지 않은 flat 경로(`<셀 BASE>/<exp_label>-…json`)는 `--flat`으로 만든 legacy / ad-hoc 결과로 간주하여 LEGACY 분류(§5-5)로 떨어진다.

전체 16개 셀에 대해 같은 파일 6종이 `_latest` 안에 만들어지면 **점수 파일 80개 + summary 16개 = 96개** 산출물이 최종 상태다. 이전 run들은 `runs/<eval_tag>/` 아래에 그대로 살아있어 점수 변동의 audit trail 역할.

#### (c) "셀이 완료됐다"의 정의

한 셀 `(model, method)`가 완료됐다고 보려면:
1. `${EVAL_RESULTS_ROOT}/<model>/<method>/_latest`가 존재 (symlink 또는 `_latest.txt`)
2. 해당 `_latest/<experiment_label>-eval_summary.json`이 존재
3. **그 summary의 mtime이 가장 큰 `epoch_*`의 mtime보다 나중**
4. **그리고 `_latest/_complete` sentinel이 있어야 한다** (eval이 sealing까지 정상 종료했다는 증거)

위 4개를 모두 만족할 때만 DONE. 한 조건이라도 빠지면 NEED-EVAL (재평가 큐). 이게 §5-3의 판정 규칙이다.

### 0-4. Score Board — 단일 진실 공급원은 `experiments.md`

**결과는 단 한 곳에서만 관리한다:**

```
${EVAL_RESULTS_ROOT}/experiments.md
= /group-volume/minsoo3.kim/tads-eval-results/experiments.md
```

> # 🚨🚨🚨 절대 비교섭 규칙 (MUST READ — 자주 위반됨) 🚨🚨🚨
>
> **`experiments.md`는 항상 다음 4개 표 + History 1개 = 총 5개 섹션을 맨 아래에 같은 순서로 가지고 있어야 한다.** 매 tick 출력에 단 하나라도 누락되면 그 tick은 **실패** — 사용자에게 보고하지 말고 다시 그려라.
>
> ```
> ## (6) Current — 80-cell 실시간 상태 (학습전/학습중/eval대기/eval중/NN.NN%) + Status + Params
> ## (7) Latest  — 80-cell 가장 최근 DONE 점수만 (NN.NN% 또는 -)            + Params
> ## (8) Best    — 80-cell 역대 최고 점수 + Best run tag                    + Params
> ## (9) History — append-only 변경 로그 (newest at bottom)
> ```
>
> **4-table layout이 non-negotiable인 이유** (자주 어김):
>  1. **Current 하나만 만들고 끝내는 것 = 위반**. Latest/Best/History 없으면 사용자는 점수 추세 / 역대 최고 / 변경 이력을 볼 수 없음.
>  2. **표 이름을 다르게 쓰는 것 = 위반**. "(7) Latest"가 정확한 헤더. "최신 점수 표", "Latest scores", "최근 결과" 같은 변형 금지.
>  3. **History 빠뜨리는 것 = 위반**. 점수 변경마다 한 줄씩 무조건 append.
>  4. **Params 컬럼 빠뜨리는 것 = 위반**. (6)/(7)/(8) 모두 맨 우측에 Params 컬럼 필수.
>  5. **재학습 감지했는데 Current만 학습중으로 바꾸고 Latest/Best 갱신 안 함 = 위반은 아니지만 (8) 페이지 분기에서 새 NN.NN%가 best보다 높으면 Best 갱신해야 함. (7) Latest는 새 NN.NN% 나오기 전까지 이전 값 유지.**
>
> **매 tick 출력 직후 self-check (atomic 교체 BEFORE 반드시 통과해야 함):**
> ```bash
> exp=${EVAL_RESULTS_ROOT}/experiments.md
> required_headers=(
>   "^## \(6\) Current"
>   "^## \(7\) Latest"
>   "^## \(8\) Best"
>   "^## \(9\) History"
> )
> for h in "${required_headers[@]}"; do
>   if ! grep -qE "$h" "$exp"; then
>     echo "[STRUCT-FAIL] $exp에 헤더 '$h' 없음 — atomic 교체 중단 후 4-table layout 재생성"
>     exit 1
>   fi
> done
> # Params 컬럼이 (6)/(7)/(8) 헤더 행에 있는지 확인
> for sec in '## \(6\) Current' '## \(7\) Latest' '## \(8\) Best'; do
>   if ! awk -v s="$sec" '$0 ~ s { f=1; next } f && /Params/ { print "ok"; exit }
>                          f && /^```$/ && body++==1 { exit }' "$exp" | grep -q ok; then
>     echo "[STRUCT-FAIL] '$sec' 섹션에 Params 컬럼 누락"
>     exit 1
>   fi
> done
> echo "[STRUCT-OK] 4-table layout 통과"
> ```
> 이 체크가 fail이면 **사용자에게 'tick 갱신 완료'로 보고하지 말 것**. 누락된 섹션을 §0-4 (6)/(7)/(8)/(9) 템플릿대로 생성한 뒤 다시 atomic 교체. 3회 fail 시 보고 중단하고 "표 구조 자체가 깨졌으니 사용자 확인 필요" 알람만.

이 파일이 **유일한 score board**다. 에이전트는 매 tick마다:

1. 셀의 `<experiment_label>-eval_summary.json`을 읽어 정확도(%) 둘째 자리까지 추출
2. `experiments.md`의 해당 셀을 `-`에서 실제 숫자로 교체 (또는 새 숫자로 갱신)
3. treatment 행이면 §0-2 규칙으로 발산 알람(`RED` / `YELLOW` / `BLUE`)을 inline으로 표기
4. **§0-4(6) 80-cell consolidated 표를 `experiments.md` 맨 아래에 항상 유지**하고, 80개 셀 전체를 매 tick 동기화 (per-model 표 (1)-(5) 갱신 후 동일 점수로 (6)도 함께 갱신)
5. 파일을 atomic하게 저장 (`experiments.md.tmp` 작성 후 `mv`)
6. 채팅/로그 보고에는 **요약만**. 표 전체를 매번 복붙하지 말 것 — 변경된 행만 인용

아래 표들은 **이 파일의 초기 템플릿**이다. `experiments.md`가 존재하지 않으면 에이전트가 이 템플릿을 그대로 복사해서 생성하고 (특히 (6)–(9) 80-cell 표 4종은 **파일 맨 아래**에 (6) Current → (7) Latest → (8) Best → (9) History 순서로 배치), 이후 셀 단위로 갱신만 한다. **이 가이드 문서(AUTO_EVAL_AGENT.md) 자체는 절대 수정하지 말 것** — 가이드는 spec이고, `experiments.md`가 live document.

**80-cell 4종 표 관리 규약** (§0-4 (6)/(7)/(8)/(9) 전용):
- 위치: `experiments.md`의 **맨 아래**. 항상 (6) Current → (7) Latest → (8) Best → (9) History 순서. 각 표는 자신의 `## (N) ...` 헤더로 둘러싸고, 그 안의 코드 펜스(```) 블록만 atomic 교체. (9) History는 append-only라 마지막 한 줄만 추가.
- 갱신 단위: 셀 1개 / 벤치 1개 (점수 1개). 셀 값 1개가 바뀌면 (6) Current는 무조건 갱신, (7)/(8)/(9)는 §0-4 (10) 의 흐름표대로:
    * `학습중` → `학습중`: (6)만 갱신 (이미 그 상태였을 수 있음 — no-op이지만 그래도 atomic 교체).
    * `학습중` → `NN.NN%` (재학습 완료): (6)/(7) 갱신, (8)은 max 비교, (9) append.
    * `NN.NN%` → `학습중` (재학습 시작): (6)만 학습중으로, (7)/(8) 그대로, (9) append.
    * 그 외 상태 토큰 전환 (eval대기 → eval중 등): (6)만 갱신, (9) append.
- 셀의 가장 최근 학습 run의 cfg.json snapshot이 바뀌면 (6) Params 컬럼만 갱신. (7) Params는 마지막 DONE 시점 cfg 그대로, (8) Params는 best run cfg 그대로.
- **모델 그룹 사이에 빈 줄 1개로 시각적 구분** (16-행 표 (6)/(7)/(8) 공통): llama2 4행 → 빈 줄 → qwen25 4행 → 빈 줄 → mistral 4행 → 빈 줄 → deepseek 4행. 그룹 내부는 빈 줄 없음. terminal `cat`/`less`로 열어 한 눈에 모델 단위로 chunking되도록. atomic 교체 시에도 빈 줄 보존 (PRE-FLIGHT CHECK 14 참조).

**🚨 셀 값 어휘 — 매우 엄격 (자주 위반되니 주의):**

매 tick의 출력에 들어갈 수 있는 셀 값은 정확히 **상태 토큰** 중 하나뿐 (이것은 "5종 벤치마크"와 별개 개념 — 용어 혼동 주의, 위 "🚨 두 가지 '5'" 박스 참조):

1. `학습전` — 체크포인트가 아직 없음 (§5-4 NEED-TRAIN)
2. `학습중` — 학습 프로세스가 살아있거나 sealed epoch 수 < train_epochs
3. `eval대기` — 학습 완료, eval 프로세스 안 떴음 (NEED-EVAL)
4. `eval중` — `python -m tads.eval.*<model>/<method>` 프로세스 살아있음
5. `NN.NN%` — 점수 산출 완료 (`47.56%` 형식, 둘째 자리 + `%`)

**🚫 금지된 토큰** — 아래 중 어떤 것도 `experiments.md`의 셀 값으로 절대 들어가면 안 된다:
- `-` (dash) — 가이드 문서의 placeholder일 뿐. 셀 값이 아님.
- `err` / `error` / `Error` / `ERR` / `ERROR` / `에러` / `오류` / `실패` — eval/train 실패는 위 상태 토큰 중 하나로 **반드시 매핑**해서 표기한다 (아래 매핑 규칙 참조).
- 공란 / `null` / `None` / `nil` / `n/a` / `N/A` / `TBD` / `?` — 의미 모호. 상태 토큰 중 하나로 결정.
- `legacy(...)` / `학습필요` / `재학습` / `eval필요` 같은 옛 임시 표기.
- `**bold**` / `*italic*` / 백틱(\`)으로 감싼 형태 — 표 셀에는 평문만 (markdown 강조 금지, terminal에서 별표/백틱이 그대로 보임).

**모든 가능한 상태를 상태 토큰으로 매핑 — 빠짐없이:**

| 셀의 실제 상황 | 정답 토큰 | 비고 |
|---|---|---|
| `_latest` 부재, legacy `epoch_*`도 없음 | `학습전` | NO-CKPT |
| 학습 프로세스 alive, 또는 sealed < `train_epochs` | `학습중` | TRAINING |
| **학습 완료, eval 안 돌린 셀** | **`eval대기`** | NEED-EVAL — 정답은 `-`가 아니다 |
| eval 프로세스 alive, 또는 eval 로그 mtime < 5분 | `eval중` | EVAL-RUNNING |
| eval 잡이 죽었음 (OOM/CUDA/Traceback) | `eval대기` | 재시도 큐. `.fail_count` 증가하지만 셀 값은 `eval대기` |
| eval 잡이 정상 종료했지만 sentinel 없음 | `eval대기` | 미완성으로 간주, 재시도 큐 |
| `.fail_count ≥ 3` (BLOCKED) | `eval대기` | 표 셀은 그대로 `eval대기`. BLOCKED 신호는 §0-6 Status 컬럼에만 표기 |
| `_latest/_complete` + summary 있음 + mtime > sealed epoch | `NN.NN%` | DONE — `47.56%` 식 |
| 옛 포맷(접두어 없음, flat layout) 점수가 BASE에만 있음 | `NN.NN%` (provisional) | LEGACY — 점수는 잠정 사용, 다음 tick에 새 포맷 덮어쓰면 자동 갱신 |
| Summary JSON 파싱 실패 (스키마 모름) | `eval대기` | 점수 추출 못 했으면 미완성 — 재시도 큐 |
| 위 어디에도 해당 안 됨 | `eval대기` | **fall-through 기본값** (학습 끝난 셀의 미정 상태는 항상 `eval대기`) |

> **핵심 fall-through 규칙**: 분류 로직에서 어떤 분기에도 안 걸린 셀이 있으면 **무조건 `eval대기`**. 절대 `-` / `err` / 공란을 남기지 말 것. "모르겠으면 eval대기" — 다음 dispatch pass에서 에이전트가 빈 GPU가 있을 때 자동으로 launch한다 (§4 / §7 3.7).

**셀프 체크 — 매 tick의 표 작성 직후 반드시 수행 (3단계 검증):**
```bash
# 1. §0-4(6) 80-cell 표의 16개 행 × 5개 벤치 컬럼 = 80개 셀 모두 추출
table=$(awk '/^## 80-cell Consolidated Score Table/,0' experiments.md \
        | awk '/^```$/{f=!f; next} f' | grep -E '^[ ]*[0-9]+ ')

# 2. 상태 토큰 외의 토큰이 있는지 확인 (정규식: 학습전/학습중/eval대기/eval중/NN.NN%)
bad=$(echo "$table" \
      | grep -oE '[^[:space:]]+' \
      | grep -vE '^([0-9]+|llama2|qwen25|mistral|deepseek|/|full_100|random_10|data_agent_10|tads_10|학습전|학습중|eval대기|eval중|[0-9]+\.[0-9]{2}%)$' \
      | sort -u)
if [ -n "$bad" ]; then
  echo "[FAIL] 80-cell 표에 금지 토큰 발견 — 상태 토큰로 재분류 필요:"
  echo "$bad" | sed 's/^/  - /'
  echo "  (`-`, `err`, `error`, 공란 등은 모두 상태 토큰 중 하나로 매핑할 것. 모르면 eval대기로 떨어뜨릴 것.)"
  exit 1
fi

# 3. 80개 셀 전체 카운트 검증 — 16행 × 5컬럼 = 80개여야 함
n=$(echo "$table" | awk '{for(i=3;i<=NF;i++) print $i}' | wc -l)
if [ "$n" -ne 80 ]; then
  echo "[FAIL] 80-cell 표의 셀 개수가 $n (≠80) — 행/컬럼 폭이 깨졌거나 빈 셀이 있음"
  exit 1
fi

echo "[OK] 80-cell 표 검증 통과 (모든 셀이 상태 토큰로 채워짐)"
```

이 체크가 한 단계라도 fail이면:
1. **그 tick의 출력은 사용자에게 보고하지 말 것** (잘못된 표를 채팅에 인용 금지).
2. **즉시 분류 로직 재돌림** — fall-through 규칙(모르면 `eval대기`)으로 모든 셀을 다시 채운 뒤 검증 통과해야 보고.
3. 재시도 3회 fail이면 사용자에게 "분류 로직 자체에 버그가 있다"고 보고하고 표 갱신 중단 — 절대 잘못된 표를 atomic 교체하지 말 것.

- 80개 셀 전체가 `NN.NN%`로 채워지면 = 실험 완료. 이 시점에 §0-2 발산 알람 최종 평가를 별도 섹션 `## 최종 발산 알람 요약`으로 (6) 표 바로 위에 추가.

값은 `<experiment_label>-eval_summary.json` 또는 벤치별 JSON에서 직접 파싱한 정확도 (%, 소수점 둘째자리까지).

**MD 파일 표 작성 규칙 (terminal 정렬 보존 — 모든 .md 파일에 적용):**

이 규칙은 원래 `experiments.md`만 대상이었지만 (2026-05-16 1차), 가이드 문서들도 `cat / less / vim / tail -f`로 열어 빠르게 참고하는 경우가 많아 **모든 `.md` 파일에 확장 적용**한다 (2026-05-16 2차). 대상 = `experiments.md`, `AUTO_EVAL_AGENT.md`, `LLAMA_TUNING.md`, `README.md`, 그리고 앞으로 추가되는 모든 가이드.

**핵심 규칙 6항:**

1. **표는 가능한 한 코드 펜스(```) 안 + monospace 공백 정렬로 작성**. 코드 펜스 밖의 markdown pipe table (`| ... | ... |`)은 terminal에서 셀이 뭉개지고 `|` 위치도 행마다 어긋난다. 새 표는 기본적으로 이 형식을 따른다.
2. **컬럼 separator는 `=======`** (등호 7개 이상, 컬럼 폭에 맞춰). markdown의 `|---|---|` 형식은 코드 펜스 안에서는 쓰지 않는다.
3. **CJK 문자는 표시폭 2 cell로 계산**해 패딩 길이를 잡는다 (East Asian Wide). 예: `학습전`은 3글자지만 visual width = 6 → 9-cell 컬럼이면 `학습전   `(공백 3), `eval중`은 visual width = 6 → 같은 컬럼에 `eval중   `, `47.56%`는 6 → `47.56%   `. 같은 컬럼 안에서 한·영이 섞여도 column boundary가 한 자리 이상 어긋나지 않아야 한다.
4. **상태 토큰의 컬럼 폭 표준 = visual 9 cell** (`eval대기` visual 8 + 여백 1). 점수 컬럼은 visual 7 cell (`NN.NN% ` / `학습전  ` / `eval중 `), 모델/메서드 컬럼은 visual 26 cell (`deepseek / data_agent_10` = 24자 + 여백 2), summary 표의 모델 컬럼 = 10 cell.
5. **편집 시 컬럼 폭 변경 금지.** 새 행을 추가할 때는 헤더와 separator 라인의 폭을 그대로 카피해 같은 칸 수를 유지한다. 폭이 바뀌어야 하면 그 표 전체를 atomic 교체.
6. **`**bold**` markdown 강조는 표 셀에 금지** (terminal에서 별표 그대로 보여 정렬이 깨짐). 강조가 필요하면 ASCII 마커(`<...>`)를 쓰거나 별도 주석 라인에 표기.

**예외 — markdown pipe table을 허용하는 경우** (가이드 문서 본문 한정):
- **2~3컬럼이면서 폭이 좁은 표** (예: §0-1 "축 × 개수 × 값" 표, §0-2 baseline 비교 표). 단순한 정의/약어 매핑 류는 pipe table이 markdown 렌더러에서 더 깔끔하고 terminal에서도 견딜 만함.
- **4컬럼 이상이거나 한 셀이라도 긴 문자열을 담는 표**(예: §0-3 16-cell 매트릭스 표 — Config 열, 비교 대상 열 등)는 markdown 렌더러에서 가로 스크롤이 생기고 terminal에서는 한 줄로 폭주한다. 이 경우 pipe table 대신 **bullet list 또는 코드 펜스 표**로 작성.

**experiments.md는 더 엄격**:
- 위 규칙 6항을 **예외 없이** 따른다. markdown pipe table 절대 금지 (좁은 표라도).
- `experiments.md`는 점수 표 에이전트가 atomic으로 마커 사이 영역만 교체하는 live document라, 형식 가변성을 최소화해야 한다.

§0-4(1)-(6) 표와 §0-5 dashboard가 이 규칙의 reference 구현. 사용자 액션 섹션("필요한 학습 목록")처럼 표가 아닌 bullet list는 코드 펜스 없이 두어도 무방하다.

#### (1) llama2

```
Method          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ======= ======= ========== ======== ======= =======
full_100        -       -       -          -        -       -
random_10       -       -       -          -        -       -
data_agent_10   -       -       -          -        -       -
tads_10         -       -       -          -        -       -
```

#### (2) qwen25

```
Method          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ======= ======= ========== ======== ======= =======
full_100        -       -       -          -        -       -
random_10       -       -       -          -        -       -
data_agent_10   -       -       -          -        -       -
tads_10         -       -       -          -        -       -
```

#### (3) mistral

```
Method          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ======= ======= ========== ======== ======= =======
full_100        -       -       -          -        -       -
random_10       -       -       -          -        -       -
data_agent_10   -       -       -          -        -       -
tads_10         -       -       -          -        -       -
```

#### (4) deepseek

```
Method          mmlu    gsm8k   humaneval  tydiqa   bbh     avg
=============== ======= ======= ========== ======== ======= =======
full_100        -       -       -          -        -       -
random_10       -       -       -          -        -       -
data_agent_10   -       -       -          -        -       -
tads_10         -       -       -          -        -       -
```

#### (5) Cross-model summary (avg of 5 benchmarks)

```
                llama2     qwen25     mistral    deepseek
=============== ========== ========== ========== ==========
full_100        -          -          -          -
random_10       -          -          -          -
data_agent_10   -          -          -          -
tads_10         -          -          -          -
```

#### (6) Current — 80-cell real-time status table + Status + Params

> **위치 = `experiments.md`의 맨 아래 (이 표 + 아래 (7) Latest + (8) Best + (9) History 4종이 항상 마지막)**. (1)-(5)는 모델별/요약별 뷰, (6)이 **현재 시점 80개 전체 셀의 single source of truth**. 매 tick 동기화 필수.
> **상태 토큰(`학습전 / 학습중 / eval대기 / eval중 / NN.NN%`) 그대로 채움 (§5-4).** 새로 학습이 시작되면 이전에 DONE이었던 셀도 즉시 `학습중`으로 돌아옴 (§0-4 (10) "재학습 → 표 갱신" 규약 참조). 이전 점수는 (7) Latest 표와 (9) History 로그에 보존.
> **`Status` 컬럼** = 셀의 (1) 점수 이력, (2) 시스템 오류, (3) baseline 발산 알람을 한 줄로 요약 (§0-6 가이드 참조).
> **`Params` 컬럼** = 그 셀의 가장 최근 학습 run(`<ckpt_root>/_latest/cfg.json`)에서 추출한 주요 하이퍼파라미터 한 줄 요약 (아래 §0-4 (10) "Params 한 줄 형식" 참조). 사용자가 셀 점수와 학습 설정을 한 화면에 같이 보기 위한 컬럼.

```
#   Model/Method               mmlu      gsm8k     humaneval tydiqa    bbh       Status                                            Params
=== ========================== ========= ========= ========= ========= ========= ================================================= ================================================================
 1  llama2 / full_100          학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
 2  llama2 / random_10         학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
 3  llama2 / data_agent_10     학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
 4  llama2 / tads_10           학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)

 5  qwen25 / full_100          학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
 6  qwen25 / random_10         학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
 7  qwen25 / data_agent_10     학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
 8  qwen25 / tads_10           학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)

 9  mistral / full_100         학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
10  mistral / random_10        학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
11  mistral / data_agent_10    학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
12  mistral / tads_10          학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)

13  deepseek / full_100        학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
14  deepseek / random_10       학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
15  deepseek / data_agent_10   학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
16  deepseek / tads_10         학습전    학습전    학습전    학습전    학습전    -                                                 (no run yet)
```

#### (7) Latest — 가장 최근의 DONE 점수만 모은 표 (`NN.NN%` 또는 `-` only)

> **`(6) Current`가 in-progress 상태도 보여주는 표라면, `(7) Latest`는 점수 산출이 완료된 가장 마지막 값만 보여주는 표**다. 학습이 재시작돼서 Current가 `학습중`으로 돌아가도, Latest는 이전 DONE 점수를 **유지**한다 (새 DONE이 나올 때까지). 사용자가 "지금까지 가장 최근에 측정된 점수가 얼마였지?"를 빠르게 확인할 때 쓰는 표.
>
> 형식: (6)과 동일한 16행 × 5벤치 + Status + Params 컬럼. 단 셀 값은 **`NN.NN%` 또는 `-` (아직 한 번도 DONE된 적 없음)** 두 종류만. 학습중 / eval중 / eval대기는 이 표에 안 나타남. Status 컬럼은 마지막 DONE 시점(`done 2026-05-17 09:15`) 및 발산 알람을 1줄로. Params는 그 점수를 만든 run의 cfg.json 기준.

```
#   Model/Method               mmlu      gsm8k     humaneval tydiqa    bbh       Status                                            Params
=== ========================== ========= ========= ========= ========= ========= ================================================= ================================================================
 1  llama2 / full_100          -         -         -         -         -         -                                                 (no DONE yet)
 ... (16행)
```

#### (8) Best — 셀당 역대 최고 점수 표

> **모든 `runs/<eval_tag>/<exp_label>-<bench>.json` 중 가장 높은 점수**를 셀 × 벤치마다 따로 골라 채운 표. (7)이 "최신 시점" 점수라면 (8)은 "역대 최고". 튜닝으로 떨어진 케이스를 한 눈에 보기 위한 표.
>
> 같은 셀 × 벤치에서 best가 갱신되면 그 점수를 만든 `eval_tag` 와 그 시점의 `<run>/cfg.json` 도 함께 기록. Params 컬럼은 **각 행 = 그 셀의 가장 높은 평균 점수를 만든 학습 run의 cfg**.

```
#   Model/Method               mmlu      gsm8k     humaneval tydiqa    bbh       Best AVG  Best run                                Params (best avg run의 cfg.json)
=== ========================== ========= ========= ========= ========= ========= ========= ======================================= ================================================================
 1  llama2 / full_100          -         -         -         -         -         -         -                                       (no DONE yet)
 ... (16행)
```

> Best 표는 (7) Latest와 동일 컬럼 폭이 아니므로 헤더가 살짝 다름 — "Best AVG" + "Best run" 컬럼 추가. 80개 셀의 best는 셀 × 벤치 단위 (행 단위 best가 아님)이므로 한 행의 mmlu/gsm8k/humaneval/tydiqa/bbh 5점이 서로 다른 run에서 나올 수 있음. Params 컬럼은 그 셀의 **avg 최고 run** 기준으로 1개만 표기.

#### (9) History — append-only 변경 로그

> Current / Latest / Best 어느 표든 셀 값이 바뀌면 그 변경을 **(9) History 섹션에 한 줄로 append**한다 (newest at bottom). 사용자가 "이 셀 점수가 언제 어떻게 움직였나"를 시계열로 따라가는 view.

```
## (9) History (append-only — newest at bottom)

# 형식: YYYY-MM-DD HH:MM  cell=<model>/<method>  bench=<bench>  prev=<old>  new=<new>  reason=<태그>  source=<file>
2026-05-16 18:30  cell=llama2/full_100      bench=mmlu       prev=-       new=46.87%   reason=initial      source=runs/20260516_180000/eval/.../llama2_full_100-mmlu.json
2026-05-16 18:32  cell=llama2/full_100      bench=gsm8k      prev=-       new=14.32%   reason=initial      source=runs/20260516_180000/eval/.../llama2_full_100-gsm8k.json
2026-05-17 09:15  cell=llama2/random_10     bench=mmlu       prev=-       new=47.14%   reason=initial      source=runs/20260517_091500/eval/.../llama2_random_10-mmlu.json
2026-05-17 22:00  cell=llama2/tads_10       bench=mmlu       prev=47.20%  new=학습중   reason=재학습 시작  source=runs/20260517_220000/ (train launched)
2026-05-18 02:30  cell=llama2/tads_10       bench=mmlu       prev=학습중  new=48.05%   reason=재학습 완료  source=runs/20260517_220000/eval/.../llama2_tads_10-mmlu.json
...
```

규칙:
- newest at bottom (append-only). 절대 이전 줄을 수정/삭제하지 말 것.
- 셀 값이 (a) 상태 토큰 사이 전환 (예: `eval대기` → `eval중` → `NN.NN%`)이거나 (b) `NN.NN%` 값이 변경되거나 (c) `NN.NN%`가 `학습중`으로 돌아갈 때 모두 한 줄씩 추가.
- 컬럼 폭은 §0-4의 표 작성 규칙(`MD 파일 표 작성 규칙 6항`)을 따름. monospace 정렬, CJK는 visual width 2 cell로 패딩.
- `reason` 태그는 §0-6 (a)의 원인 태그(`lr↑`, `seed`, `재학습`, `cfg변경`, `param`, `initial`, `재학습 시작`, `재학습 완료` 등) 또는 사용자가 cell에 명시한 자유 텍스트.
- 매 tick 종료 시 그 tick에서 발생한 변경만 추가. 변경 없으면 줄 추가 안 함.
- per-cell 상세 시계열은 `${EVAL_RESULTS_ROOT}/<model>/<method>/HISTORY.md` (§0-6) — (9)는 그 요약 view.

#### (10) 재학습 ↔ 표 갱신 규약 + Params 한 줄 형식

**재학습 감지 시 표 흐름:**
1. 사용자가 어떤 셀에 새 학습 launch (예: `python -m tads.train --config configs/.../llama2/tads_10.yaml`)
   → `<ckpt_root>/runs/<new_tag>/` 생성 + `_latest` 포인터가 그쪽으로 이동.
2. 다음 tick에서 §5-4 분류기가 `학습 프로세스 alive` (pgrep) 신호를 잡음 → (6) Current 표의 그 셀 5개 컬럼 모두 `학습중`으로 전환.
3. (7) Latest 표는 **건드리지 않음** — 이전 DONE 값(예: `47.20%`)이 그대로 남아있음. 사용자는 "현재 학습중이지만 직전 마지막 점수는 47.20%였다"를 두 표를 동시에 보고 확인.
4. (8) Best 표도 건드리지 않음 — 새 학습이 끝나서 새 점수가 나올 때 비교.
5. (9) History에 `prev=47.20% new=학습중 reason=재학습 시작` 한 줄 append.
6. 학습 → eval 완료되어 새 NN.NN% 나오면:
   - (6) Current = 새 NN.NN%
   - (7) Latest = 새 NN.NN%로 갱신 (직전 47.20% 덮어씀, 단 (9) History엔 한 줄 더 append)
   - (8) Best = max(이전 best, 새 NN.NN%)로 갱신. 새 값이 best면 Best run 컬럼도 새 `<eval_tag>`로 갱신.
   - (9) History에 `prev=학습중 new=<new>% reason=재학습 완료` append.

**핵심 차이 요약:**

| 표 | 셀이 학습중일 때 | 새 NN.NN% 나오면 |
|---|---|---|
| (6) Current | `학습중` 표기 (덮어씀) | `NN.NN%`로 덮어씀 |
| (7) Latest  | **이전 NN.NN% 유지** | 새 NN.NN%로 덮어씀 |
| (8) Best    | **이전 Best 유지** | max(이전, 새)로 갱신 (낮으면 그대로) |
| (9) History | 한 줄 append      | 한 줄 더 append |

**Params 한 줄 형식 (rightmost column, ≤ 100자):**

해당 셀의 가장 최근 학습 run의 `cfg.json`에서 다음 키만 골라 1줄로 표기. 길어지면 `_50` 등 method 변형명을 따로 표시.

```
lr=<learning_rate> ep=<train_epochs> bs=<batch_size>/ga=<grad_accum> wd=<weight_decay>
ratio=<selection_ratio> wmup=<warmup_ratio> mode=<training_mode>
[attn=<attn_implementation>] [opt=AdamW(fp32|8bit)] [prompt=<prompt_style>]
[lam=<tads.lam>] [anchor=<anchor.layer_indices>] [ent=<agent.entropy_coef>]
```

값이 base.yaml 기본값과 같으면 생략 가능 (간결성). method-specific 키는 method가 `tads`/`data_agent`일 때만 표기.

예시:
- llama2/full_100: `lr=2e-5 ep=3 bs=8/ga=4 ratio=1.0 mode=full opt=AdamW8bit prompt=alpaca_default`
- llama2/tads_10:  `lr=2e-5 ep=3 bs=8/ga=4 ratio=0.1 mode=full opt=AdamW8bit prompt=alpaca_default lam=1.0 anchor=all ent=0.02`
- qwen25/random_10: `lr=2e-5 ep=3 bs=8/ga=4 ratio=0.1 mode=full opt=AdamW8bit`

**Params 컬럼 갱신 시점:**
- 새 학습 launch 시 (6) Current의 그 행 Params를 새 run의 cfg.json 기준으로 갱신.
- (7) Latest의 Params는 마지막 DONE이 나온 run의 cfg.json 기준 — 학습중이어도 안 바뀜.
- (8) Best의 Params는 best avg를 낸 run의 cfg.json 기준.

채우는 규칙 (§5-4의 4-state 분류를 그대로 반영) — **모든 표 ((1)-(6))에 동일한 5가지 표기만 사용**:

| 상태 | 셀 표기 | 판정 신호 (정확히 이렇게) |
|---|---|---|
| **NO-CKPT** | `학습전` | `_latest`도 `_latest.txt`도 legacy flat `epoch_*`도 없음. 행 전체 5컬럼 동일. 사용자에게 "학습이 필요한 셀 목록" 보고. |
| **TRAINING** | `학습중` | 학습 프로세스 alive **또는** sealed epoch < `cfg_target_epochs`. 행 전체 5컬럼 동일. |
| **NEED-EVAL** | `eval대기` | (a) 학습 끝남 (sealed == `cfg_target_epochs`) AND (b) `_latest/_complete` + `_latest/<label>-eval_summary.json` 부재 또는 stale AND **(c) `python -m tads.eval.*<model>/<method>` 프로세스 부재 AND (d) eval 로그 mtime > 5분 (= 진짜 idle)**. 다음 dispatch pass에서 에이전트가 자동 launch (§4 / §7 3.7). |
| **EVAL-RUNNING** | `eval중` | `python -m tads.eval.*<model>/<method>` 프로세스 alive **또는** eval 로그 mtime < 5분 이내 활동 (pgrep이 잠시 놓쳤어도 로그가 움직이면 살아있는 것). |
| **DONE** | `NN.NN%` 예: `47.56%` | §5-3 의 3-조건 만족. 정확도(%) 소수점 둘째자리 + `%`. **humaneval은 pass@10** (NAIT 논문 기준, n=20 sampled completions, T=0.8, p=0.95). 점수 JSON엔 `pass@1`도 같이 기록되지만 점수표는 `pass@10`만 본다. |

> **분류 결정 트리 — 매 tick 반드시 위에서 아래로 순회. 어느 분기에도 안 걸린 셀은 마지막에 `eval대기`로 떨어뜨림** (절대 `-` / `err` / 공란 금지):
>
> ```
> 1. ckpt 측 _latest 부재 + flat epoch_* 부재
>    → 학습전 [END]
>
> 2. 학습 프로세스 alive (pgrep "python.*tads.train.*<model>/<method>\.yaml")
>    OR sealed_epoch_count < cfg_target_epochs
>    (※ cfg_target_epochs = `<latest_run>/cfg.json`의 train_epochs.
>     라이브 YAML이 아님 — 학습 launch 시점 snapshot. §5-4 참조.)
>    → 학습중 [END]
>
> 3. eval 프로세스 alive (pgrep "python.*tads.eval.*<model>/<method>\.yaml")
>    OR eval 로그 mtime < 5분 이내 (pgrep 사이 잠시 안 잡힌 케이스 보호)
>    → eval중 [END]
>
> 4. eval _latest/_complete 존재
>    AND eval _latest/<exp_label>-eval_summary.json 존재
>    AND summary mtime > latest sealed epoch mtime
>    → NN.NN% (정확도 % 둘째 자리 + %) [END]
>
> 5. eval 로그 마지막 50줄에 OOM / CUDA error / RuntimeError / Killed / Traceback 발견
>    → eval대기 [END]  (.fail_count 증가, HISTORY.md `[eval]` 오류 엔트리 작성)
>
> 6. ← 위 5단계 모두 안 걸린 셀 = "학습 끝났는데 eval 결과 없는 셀"
>    → eval대기 [END]  (fall-through 기본값. 다음 dispatch pass에서 auto-launch)
> ```
>
> **`eval대기`로 떨어진 셀은 §7 의사 코드의 `3.7 dispatch pass`에서 빈 GPU 만큼 자동 launch된다** (§0 / §4 — 학습 동시 실행 없는 운영 가정). 사용자 입력 불필요. launch 직후 §0-4 (6) Current 표에서 즉시 `eval대기` → `eval중`으로 갱신.
>
> 이전 tick의 분류 값을 캐싱하지 말 것. 에이전트가 매 tick dispatch한 잡은 #3 (pgrep 또는 로그 mtime)에서 잡혀 `eval중`, 끝나면 #4에서 잡혀 `NN.NN%` — 매 tick 처음부터 6단계 다시 돌려야 정확.
>
> **재학습 감지** (§0-4 (10)): 셀이 이전 tick에 DONE(`NN.NN%`)이었어도, 현재 tick에서 위 결정 트리 #2 (학습 프로세스 alive 또는 새 run의 sealed < cfg_target)가 만족되면 **즉시 `학습중`으로 되돌린다**. 이전 NN.NN%는 (7) Latest 표에 유지되고 (9) History에 prev=NN.NN% new=학습중 한 줄 append. (6) Current만 학습중으로 덮어씀. DONE → 학습중 → 새 NN.NN% 흐름이 매 tick 동기화됨.

> **표기는 위 상태 토큰만 허용** (5종 벤치마크와 혼동 금지). `-`는 본 가이드 문서 안의 placeholder일 뿐 `experiments.md`의 셀 값으로는 **절대 쓰지 말 것** — `experiments.md`를 최초 생성할 때부터 §5-4 분류를 돌려 상태 토큰 중 하나를 채운다. `…`, `학습필요`, `legacy(...)`, `실패` 같은 옛 표기는 모두 위 상태 토큰 중 하나로 매핑: 진행 중 → `eval중`, 옛 포맷 점수 → 그대로 `NN.NN%`(다음 tick에 새 포맷으로 덮어씀), 실패 → `eval대기`로 되돌리고 fail_count(§10-3)만 별도 카운터.
>
> 흔한 실수: 가이드 §0-4(6) 코드 블록의 `-`로 채워진 행을 그대로 `experiments.md`에 복사하고 끝내는 것. 이건 잘못된 출력이다. **복사 직후 같은 tick에서 분류해서 상태 토큰로 덮어써야 매 tick이 마무리된다.**

추가 규칙:
- treatment 행이 DONE이면 옆에 §0-2의 발산 알람을 인라인으로 붙일 것. 예: `tads_10  41.20% (RED < random_10)`
- `학습전` 셀은 score board 갱신 외에 **별도 섹션 "필요한 학습 목록"을 `experiments.md` 상단에 자동 추가**해서 사용자가 한눈에 보게 할 것. 형식:
  ```
  ## 사용자 액션 필요 — 아직 학습되지 않은 셀 (학습전)
  - llama2 / data_agent_10
  - llama2 / tads_10
  - mistral / random_10
  ...
  ```
- **`eval대기` 셀은 별도 섹션 "Auto-dispatch queue (eval대기)"으로 `experiments.md` 상단에 표시** (학습전 섹션 바로 아래). 사용자 액션은 필요 없음 — 에이전트가 다음 tick에 빈 GPU 만큼 자동 launch (§4 / §7 3.7). 사용자에겐 진행 상황 알림 용도. 형식:
  ```
  ## Auto-dispatch queue — eval 미실행 셀 (다음 tick에 자동 launch 예정)
  - llama2 / random_10        (epoch 3 sealed at 2026-05-16 18:00; eval 로그 없음)
  - qwen25 / data_agent_10    (epoch 3 sealed at 2026-05-16 17:30; 직전 eval 5h 전 idle)
  - mistral / tads_10         (epoch 3 sealed at 2026-05-16 16:15)
  ...
  현재 빈 GPU: 2개 (cuda:1, cuda:3) — 위 큐 중 앞 2개 launch 예정.
  ```
  각 줄에 ckpt sealed 시각 + 직전 eval 활동 시각 부기. 맨 아래에 현재 빈 GPU 수 + 다음 tick에 어느 셀까지 launch될지 명시.
- 옛 포맷에서 추출한 점수도 그대로 `NN.NN%`로 표기하되, 발산 알람은 다음 tick 재평가로 DONE 전환될 때까지 회색(`(provisional)`)으로 표기.

### 0-5. Status Dashboard — **5×16 한눈 보기 표 (의무 출력)**

매 tick 보고에 에이전트는 **다음 80-cell 표 한 개를 반드시 출력**한다. §0-4의
score board(`experiments.md`)는 점수 디테일을 위한 long-form, 이건 사용자가 한
번에 진행 상태를 파악하는 dashboard. 셀 값은 정확히 **상태 토큰 중 하나** (§0-4 채우는 규칙과 동일 어휘):

| 상태 표기 | 의미 | 분류 조건 |
|---|---|---|
| `학습전` | 체크포인트 없음 | `<ckpt_root>/_latest`도 `_latest.txt`도 없고, legacy flat `epoch_*`도 없음. 행(=한 셀) 전체 5개 컬럼이 모두 이 표기. |
| `학습중` | 학습 진행 중 | (a) `python -m tads.train.*<model>/<method>` 프로세스가 살아있음, **또는** (b) `<latest_run>/cfg.json`의 `train_epochs` 값보다 sealed (`_complete`) epoch 수가 적음. 행 전체 5개 컬럼 모두 이 표기. |
| `eval대기` | 학습 끝남, eval 프로세스도 아직 안 떴음 | (a) sealed_n >= `cfg_target_epochs` AND (b) `<eval_base>/_latest/<exp_label>-<bench>.json` 없음 또는 mtime ≤ sealed epoch AND (c) `python -m tads.eval.*<model>/<method>` 프로세스 부재 AND (d) eval 로그 mtime ≥ 5분 (= 진짜 idle). **다음 §7 3.7 dispatch pass에서 에이전트가 자동 launch.** |
| `eval중` | eval 프로세스 떠있거나 log 활동 중 | (a) `python -m tads.eval.*<model>/<method>` 프로세스 alive **또는** (b) `<eval_base>/_latest/logs/eval_*.log` mtime < 5분 이내 (pgrep이 잠시 놓친 케이스 포함). |
| `47.56%` | 점수 산출 완료 | `<eval_base>/_latest/<exp_label>-<bench>.json`이 존재하고 mtime > latest sealed epoch. 점수는 §5-5의 정규화 규칙으로 추출, **소수점 둘째 자리 + `%`**. |

**판정 의사 코드** (cell-by-cell, §5-4 의 4-state 분류를 5×16에 투영):

```python
def status_cell(model, method, bench):
    ckpt_root = f"{OUTPUT_ROOT}/main_7b/{model}/{method}"
    latest_run = resolve_latest_run(ckpt_root)
    if latest_run is None:
        return "학습전"
    # 학습 프로세스 검출 (행 단위 - 5개 벤치 모두 학습중)
    if pgrep_alive(f"python.*-m tads.train.*{model}/{method}\\.yaml"):
        return "학습중"
    cfg = json.load(open(f"{latest_run}/cfg.json"))
    target = int(cfg.get("train_epochs", 3))
    # epoch_last/ 레이아웃 우선 — _complete sentinel 내용이 마지막 sealed
    # epoch 번호. 없으면 legacy epoch_<N>/ glob fallback.
    last_dir = f"{latest_run}/epoch_last"
    if exists(f"{last_dir}/_complete"):
        sealed_n = int(open(f"{last_dir}/_complete").read().strip() or 0)
        sealed_max = last_dir
    else:
        legacy = [p for p in glob(f"{latest_run}/epoch_*")
                  if basename(p) != "epoch_last" and exists(f"{p}/_complete")]
        if legacy:
            sealed_max = max(legacy, key=lambda p: int(basename(p).replace("epoch_", "")))
            sealed_n = int(basename(sealed_max).replace("epoch_", ""))
        else:
            sealed_n = 0
            sealed_max = None
    if sealed_n < target:
        return "학습중"
    # 학습 완료. 이 벤치 평가 상태 확인 — eval 측 _latest를 거쳐 본다.
    eval_base = f"{EVAL_RESULTS_ROOT}/{model}/{method}"
    eval_latest = resolve_latest_run(eval_base)
    label   = f"{model}_{method}"
    if eval_latest is not None and exists(f"{eval_latest}/_complete"):
        bench_json = f"{eval_latest}/{label}-{bench}.json"
        if exists(bench_json) and mtime(bench_json) > mtime(sealed_max):
            return f"{extract_score_pct(bench_json):.2f}%"
    # 점수 JSON 아직 없음 또는 _latest unsealed — pgrep + log mtime으로
    # eval중 vs eval대기 구분 (§5-4와 동일 룰).
    if pgrep_alive(f"python.*-m tads.eval.*{model}/{method}"):
        return "eval중"
    # pgrep이 잠시 놓친 케이스: 로그 mtime이 5분 이내면 살아있는 잡.
    eval_log = newest(f"{eval_base}/_latest/logs/eval_*.log")
    if eval_log and (now() - mtime(eval_log)) < 300:
        return "eval중"
    return "eval대기"
```

#### 80-cell 표 (16 행 × 5 벤치 컬럼) — 초기 상태 / 갱신 템플릿

행 순서는 `(model, method)` 묶음, §0-3 16-cell 표와 동일.

**experiments.md에 쓸 형식 — 반드시 아래처럼 코드 펜스 + 고정폭 정렬** (§0-4 표 작성 규칙 적용. 컬럼 폭: `#`=3, `Model/Method`=26, 각 벤치 상태 컬럼=9 — `eval대기`(visual 8) + 여백 1).

```
#   Model/Method               MMLU      GSM8K     HumanEval TyDiQA    BBH
=== ========================== ========= ========= ========= ========= =========
 1  llama2 / full_100          학습전    학습전    학습전    학습전    학습전
 2  llama2 / random_10         학습전    학습전    학습전    학습전    학습전
 3  llama2 / data_agent_10     학습전    학습전    학습전    학습전    학습전
 4  llama2 / tads_10           학습전    학습전    학습전    학습전    학습전

 5  qwen25 / full_100          학습전    학습전    학습전    학습전    학습전
 6  qwen25 / random_10         학습전    학습전    학습전    학습전    학습전
 7  qwen25 / data_agent_10     학습전    학습전    학습전    학습전    학습전
 8  qwen25 / tads_10           학습전    학습전    학습전    학습전    학습전

 9  mistral / full_100         학습전    학습전    학습전    학습전    학습전
10  mistral / random_10        학습전    학습전    학습전    학습전    학습전
11  mistral / data_agent_10    학습전    학습전    학습전    학습전    학습전
12  mistral / tads_10          학습전    학습전    학습전    학습전    학습전

13  deepseek / full_100        학습전    학습전    학습전    학습전    학습전
14  deepseek / random_10       학습전    학습전    학습전    학습전    학습전
15  deepseek / data_agent_10   학습전    학습전    학습전    학습전    학습전
16  deepseek / tads_10         학습전    학습전    학습전    학습전    학습전
```

> `**tads_10**` 같은 markdown bold는 experiments.md에는 쓰지 않는다 (terminal에서 `**` 별표가 그대로 보임). 강조가 필요하면 별도 ASCII 마커(`<` `>` 같은) 사용하거나 그냥 평문.

#### 진행 예시 (이런 형태로 보고 + 그대로 experiments.md에 들어가는 모양)

학습이 일부 진행되고 평가도 시작된 시점의 예시. 셀 값이 섞여도 컬럼 boundary는 그대로 유지(좌측 정렬, 우측에 공백 패딩):

```
#   Model/Method               MMLU      GSM8K     HumanEval TyDiQA    BBH
=== ========================== ========= ========= ========= ========= =========
 1  llama2 / full_100          47.56%    14.63%    27.87%    39.48%    39.94%
 2  llama2 / random_10         47.14%    14.13%    eval중    eval대기  eval대기
 3  llama2 / data_agent_10     학습중    학습중    학습중    학습중    학습중
 4  llama2 / tads_10           학습중    학습중    학습중    학습중    학습중

 5  qwen25 / full_100          eval중    eval대기  eval대기  eval대기  eval대기
 ...
16  deepseek / tads_10         학습전    학습전    학습전    학습전    학습전
```

해석 가이드:
- 행 전체가 `학습전`/`학습중`이면 학습 단계 → **에이전트는 자동 트리거 금지**, 사용자에게만 보고.
- 행에 `eval대기`가 섞여 있으면 §4-3 dispatch 큐에 자동 enqueue → 빈 GPU 생기는 대로 launch (launch 직후 `eval대기` → `eval중`).
- 행 전체가 `NN.NN%`이면 DONE → §0-4 score board에 점수 반영 + §0-2 발산 알람 평가.
- 한 행 내에서 `47.56%`와 `eval중`/`eval대기`가 섞이는 건 정상 (eval은 5개 벤치 순차 처리, JSON 떨어진 순으로 셀이 갱신됨).

#### 갱신 빈도 / 출력 위치

- 매 tick (cron 15분 주기)마다 dispatch pass 직전 1회, dispatch 직후 1회 = tick당 2회 출력.
- 출력은 `experiments.md` **상단**에 §0-4의 "표 작성 규칙"을 따른 **코드 펜스(```) + 고정폭 정렬** 표로 갱신 (markdown pipe table 금지 — terminal `cat`에서 깨짐). 전체 파일을 매번 다시 쓰지 말고 `<!-- STATUS DASHBOARD START -->` ... `<!-- STATUS DASHBOARD END -->` 마커 사이만 atomic 교체. 마커 자체는 markdown 렌더러에선 안 보이고 terminal에선 한 줄로 보여 위치 식별에 도움.
- 채팅/슬랙 보고에는 변경된 셀만 `diff` 형태로 인용 ("[#4 llama2/tads_10] 학습중 → eval대기", "[#1 llama2/full_100, MMLU] eval중 → 47.56%" 등). 표 전체 매번 복붙 금지.

### 0-6. History Tracking Guide — Status 컬럼 + per-cell `HISTORY.md`

§0-4(6) 80-cell 표의 맨 우측 `Status` 컬럼이 잡아야 하는 정보는 3종:

1. **history** — 이 셀의 점수가 시간에 따라 어떻게 움직였나 (튜닝, 재학습으로 인한 +/− delta)
2. **시스템 오류/상황** — eval 실행 중 OOM, crash, timeout, dataset 다운로드 실패 등
3. **baseline 발산** — §0-2 규칙 기반 RED/YELLOW/BLUE 알람 (BASE-FULL 대비 너무 낮음, BASE-NAIT보다 낮음 등)

전체 시계열은 `Status` 한 줄에 담을 수 없으므로 **2-tier 저장**:
- `Status` 컬럼 (표 안) — **최근 이벤트 1개 + 현재 발산 알람**의 한 줄 요약 (≤ 60자, 넘으면 `...` 잘라쓰기)
- `${EVAL_RESULTS_ROOT}/<model>/<method>/HISTORY.md` — append-only 시계열 로그 (최신이 위)

#### (a) Status 컬럼 표기 규칙 — 3 facet 조합 의무

**Status 한 줄은 정확히 3개의 facet으로 구성** (≤ 60자, 넘으면 우측에서 `...`로 잘라씀):

```
[facet1: history]  ·  [facet2: 오류/상황]  ·  [facet3: 발산 알람]
```

- 구분자 = ` · ` (양옆에 공백 1칸씩 + middle dot U+00B7).
- 어느 facet에 정보가 없어도 **그 자리에 placeholder를 채워서 3-facet 구조 유지**. 그냥 facet을 생략하지 말 것 — agent가 자주 facet 1만 적고 끝내는 패턴이 발생해 이번 강화.
- placeholder는 facet별로 정해진 값을 씀 (아래 표 참조).

| facet 위치 | 정상값 예시 | 정보 없을 때 placeholder |
|---|---|---|
| **facet1 (history)** | `init 47.56%` / `46.98% ↑ +2.20 (lr↑)` / `44.78% ↓ -2.78 (seed)` / `47.56% stable ×5` | `(no DONE yet)` — 아직 점수 산출 없음 |
| **facet2 (오류/상황)** | `OOM ×2 retry pending` / `crash(CUDA) ×1` / `dataset DL fail ×3 BLOCKED` | `no err` — 직전 5 tick 동안 오류 없음 |
| **facet3 (발산 알람)** | `🔴 < random_10` / `🟡 -5.2p vs full_100` / `🔵 < data_agent_10` / `OK` | `baseline` (full_100/random_10 행) 또는 `no score` (아직 점수 없어 비교 불가) |

**facet별 상세 토큰:**

| facet1 (history) 토큰 | 의미 |
|---|---|
| `init NN.NN%` | 첫 eval 결과 |
| `NN.NN% ↑ +Δ (태그)` | 직전 대비 점수 상승 (`Δ` = 둘째 자리, 태그는 원인) |
| `NN.NN% ↓ -Δ (태그)` | 점수 하락 |
| `NN.NN% stable ×K` | K회 연속 동일 점수 |
| `(no DONE yet)` | placeholder — 점수 한 번도 안 나옴 |

| facet2 (오류) 토큰 | 의미 |
|---|---|
| `OOM ×N retry pending` | N회 OOM, 자동 재시도 큐 (§10-3) |
| `crash(<kind>) ×N` | kind ∈ {`CUDA`, `Killed`, `Traceback`, `Timeout`}; eval/train 비정상 종료 |
| `dataset DL fail ×N BLOCKED` | 3회 연속 dataset 다운로드 실패, 재시도 중단 |
| `HANG <sec>s` | log mtime 30분+ 정체 (§0-7 (d)) |
| `(slow <sec>s)` | log mtime 5~30분 정체 (의심만, 알람 X) |
| `no err` | placeholder — 직전 5 tick 동안 오류 0 |

| facet3 (발산 알람) 토큰 | 의미 |
|---|---|
| `🔴 < random_10` | RED — BASE-NAIT보다 낮음 (§0-2) |
| `🟡 -X.Xp vs full_100` | YELLOW — BASE-FULL 대비 5%p 이상 처짐 |
| `🔵 < data_agent_10` | BLUE — 경쟁 method가 더 잘함 |
| `OK` | 모든 비교 정상 (treatment 행 점수 ≥ random_10 AND BASE-FULL 대비 ≤ 5%p) |
| `baseline` | placeholder — 이 행 자체가 baseline (full_100 / random_10) |
| `no score` | placeholder — 아직 점수 없어 비교 불가 |

**조합 예시 (모두 정확히 3-facet, ` · ` 구분):**

```
init 47.56%             · no err                · OK              ← treatment, 첫 eval, 정상
46.98% ↑ +2.20 (lr↑)    · no err                · 🟡 -5.2p vs full_100  ← lr 튜닝 상승, BASE-FULL 대비 처짐
47.56% stable ×3        · no err                · 🔴 < random_10  ← 점수는 안정이지만 RED
(no DONE yet)           · OOM ×2 retry pending  · no score        ← 아직 점수 없음, OOM 진행 중
47.56%                  · no err                · baseline        ← full_100 행
(no DONE yet)           · no err                · baseline        ← random_10 행, 아직 점수 없음
```

**잘못된 예시 (agent가 자주 범하는 위반):**

```
❌ "init 47.56%"                      ← facet 2, 3 누락
❌ "47.56% · OK"                      ← facet 2 누락 (자리에 `no err`라도 넣어야 함)
❌ "init 47.56% OK"                   ← 구분자 ` · ` 없음
❌ "OK"                               ← facet 1, 2 누락
❌ "init 47.56% / no err / OK"        ← 구분자가 ` / ` (반드시 ` · ` middle dot)
```

**원인 태그 (괄호 안, facet 1의 history 변동 사유):**

- `lr↑` / `lr↓` — 학습률 변경
- `seed` — 시드 변경
- `재학습` — 같은 cfg로 재학습
- `cfg변경` — yaml 변경 (자세한 diff는 HISTORY.md)
- `param` — 기타 파라미터 변경

**셀프 체크 (Status 컬럼 갱신 후 반드시):**

```bash
# (6) Current 표 Status 컬럼 추출 후 검증:
#   1. 각 행 Status 셀에 ` · ` 가 정확히 2번 등장 (= 3-facet)
#   2. 어느 facet도 빈 문자열이 아님 (placeholder는 OK)
awk '/^## \(6\) Current/,/^```$/' experiments.md \
  | awk '/^[ ]*[0-9]+ /{
       n=split($0, _, " · ");
       if (n != 3) { print "[STATUS-FAIL] row " NR " facet count=" n; exit 1 }
     }'
```

facet count ≠ 3 → atomic 교체 중단, Status 컬럼을 위 placeholder 규칙대로 다시 채운 후 재시도.

#### (b) per-cell `HISTORY.md` 포맷

위치: `${EVAL_RESULTS_ROOT}/<model>/<method>/HISTORY.md`
- append-only markdown, **최신이 위** (역시간순)
- 새 이벤트마다 한 섹션 추가, atomic write (`.tmp` → `mv`)

템플릿:
```
# History — <model> / <method>
(experiment_label: <model>_<method>)

## 2026-05-18 14:32  [eval]  OOM at MMLU loader
- GPU: 3, batch_size: 4, model: 7B fp16
- last successful eval: 2026-05-17 (47.56%)
- log tail: ${EVAL_RESULTS_ROOT}/<model>/<method>/logs/mmlu-2026-05-18.log:120-145
- action: retry queued for next tick (fail_count=2)

## 2026-05-17 09:15  [eval]  done — 47.56% (avg of 5 bench)
- mmlu 47.56, gsm8k 14.63, humaneval 27.87, tydiqa 39.48, bbh 39.94
- delta vs prev avg (44.78%): +2.78
- 발산: 🟡 -5.20p vs full_100 (52.76%)
- summary: ${EVAL_RESULTS_ROOT}/<model>/<method>/_latest/<label>-eval_summary.json
- eval run dir: ${EVAL_RESULTS_ROOT}/<model>/<method>/runs/<eval_tag>/  (snapshot)

## 2026-05-16 22:00  [param]  lr 5e-5 → 1e-4, warmup 200 → 500
- cfg: configs/experiments/main_7b/llama2/tads_10.yaml
- git: <commit hash>
- 이유: 직전 평균 44.78%가 BASE-FULL 대비 -8%p (YELLOW)

## 2026-05-16 18:00  [train]  학습 완료 (3 epochs)
- ckpt: <ckpt_root>/_latest/epoch_last  (sealed epoch=3, train_epochs=3 → DONE)
- duration: 4h 12m
- cfg snapshot: <ckpt_root>/_latest/cfg.json
```

이벤트 태그 종류 (대괄호):
- `[train]` — 학습 완료 / 실패
- `[eval]` — eval 완료 / 실패 (OOM, crash, timeout)
- `[param]` — 파라미터 변경 (cfg diff 첨부)
- `[note]` — 사용자/에이전트 메모

#### (c) 시스템 오류 (OOM 등) 기록 의무

eval이 0 아닌 exit code로 종료되면 에이전트는 **반드시**:

1. `Status` 컬럼 → `<오류종류> ×N retry pending · last <직전 점수>` 갱신
2. `HISTORY.md`에 `[eval] <오류종류>` 엔트리 append:
   - 오류 종류 (OOM / crash / timeout / dataset / unknown)
   - GPU 번호, 모델 크기, batch_size
   - log tail (마지막 ~25줄 또는 stack trace)
   - 직전 성공 eval 점수와 날짜
3. `${EVAL_RESULTS_ROOT}/<model>/<method>/.fail_count` 1 증가 (없으면 생성)
4. **3회 연속 동일 셀 실패** 시:
   - `Status` 컬럼에 `BLOCKED ×3` 표기
   - 자동 retry 중단 (다음 tick에 큐잉 안 함)
   - §10 cleanup 트리거 + 사용자에게 채팅으로 알림
   - 사용자가 `${EVAL_RESULTS_ROOT}/<model>/<method>/.fail_count`를 직접 reset해야 재시도 재개
5. 성공한 eval이 한 번이라도 발생하면 `.fail_count` = 0으로 리셋

OOM 특화 처리:
- `Status`에 `OOM` 명시 → 사용자가 메모리 튜닝 후보 (batch_size↓, grad accumulation↑) 즉시 인지
- HISTORY.md에 GPU 번호 + 다른 잡 점유 상태 기록 (메모리 경합 가능성 진단용)

#### (d) 발산 알람 (3번 facet) 갱신 시점

- 점수 셀이 `eval중`/`eval대기` → `NN.NN%`로 전환되는 매 tick
- 해당 행이 treatment (data_agent_10 / tads_10)일 때만 평가. baseline 행은 알람 없음.
- 비교 대상 셀(예: tads_10이면 같은 모델의 random_10 / data_agent_10 / full_100)이 아직 `NN.NN%`가 아니면 알람 보류 (`pending baseline`).
- 알람 규칙은 §0-2 그대로:
  - `🔴 RED` — `tads_10 < random_10`
  - `🟡 YELLOW` — `tads_10 < full_100 − 5%p`
  - `🔵 BLUE` — `data_agent_10 > tads_10 + 1%p`
  - 모두 통과 → `OK`

#### (e) Status 업데이트 트리거 (요약)

| 트리거 | Status 갱신 |
|---|---|
| 첫 eval JSON 등장 | `init NN.NN%` |
| 후속 eval JSON 등장 (점수 변화) | `<new>% ↑/↓ ±Δ (원인태그)` |
| eval 프로세스 비정상 종료 | `<오류> ×N retry pending` |
| `.fail_count` ≥ 3 | `BLOCKED ×N` |
| baseline 행 점수 갱신 → 발산 재계산 | 해당 모델의 모든 treatment 행 facet3 재평가 |
| 학습 새로 끝남 (sealed epoch 증가) | `learn complete <date> · eval큐` |

원인 태그는 `HISTORY.md`의 가장 최근 `[param]` 또는 `[train]` 엔트리에서 유추 (없으면 `재학습`).

### 0-7. Log-Tail Polling — 로컬 로그 파일을 매 tick 읽어서 상태 최신화

> **⚠️ 로그 polling은 분류의 입력 신호일 뿐이다. deliverable이 아님.** 이 섹션을 다 읽고 "agent는 log 추출기"라고 결론 내리지 말 것 — 매 tick의 핵심 deliverable은 §7 의 **3.7 dispatch pass에서 빈 GPU 만큼 `python -m tads.eval`을 launch**하는 것. log는 dispatch 가능 여부 + 상태 확인용 input. PRE-FLIGHT CHECK 15 참조.

**원칙**: 파일 시스템 state(체크포인트 sealed 여부, summary JSON 존재)만으로는 "현재 학습이 잘 돌고 있는지 / eval이 어디까지 갔는지 / 방금 OOM 났는지" 알 수 없다. 에이전트는 **매 tick마다 로컬 로그 파일들을 tail해서** 그 정보를 §0-5 dashboard / §0-6 HISTORY.md에 반영해야 한다.

#### (a) 로그가 사는 위치 (모두 polling 대상)

| 로그 종류 | 경로 | 만들어지는 곳 |
|---|---|---|
| **학습 (script 기준, append)** | `logs/main_7b_<model>_<method>.log` | `scripts/run_main_7b.sh` (`tee -a`) |
| **학습 (per-run)** | `${OUTPUT_ROOT}/main_7b/<model>/<method>/runs/<tag>/logs/train_<method>_<ts>_r<rank>.log` | `tads/train.py` (DDP rank마다 1개) |
| **학습 (per-run, _latest 경유)** | `${OUTPUT_ROOT}/main_7b/<model>/<method>/_latest/logs/` | 위와 같은 파일, _latest로 접근 |
| **평가 (script 기준, append)** | `logs/eval_main_7b_<model>_<method>.log` | `scripts/run_eval_main_7b.sh` (`tee -a`) |
| **평가 (per-run, _latest 경유)** | `${EVAL_RESULTS_ROOT}/<model>/<method>/_latest/logs/eval_<ts>_r0.log` | `tads/eval.py` (history layout) |
| **평가 (per-run, snapshot)** | `${EVAL_RESULTS_ROOT}/<model>/<method>/runs/<eval_tag>/logs/` | 위와 같은 파일 (과거 run audit용) |
| **cron tick 로그** | `${EVAL_RESULTS_ROOT}/.auto_eval_logs/tick_*.log` | §9-3 의 tick 스크립트가 만들 것 |

> script-side 로그 (`logs/main_7b_*`, `logs/eval_main_7b_*`)는 **계속 append**되어 한 셀에서 여러 run을 거치면 누적된다. 최신 run만 보려면 mtime 가장 최근의 `runs/<tag>/logs/...`를 보는 게 정확. 단, 사용자가 종종 `tail -f logs/main_7b_llama2_tads_10.log`로 모니터링하므로 둘 다 살아있는 게 정상.

#### (b) 매 tick 로그에서 뽑아야 하는 신호 (필수)

각 (model, method) 셀의 가장 최근 train 로그 + 가장 최근 eval 로그에 대해, 아래 7가지 signal을 추출:

```bash
# 셀의 최신 학습 로그 1개 (latest run의 rank 0)
TRAIN_LOG=$(ls -t "${OUTPUT_ROOT}/main_7b/${model}/${method}/_latest/logs/train_"*_r0.log 2>/dev/null | head -1)
# 셀의 최신 평가 로그 1개
EVAL_LOG=$(ls -t "${EVAL_RESULTS_ROOT}/${model}/${method}/_latest/logs/eval_"*.log 2>/dev/null | head -1)
```

| Signal | grep / tail 패턴 | 사용처 |
|---|---|---|
| **1. SFT 진행** | `grep -E "SFT \| epoch=[0-9]+ \| step=" "$TRAIN_LOG" \| tail -1` | dashboard Status (`step=N/T loss=X`) |
| **2. PCA / anchor 진행** | `grep -E "TrajectoryAnchor.update\|collect_episode" "$TRAIN_LOG" \| tail -3` | tads/data_agent의 selection phase 정체 감지 |
| **3. epoch sealed 마커** | `grep -E "Checkpoint saved \+ sealed\|_latest -> runs/" "$TRAIN_LOG" \| tail -2` | "학습 N/M epoch 끝남" 표기 |
| **4. eval 벤치 진행** | `grep -E "Progress:\|Eval start\|Benchmark .* failed" "$EVAL_LOG" \| tail -5` | dashboard "eval중" 셀의 현재 bench |
| **5. eval 완료 마커** | `grep -E "Eval run sealed:\|_latest -> runs/.* under " "$EVAL_LOG" \| tail -2` | 신규 _latest 갱신 즉시 감지 (mtime 동기화 전에도) |
| **6. 오류 / 경고** | `grep -iE "(out of memory\|OOM\|CUDA error\|RuntimeError\|Killed\|Traceback)" "$LOG" \| tail -10` | §0-6 (c) HISTORY.md `[eval]` 또는 `[train]` 오류 엔트리 |
| **7. 마지막 활동 시점** | `stat -c %Y "$LOG"` (mtime) | hang 감지 (아래 (d) 참조) |

전부 한 셀당 ~수십 KB grep — 매 tick 16 셀 × (train + eval) = 32개 grep, 1초 이내.

#### (c) Status 컬럼에 반영하는 패턴

§0-6 (a)에서 정의한 Status facet들을 **이 신호들로 갱신**:

| 신호 추출 결과 | Status 표기 |
|---|---|
| Signal 1 매치 + 학습 프로세스 alive → `SFT \| epoch=2 \| step=350/2031 \| loss=1.84` | dashboard cell `학습중`, Status에 `epoch 2 step=350/2031 loss=1.84` |
| Signal 4 매치 → `Progress: 60/164 (n_samples=20, chunks=5×4)` | dashboard cell `eval중`, Status에 `humaneval 60/164` |
| Signal 6 매치 (예: OOM) | Status 즉시 `OOM ×N retry pending · last <score>` 갱신 + §0-6 (c) HISTORY.md `[eval] OOM` 엔트리 작성 (log tail 25줄 포함) |
| Signal 5 매치 (방금 _latest sealed) | 점수 파싱을 다음 tick까지 기다리지 말고 즉시 §0-4 점수 표 + HISTORY.md `[eval] done` 갱신 (선제 동기화) |
| Signal 3 매치 (방금 sealed epoch 증가) | Status `learn complete <date> · eval큐` + 해당 셀을 NEED-EVAL 큐에 즉시 enqueue (다음 tick 안 기다림) |

#### (d) Hang 감지 — log mtime이 N분 이상 안 움직이는데 프로세스가 살아있음

특히 다음 두 케이스가 흔하다:
- 학습 collect_episode 진입(`[trace] rank=0 BEFORE collect_episode`) 이후 30분+ 무응답 → PCA 또는 DataLoader 워커 wedge.
- eval HumanEval 중 한 문제에서 generate가 ∞ loop.

**탐지 로직**:
```bash
# 학습이 5분 이상 로그를 안 쓰면 의심, 30분 이상이면 hang 확정
now=$(date +%s)
mtime=$(stat -c %Y "$TRAIN_LOG")
elapsed=$(( now - mtime ))
if pgrep -af "python.*tads.train.*${model}/${method}\.yaml" >/dev/null; then
  if [ "$elapsed" -ge 1800 ]; then
    # 30분+ idle → HANG. Status 컬럼에 표시, HISTORY.md에 기록.
    # 자동 kill 하지 말 것 — 사용자가 결정. 단 Status는 `HANG ${elapsed}s · last <signal>` 갱신.
    :
  elif [ "$elapsed" -ge 300 ]; then
    # 5~30분 → STALLED 의심. Status에 `(slow ${elapsed}s)` 표기만, 알람 NO.
    :
  fi
fi
```

자동 복구는 §10 cleanup 절차의 트리거 조건과 연동. **자동 kill / restart 금지** (학습 1잡 = 사용자 정책 영역). 에이전트는 보고만.

#### (e) 채팅 보고 시 로그 인용 규칙

사용자에게 매 tick 메시지를 보낼 때 로그 raw dump는 금지. 다음 형식만:

- 변경된 셀 1줄 요약: `[#4 llama2/tads_10] 학습중 → eval대기 (sealed epoch 4/4, ${OUTPUT_ROOT}/.../runs/20260516_120000/)`
- 오류 셀 1줄 요약 + log 경로: `[#7 qwen25/data_agent_10] OOM at eval mmlu (log: ${EVAL_RESULTS_ROOT}/qwen25/data_agent_10/_latest/logs/eval_20260516_180000_r0.log:142-167)`

전체 로그 인용은 HISTORY.md에만. 사용자 메시지는 핵심 정보 + 파일 경로 + line range만.

#### (f) tick 진입 시 로그 polling 순서 (의사 코드)

```python
def tick():
    for model, method in ALL_16_CELLS:
        train_log = newest("${OUTPUT_ROOT}/main_7b/" + model + "/" + method
                           + "/_latest/logs/train_*_r0.log")
        eval_log  = newest("${EVAL_RESULTS_ROOT}/" + model + "/" + method
                           + "/_latest/logs/eval_*.log")
        # script-side append 로그는 보조 (per-run 로그가 source of truth)
        train_log_aux = "logs/main_7b_" + model + "_" + method + ".log"
        eval_log_aux  = "logs/eval_main_7b_" + model + "_" + method + ".log"

        # (b) Signal 1~7 추출. train_log가 없으면 train_log_aux, eval도 마찬가지.
        signals = extract_signals(train_log or train_log_aux,
                                  eval_log  or eval_log_aux)

        # (d) hang 체크 — 가장 최신의 살아있는 로그 기준
        if signals.train_running and signals.train_mtime_age > 1800:
            mark_hang(model, method, signals)

        # (c) Status / HISTORY.md / 점수표 동기화
        update_status_column(model, method, signals)
        if signals.new_oom_or_error:
            append_history_md(model, method, signals)
        if signals.new_sealed_eval:
            sync_score_board_now(model, method)  # 다음 tick 안 기다림
```

§7 의사 코드와 §9-3 tick 스크립트는 위 polling pass를 **classify pass보다 먼저** 실행해야 한다 (signals이 classify의 입력이기 때문). 다음 §7 / §9-3 갱신본 참조.

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
│   │   ├── epoch_last/                 ← 단일 ckpt dir, 매 epoch마다 덮어씀
│   │   │   ├── _complete               ← 마지막 sealed epoch 번호 (int) 기록
│   │   │   ├── env_meta.json           ← torch / bitsandbytes 버전
│   │   │   ├── optimizer.pt, scheduler.pt
│   │   │   └── (model.safetensors 등 HF 표준)
│   │   ├── metrics.json                ← 매 epoch row append (epoch 번호 포함)
│   │   └── selected_indices_epoch{N}.json
│   ├── 20260515_193000_lr5e5/          ← 같은 셀에 파라미터 튜닝 재학습
│   └── 20260516_080000_anchor_all/
└── _latest -> runs/20260516_080000_anchor_all/   ← 항상 최신 가리킴 (symlink)
```

- `<model>` ∈ {`llama2`, `qwen25`, `mistral`, `deepseek`}
- `<method>` ∈ {`full_100`, `random_10`, `data_agent_10`, `tads_10`} (또는 `_50` 변형)
- **epoch_last/ 레이아웃 (2026-05-16~)**: 학습은 매 epoch마다 단일 `epoch_last/` 디렉터리를 atomic하게 덮어쓴다. 디렉터리 이름엔 epoch 번호를 안 박는다 — 번호는 cfg.json snapshot의 `train_epochs`(target)와 `_complete` sentinel 내용(현재 마지막 sealed epoch, int)에서 읽는다.
- 매 sealed epoch 저장 후 `_latest` 포인터가 atomic하게 갱신됨 (학습 중 eval 가능).
- symlink 불가 FS는 `_latest.txt` (run_tag 한 줄)로 fallback.
- **eval은 항상 `_latest/epoch_last` 기준** — 사용자가 새 튜닝 잡을 띄우면 자동으로 새 결과가 평가됨.

대응 config: `configs/experiments/main_7b/<model>/<method>.yaml`

**최신 ckpt 경로 + 마지막 sealed epoch 찾는 정식 방법**:
```bash
ckpt_root="${OUTPUT_ROOT}/main_7b/<model>/<method>"
latest_run="$(readlink -f "$ckpt_root/_latest" 2>/dev/null || \
              { [ -f "$ckpt_root/_latest.txt" ] && echo "$ckpt_root/runs/$(cat "$ckpt_root/_latest.txt")"; })"
ckpt_dir="${latest_run}/epoch_last"
[ -f "${ckpt_dir}/_complete" ] || { echo "WARN: ${ckpt_dir} is not sealed"; }
# 마지막 sealed epoch 번호 = _complete sentinel 내용 (int).
sealed_n="$(cat "${ckpt_dir}/_complete" 2>/dev/null || echo 0)"
# 학습 target epoch = cfg.json snapshot.
target_n="$(python3 -c "import json;print(json.load(open('${latest_run}/cfg.json')).get('train_epochs',3))")"
echo "sealed=${sealed_n} / target=${target_n}"
```

> Back-compat: 이전 학습 결과는 `epoch_1/`, `epoch_2/`, … 처럼 epoch 번호별 별도 dir이었다 (legacy). 새 코드는 `epoch_last/`가 있으면 우선 사용하고, 없으면 가장 큰 sealed `epoch_<N>/`로 fall-back한다 (`tads/core/run_layout.py:find_latest_complete_epoch`).

### 3-2. Legacy flat layout (`feature/run-layout` 이전 학습)

```
${OUTPUT_ROOT}/main_7b/<model>/<method>/epoch_<N>/   ← runs/ 와 _latest 없음
```

`feature/run-layout` 머지 이전에 학습된 셀은 평탄 구조로 떨어져 있다. eval 측은
`_latest`가 없으면 평탄 구조를 fallback으로 자동 인식 (§4-4 참고).

추가로 **epoch_last/ migration 이전 (~2026-05-15)에 학습된 run dir** 안에는 `epoch_1/`, `epoch_2/`, …처럼 epoch 번호별 별도 dir이 존재할 수 있다. 새 코드(`tads/core/run_layout.py:find_latest_complete_epoch`)는 `epoch_last/`를 우선 보고, 없으면 numeric `epoch_<N>/` glob fallback으로 자동 처리하므로 별도 마이그레이션 불필요. epoch 번호는 dir 이름(legacy) 또는 `_complete` sentinel 내용(new layout)에서 추출.

이전 체크포인트를 새 구조로 마이그레이션하려면 `runs/<tag>/`로 이동 후 `_latest`를 수동 설정:

```bash
mv ${OUTPUT_ROOT}/main_7b/llama2/tads_10/epoch_* ${OUTPUT_ROOT}/main_7b/llama2/tads_10/runs/legacy/
ln -s runs/legacy ${OUTPUT_ROOT}/main_7b/llama2/tads_10/_latest
```

### 3-3. Legacy 7b_fullft (더 옛날 구버전, 보통은 건드릴 일 없음)

```
${OUTPUT_ROOT}/7b_fullft/<run>/epoch_last/   ← 새 학습이면 epoch_last
${OUTPUT_ROOT}/7b_fullft/<run>/epoch_3/      ← 옛 학습이면 epoch_3 (legacy)
```

대응 config: `configs/experiments/7b_fullft_<run>.yaml`. `auto_eval_7b_fullft.sh`는 사용 금지(§6) — 평가가 필요하면 §4-1 1셀-1명령으로 직접 launch.

---

## 4. Eval 실행 — **에이전트 auto-launch, 1 GPU = 1 cell** (2026-05-16 재정의)

**원칙**:
- **에이전트가 `eval대기` 셀을 자동 launch한다.** 빈 GPU가 K개 있으면 큐에서 K개를 골라 동시에 launch — **GPU 1장당 cell 1개**, 1셀 = 1 `python -m tads.eval` 명령 = 1 백그라운드 프로세스.
- **batch wrapper(`run_eval_main_7b.sh`, `auto_eval_7b_fullft.sh`) 절대 사용 금지** — 사용자도, 에이전트도. wrapper는 한 호출로 여러 셀을 sequential/parallel dispatch하는데, 우리는 셀별로 별개 PID + 별개 log + 별개 dispatch line이 필요. wrapper의 cascade fail / log race / `_latest` symlink race 문제는 1셀-1명령 분해로만 해소됨.
- 학습은 사용자 영역. 에이전트는 `python -m tads.train` / `torchrun` / `run_main_7b.sh` 절대 실행 금지.
- 매 tick에 에이전트가 하는 일: (1) 32 log polling (§0-7), (2) 80셀 분류, (3) eval대기 큐 만들기, (4) 빈 GPU 만큼 launch, (5) 표 atomic 갱신, (6) 사용자 보고.

### 4-1. 정식 명령 형태 — 에이전트(자동, 1-bench 순차) vs 사용자(수동, BBH)

**에이전트 자동 dispatch — 단일 bench 하나씩만 (`mmlu` → `gsm8k` → `humaneval` → `tydiqa` 순서)**:

```bash
# 매 cron tick에서 빈 GPU 만큼 자동 실행. 각 launch는 단일 (cell, bench).
# ⚠️ 1 GPU = 1 (cell, bench). 4개를 한 명령에 묶지 말 것.
# bench는 dispatch 큐 순서대로: mmlu 끝나면 같은 cell의 gsm8k, 그 다음...
CUDA_VISIBLE_DEVICES=<free_gpu> nohup python -m tads.eval \
    --config configs/experiments/main_7b/<model>/<method>.yaml \
    --benchmarks <single_bench> \    # mmlu / gsm8k / humaneval / tydiqa 중 하나
    --out_dir ${EVAL_RESULTS_ROOT}/<model>/<method> \
    >> logs/eval_<model>_<method>_<single_bench>.log 2>&1 &
```

예: 4 개 GPU가 비어있고 4 cells에서 각각 mmlu가 다음 차례라면:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m tads.eval --config .../llama2/full_100.yaml      --benchmarks mmlu --out_dir ... &
CUDA_VISIBLE_DEVICES=1 nohup python -m tads.eval --config .../llama2/random_10.yaml     --benchmarks mmlu --out_dir ... &
CUDA_VISIBLE_DEVICES=2 nohup python -m tads.eval --config .../llama2/data_agent_10.yaml --benchmarks mmlu --out_dir ... &
CUDA_VISIBLE_DEVICES=3 nohup python -m tads.eval --config .../llama2/tads_10.yaml       --benchmarks mmlu --out_dir ... &
```
이 4개 mmlu가 다 끝나면 다음 tick에서 같은 4 cells에 대해 `--benchmarks gsm8k` 로 dispatch.

**사용자 수동 dispatch — BBH 전용** (2026-05-17 분리 정책. 셀당 ~15h+):

```bash
# 사용자가 시간 있을 때 직접 launch. 에이전트는 자동 실행 안 함.
# 결과 JSON(_latest/<label>-bbh.json)이 떨어지면 score-board가 자동 detect.
CUDA_VISIBLE_DEVICES=<gpu> nohup python -m tads.eval \
    --config configs/experiments/main_7b/<model>/<method>.yaml \
    --benchmarks bbh \
    --out_dir ${EVAL_RESULTS_ROOT}/<model>/<method> \
    >> logs/eval_<model>_<method>_bbh.log 2>&1 &
```

빈 GPU가 K개일 때 에이전트는 매 tick에서 **위 명령을 K번 따로 실행**한다 (각각 다른 `<free_gpu>` + 다른 큐의 (model, method)). 절대 wrapper 한 번 호출로 K개 잡을 한꺼번에 띄우지 말 것 — wrapper 호출은 다음과 같은 운영 문제를 만든다:

| Wrapper 사용 시 문제 | 1셀-1명령으로 우회되는 이유 |
|---|---|
| 한 잡 fail이 다른 잡까지 cascade | 잡마다 독립 PID → 한 잡 kill해도 나머지 영향 없음 |
| `tee -a logs/...log` 공유 → 로그 race | 셀마다 별개 log path |
| 같은 셀에 두 번 launch (큐 알고리즘 vs 사용자) | 에이전트가 dispatch 전 pgrep 체크로 중복 방지 |
| OOM / hang 진단 시 어느 잡인지 모호 | pgrep 패턴이 셀별로 명확하게 분리됨 |
| `_latest` symlink atomic 갱신이 여러 잡과 race | 한 셀당 한 잡이라 race 자체가 없음 |

**금지 명령:**
```bash
# ❌ 절대 금지 — wrapper 일괄 dispatch
bash scripts/run_eval_main_7b.sh ...
bash scripts/run_eval_main_7b.sh --parallel ...
bash scripts/auto_eval_7b_fullft.sh ...

# ❌ 절대 금지 — 한 명령으로 여러 셀 launch (for 루프 등)
for m in llama2 qwen25; do for x in tads_10 random_10; do
  python -m tads.eval --config configs/.../$m/$x.yaml ... &
done; done
```

**허용 패턴:**
```bash
# ✓ 셀 1개 launch → 결과 확인 → 그 다음 셀 launch
CUDA_VISIBLE_DEVICES=0 nohup python -m tads.eval \
    --config configs/experiments/main_7b/llama2/tads_10.yaml \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
    --out_dir ${EVAL_RESULTS_ROOT}/llama2/tads_10 \
    >> logs/eval_llama2_tads_10.log 2>&1 &
# (잡 끝나길 기다리거나 다른 빈 GPU 확인 후 다음 명령 입력)

# ✓ 다른 GPU에 다른 셀 — 각각 별개 명령
CUDA_VISIBLE_DEVICES=1 nohup python -m tads.eval \
    --config configs/experiments/main_7b/qwen25/data_agent_10.yaml \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
    --out_dir ${EVAL_RESULTS_ROOT}/qwen25/data_agent_10 \
    >> logs/eval_qwen25_data_agent_10.log 2>&1 &
```

요점:
- `--ckpt` 생략 → `<output_dir>/_latest`에서 자동 resolve (§3-1).
- `--out_dir` 명시 → 점수표 에이전트가 보는 표준 경로(`${EVAL_RESULTS_ROOT}/<model>/<method>/_latest/`)로 결과가 떨어짐. 생략하면 기본값이 `<ckpt>/eval/`이라 점수표가 못 찾음.
- `tads.eval`이 자동으로 `runs/<eval_tag>/` 생성 + `_complete` sentinel + `_latest` 갱신 (§5-1 history layout).
- 학습 중인 GPU는 피할 것 (OOM).

### 4-2. 에이전트가 매 tick 할 일 — polling → 분류 → dispatch → report

매 tick 시작 시 (자세한 의사 코드는 §7 / §9-3):
1. §0-7 (b) log-tail polling으로 각 셀의 train + eval 로그 상태 확인.
2. §5-4 표의 5가지 판정 신호로 셀별 상태 결정 — `eval대기` vs `eval중` 구분은 §4-3.
3. **빈 GPU 목록 ↔ `eval대기` 큐 매칭 (dispatch pass)**:
   - free_gpus = `nvidia-smi`에서 `memory.used < 2GB` AND `pgrep tads.train` 없는 GPU.
   - eval_queue = §5-4 분류에서 `eval대기`인 셀 목록 — 단, 이미 다른 GPU에서 같은 셀이 running(`pgrep tads.eval.*<model>/<method>\.yaml`)이면 큐에서 제외.
   - while free_gpus and eval_queue:
       * gpu = free_gpus.pop()
       * cell = eval_queue.pop_front()
       * §4-1 명령 형식으로 background launch (1 GPU = 1 cell = 1 명령).
       * `sleep 0.5` (CUDA init race 회피).
4. 결과를 `experiments.md` §0-4 점수표 4종(Current/Latest/Best/History) + §0-5 dashboard에 atomic 반영.
5. 새로 sealed된 eval(`_latest/_complete` + summary mtime > sealed epoch)은 즉시 `NN.NN%`로 전환.
6. `eval대기` 셀이 launch 가능했지만 빈 GPU가 모자라 큐에 남았으면 §0-4 "사용자 액션 필요" 섹션 대신 "다음 tick 자동 dispatch 예정" 섹션에 그 목록 표기. 사용자가 직접 띄울 필요 없음 (단, 사용자가 수동으로 띄워도 §4-3 1번에서 잡혀 중복 launch 안 됨).
7. 학습 중인 GPU는 절대 건드리지 말 것 — dispatch pass의 free_gpus 필터에서 `tads.train` 점유 GPU 제외.

### 4-3. eval대기 vs eval중 — 매 tick 재판정 (캐싱 금지)

§5-4 표의 판정 신호를 그대로 사용. 핵심 5단계 우선순위:

1. `pgrep -af "python -m tads.eval.*<model>/<method>\.yaml"` ≥ 1 → **`eval중`**.
2. pgrep 0개 + eval 로그 mtime < 5분 → **`eval중`** (grep 사이 잠시 안 잡힌 케이스 대비).
3. pgrep 0개 + 로그 mtime ≥ 5분 + 로그 마지막 50줄에 OOM/CUDA/Traceback **있음** → **`eval대기`** + `.fail_count` 증가 + HISTORY.md `[eval]` 오류 엔트리.
4. pgrep 0개 + 로그 mtime ≥ 5분 + 오류 없음 + sealed `_complete` 있고 summary mtime > sealed epoch → **`NN.NN%`** (DONE).
5. 위에 다 해당 안 되고 sealed_epoch < `cfg_target_epochs` (= 해당 run의 `cfg.json` snapshot 값, 라이브 YAML 아님)이면 → **`학습중`**, 그 외 학습 끝났는데 eval 결과 없음 → **`eval대기`**.

이전 tick의 분류 값을 그대로 들고 오지 말 것 — 매 tick에서 위 5단계를 처음부터 다시 돌릴 것. 사용자가 tick 사이에 eval을 launch하면 1번에서 잡혀야 표가 정확해진다.

### 4-4. 필터 / 제한 (사용자 수동 실행 시 참고)

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

### 5-1. 결과 저장 위치 (history-preserving, 2026-05-16~)

학습(train)과 동일한 `runs/<tag>/` + `_latest` 레이아웃. 셀의 BASE는

```
${EVAL_RESULTS_ROOT}/<model>/<method>/
```

(주의: `main_7b/` 접두어 **없음**. `OUTPUT_ROOT`와 레이아웃이 다름.)

그 안에:
```
runs/<eval_tag>/         ← eval_tag = YYYYMMDD_HHMMSS (또는 사용자가 --eval_tag로 지정)
  ├── <experiment_label>-mmlu.json
  ├── <experiment_label>-gsm8k.json
  ├── <experiment_label>-humaneval.json
  ├── <experiment_label>-tydiqa.json
  ├── <experiment_label>-bbh.json
  ├── <experiment_label>-eval_summary.json
  ├── logs/
  └── _complete           ← summary 작성 후 atomic 마킹 (sentinel)
_latest -> runs/<eval_tag>  ← symlink (또는 _latest.txt fallback)
HISTORY.md                ← §0-6 시계열 로그
.fail_count               ← §10 실패 카운터
```

요약:
- 모든 점수 파싱은 `${EVAL_RESULTS_ROOT}/<model>/<method>/_latest/<exp_label>-…json` 경로만 사용. flat 경로(`<base>/<label>-….json`)에 점수가 있으면 LEGACY (§5-5 (3)) — `--flat` 호출로 생긴 ad-hoc 결과로 간주.
- 같은 셀을 재평가할 때마다 새 `runs/<eval_tag>/`가 생기고 `_latest`가 그쪽으로 atomic 이동. 이전 평가는 `runs/`에 그대로 보존되어 §0-6 history의 입력.
- 기존 `_latest`에 벤치만 추가하고 싶을 때(예: 처음 4개 벤치만 끝났고 BBH만 마저 돌리려는 경우): `python -m tads.eval ... --eval_tag=latest --benchmarks bbh` — 새 dir을 안 만들고 같은 run에 JSON만 끼워 넣음.

### 5-2. 로그

```
${EVAL_RESULTS_ROOT}/<model>/<method>/_latest/logs/eval_<ts>_r0.log
```

(과거 run의 로그는 `${EVAL_RESULTS_ROOT}/<model>/<method>/runs/<eval_tag>/logs/`에 그대로 있음.)

### 5-3. 중복 평가 방지 로직 (에이전트가 직접 판정해야 함)

`python -m tads.eval`은 자체적인 "이미 함" 체크가 **없다** (기존 bash wrapper도 마찬가지였음). 호출 전에 에이전트가 판단해야 한다.

새 run-layout (§3-1) + eval history layout (§5-1) 기준 — 셀 한 개에 대해 **다음이 모두 만족**되면 eval 재실행 불필요:

1. `${OUTPUT_ROOT}/main_7b/<model>/<method>/_latest`가 가리키는 run의 가장 큰
   sealed `epoch_N/`(= `_complete` 파일이 있는 것)을 찾는다.
2. `${EVAL_RESULTS_ROOT}/<model>/<method>/_latest`가 존재하고, 그 안에
   `_complete` sentinel과 `<experiment_label>-eval_summary.json`이 둘 다 있음.
3. 그 summary의 mtime이 위 sealed epoch의 mtime보다 **나중**.

세 가지 모두 만족하지 않으면 NEED-EVAL → 다음 tick에 재평가 큐잉.

`_latest` 포인터가 갱신됐다는 건:
- 학습 측 `_latest` 갱신 → 새 ckpt가 sealed됐다 → eval 측 `_latest`의 summary mtime이 sealed보다 옛날이면 자동으로 NEED-EVAL.
- eval 측 `_latest` 갱신 → 새 평가가 완료됐다 → DONE으로 전환, 점수는 새 `_latest/`에서 다시 읽기.

**`runs/<tag>/`의 다른 과거 eval run들은 점수 표 갱신 대상이 아니다** (단, `--eval_tag <과거 tag>`로 사용자가 명시적으로 그걸 다시 promote할 수는 있음 → 그러면 `_latest`가 그쪽을 가리키고 자동으로 그 점수가 표에 반영). §0-6 HISTORY.md에는 모든 run의 점수 변동을 기록.

(Legacy 스크립트 `auto_eval_7b_fullft.sh`의 `.eval_done` 센티넬은 더 이상 권장 형식이 아니다 — 새 layout에서는 `_latest/_complete`가 그 역할을 한다. 새 에이전트는 `.eval_done`을 만들지 말 것. 이미 있는 건 무시하고 §5-3 의 3-조건 판정만 사용.)

### 5-4. 셀 상태 분류 (4 파일-state + 1 process-state = 5 display state) — 매 tick 시작 시 모든 셀에 대해 판정

매트릭스 16개 셀 각각을 다음 **4가지 file-state 중 하나**로 분류 (디스크 상태 기반). 그 위에 **process-state** 1종(eval 프로세스가 떠있는지)을 겹쳐 score board에는 항상 **상태 토큰 (`학습전` / `학습중` / `eval대기` / `eval중` / `NN.NN%`) 중 하나**로만 표기.

**🔑 핵심 규칙 — `cfg_target_epochs`의 정의**:

`cfg_target_epochs`는 **그 run dir의 `cfg.json` snapshot에 저장된 `train_epochs`** (= 학습이 launch된 시점의 값)이지, **라이브 YAML 파일이 아니다**. 즉:

- 사용자가 epoch=3으로 학습 launch → 그 run dir의 cfg.json엔 `train_epochs: 3` 박힘 → epoch_3 sealed = DONE (이후 라이브 YAML이 4로 바뀌어도 영향 없음).
- 사용자가 그 셀을 **새로** launch (auto timestamp로 새 `runs/<tag>/` 생성) → 그때의 라이브 YAML이 4면 새 cfg.json엔 `train_epochs: 4` 박힘 → epoch_4 sealed = DONE.
- `_latest` 포인터는 가장 최신의 sealed run을 가리키므로, 새 run이 완료되면 자동으로 그쪽 cfg.json이 기준이 됨.

**이 규칙 때문에**: 라이브 YAML의 `train_epochs` 값을 올려도(예: qwen25/mistral/deepseek 3→4), **이미 sealed된 3-epoch runs는 재학습 대상이 아님**. 그 셀들은 자기 run dir의 cfg.json(=3)을 기준으로 DONE 유지. 사용자가 다시 학습 launch했을 때부터 새 값(4) 적용.

| 상태 | 판정 조건 | 에이전트 행동 | Score board 마커 |
|---|---|---|---|
| **NEED-TRAIN** | `${OUTPUT_ROOT}/main_7b/<model>/<method>/`이 없거나, `_latest` 포인터(또는 `_latest.txt`)가 없고 평탄 layout의 `epoch_*`도 없음 | **학습 자동 실행 금지** (사용자 영역). 사용자에게 "이 셀은 학습이 아직 안 됐다" 보고만. eval은 ckpt가 없어 의미 없음. | `학습전` (학습 프로세스가 살아있으면 `학습중`) |
| **NEED-EVAL** | sealed epoch이 존재하지만, `_latest/<exp_label>-eval_summary.json` + `_latest/_complete` 조합이 없거나 mtime ≤ latest sealed epoch mtime | **다음 tick 의 §7 3.7 dispatch pass에서 빈 GPU 만큼 에이전트가 자동 launch** (1 GPU = 1 cell, §4-1 형식). 큐가 GPU 보다 많으면 다음 tick에 또 dispatch. | `eval대기` (eval 프로세스가 떠있거나 eval 로그 mtime < 5분이면 `eval중`) |
| **LEGACY** | `_latest`가 없고 BASE eval dir에 flat 포맷 파일만 있음 (예: `<base>/<exp_label>-eval_summary.json`, `<base>/eval_summary.json` 접두어 없음, 또는 벤치별 `<base>/mmlu.json` 류) | 점수는 flat 파일에서 추출해 표에 잠정 기재하되, **재실행 권장 (NEED-EVAL과 동일하게 큐잉)**. 새 history layout으로 다시 평가하면 자동으로 DONE으로 갱신. | `NN.NN%` (옛 점수 그대로, `(provisional)` 주석 별도 라인) |
| **DONE** | `${EVAL_RESULTS_ROOT}/<model>/<method>/_latest/` 안에 `_complete` sentinel과 `<exp_label>-eval_summary.json`이 모두 있고, summary mtime > latest sealed epoch mtime | 건너뜀. 점수 표에 숫자 반영. | `NN.NN%` 예: `47.56%` |

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
    """Return path of the most-recent sealed ckpt under run_dir, or None.

    Priority:
      (1) <run_dir>/epoch_last/_complete (new layout, 2026-05-16~)
      (2) max sealed epoch_<N>/ (legacy)
    """
    # (1) epoch_last/ — single rolling ckpt dir, _complete sentinel content
    #     is the int epoch number. Beats any numeric dir.
    last = f"{run_dir}/epoch_last"
    if os.path.exists(f"{last}/_complete"):
        return last
    # (2) Legacy numeric epoch_<N>/.
    sealed = []
    for p in glob(f"{run_dir}/epoch_*"):
        name = os.path.basename(p)
        if name == "epoch_last":
            continue
        try: n = int(name.replace("epoch_", ""))
        except ValueError: continue
        if not os.path.exists(f"{p}/_complete"): continue
        sealed.append((n, p))
    return max(sealed)[1] if sealed else None


def _sealed_epoch_n(sealed_path):
    """Read the epoch number from a sealed ckpt dir.

    For epoch_last/_complete, the int is the sentinel content.
    For legacy epoch_<N>/, parse from the directory name.
    """
    if sealed_path is None:
        return 0
    name = os.path.basename(sealed_path)
    if name == "epoch_last":
        try:
            return int(open(f"{sealed_path}/_complete").read().strip() or 0)
        except (OSError, ValueError):
            return 0
    try:
        return int(name.replace("epoch_", ""))
    except ValueError:
        return 0

def classify_cell(model, method):
    ckpt_root = f"{OUTPUT_ROOT}/main_7b/{model}/{method}"
    # 1) 학습 ckpt 최신성 확인 (새 run-layout 우선, legacy flat fallback)
    latest_run = _resolve_latest_run(ckpt_root)
    if latest_run:
        latest_epoch = _largest_sealed_epoch(latest_run)
    else:
        # Legacy flat layout fallback (옛날 학습 결과용)
        flat = sorted(glob(f"{ckpt_root}/epoch_*"))
        latest_epoch = flat[-1] if flat else None
    if latest_epoch is None:
        return "NEED-TRAIN", None

    # 2) eval 결과 최신성 확인 — 새 history layout (_latest 경유) 우선
    eval_base = f"{EVAL_RESULTS_ROOT}/{model}/{method}"
    label = f"{model}_{method}"     # experiment_label 규칙 (parent_stem)
    eval_latest = _resolve_latest_run(eval_base)   # 같은 helper 재사용
    if eval_latest is not None and exists(f"{eval_latest}/_complete"):
        new_summary = f"{eval_latest}/{label}-eval_summary.json"
        if exists(new_summary) and mtime(new_summary) > mtime(latest_epoch):
            return "DONE", new_summary

    # 3) 새 layout이 미완성 / 없음 → flat layout(legacy, `--flat` ad-hoc 결과
    #    또는 history layout 도입 전 결과)도 후보로 검사
    flat_summary = f"{eval_base}/{label}-eval_summary.json"
    if exists(flat_summary) and mtime(flat_summary) > mtime(latest_epoch):
        return "LEGACY", flat_summary   # 점수는 잠정 사용, 새 eval로 갱신 필요

    legacy_candidates = (
        glob(f"{eval_base}/eval_summary.json")        # 옛 접두어 없는 이름
        + glob(f"{eval_base}/{label}-*.json")         # flat 벤치별 (새 포맷이지만 history layout 밖)
        + glob(f"{eval_base}/mmlu.json")              # 옛 벤치별 (접두어 없음)
        + glob(f"{eval_base}/gsm8k.json")
        + glob(f"{eval_base}/humaneval.json")
        + glob(f"{eval_base}/tydiqa.json")
        + glob(f"{eval_base}/bbh.json")
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
| 1 (최신) | `_latest/<experiment_label>-eval_summary.json` <br/> 예: `_latest/llama2_tads_10-eval_summary.json` | history layout (§5-1)의 표준. `_latest` 안에 `_complete` sentinel과 같이 있을 때만 DONE. 모든 벤치 점수와 메타데이터 포함. |
| 2 (최신, 벤치별) | `_latest/<experiment_label>-<bench>.json` <br/> 예: `_latest/llama2_tads_10-mmlu.json` | 벤치별 상세. summary가 없으면 이것들로 합산 (단, 5개 벤치 다 있을 때만 DONE 처리). |
| 3 (legacy, flat) | `<exp_label>-eval_summary.json` (BASE 디렉터리 직속) | `--flat` 호출이나 history layout 도입 전 산출물. `_latest`가 없을 때만 본다. LEGACY로 분류. |
| 4 (legacy, 접두어 없음) | `eval_summary.json` (BASE 디렉터리 직속, 접두어 없음) | 옛 코드 산출물. label 충돌 위험(다른 셀의 결과를 덮어썼을 수 있음). LEGACY로 분류. |
| 5 (legacy, 벤치별 접두어 없음) | `mmlu.json` / `gsm8k.json` / `humaneval.json` / `tydiqa.json` / `bbh.json` (BASE 디렉터리 직속, 접두어 없음) | 옛 벤치별. LEGACY로 분류. |

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

LEGACY로 분류된 셀은 가능하면 **다음 tick에 새 eval을 history layout으로 돌려 `_latest`를 새로 만든다**. eval.py가 자동으로 `runs/<eval_tag>/`에 결과를 떨구고 `_latest`를 atomic 갱신하므로, BASE dir의 flat 파일은 그대로 둬도 LEGACY → DONE 전환에 방해되지 않는다 (점수 우선순위 표 §5-5 (1)에 따라 `_latest`가 항상 이긴다).

선택적으로 BASE의 flat 잔재를 정리하려면 (필수 아님 — `_latest`만 보고 동작하기 때문에 정리 안 해도 무해, 단지 `ls`로 봤을 때 깔끔해짐):

```bash
mkdir -p "${eval_base}/legacy"
# BASE 직속의 옛 flat 결과만 옮긴다. runs/ 디렉터리와 _latest, HISTORY.md,
# .fail_count는 절대 건드리지 말 것.
for f in "${eval_base}"/eval_summary.json \
         "${eval_base}"/mmlu.json "${eval_base}"/gsm8k.json \
         "${eval_base}"/humaneval.json "${eval_base}"/tydiqa.json \
         "${eval_base}"/bbh.json \
         "${eval_base}"/*-eval_summary.json \
         "${eval_base}"/*-mmlu.json "${eval_base}"/*-gsm8k.json \
         "${eval_base}"/*-humaneval.json "${eval_base}"/*-tydiqa.json \
         "${eval_base}"/*-bbh.json; do
    [ -f "$f" ] && mv "$f" "${eval_base}/legacy/" 2>/dev/null || true
done
```

(`runs/<eval_tag>/` 안의 파일은 새 포맷이므로 절대 옮기지 말 것 — `_latest`가 그쪽을 가리킨다.)

---

## 6. Legacy 7b_fullft 자동 감시 모드 (사용 금지 — 역사 기록)

```bash
# ❌ 더 이상 쓰지 말 것 — §4 의 1셀-1명령 정책에 위배
bash scripts/auto_eval_7b_fullft.sh <gpu_id> [run1 run2 ...]
```

`auto_eval_7b_fullft.sh`는 무한 루프(60초 sleep)로 여러 run을 polling하고 차례로 eval을 자동 실행하는 옛 자동화 스크립트다. 다음 이유로 **사용자도, 에이전트도 호출하지 말 것** (§4 / §8 금지 목록):

- 여러 잡을 sequential로 dispatch → 한 잡 fail이 다음 잡 시작에 영향 (cascade).
- `.eval_done` sentinel을 BASE 디렉터리에 직접 touch → 새 history layout(`runs/<eval_tag>/_complete`)과 비호환.
- 무한 루프라 tick 모델에 안 맞음. 1셀-1명령 정책과 정면 충돌.
- 메인 매트릭스(`main_7b/...`)는 절대 안 건드림 (legacy `7b_fullft/...` 트리만 다룸) — 매트릭스 외 트리에 대해서도 이 스크립트로 자동 실행하지 말 것.

이 트리(`${OUTPUT_ROOT}/7b_fullft/...`)를 평가해야 한다면 §4-1 패턴(단일 셀 단일 명령)을 따른다:
```bash
# 셀 1개 — 사용자가 직접 입력
CUDA_VISIBLE_DEVICES=<gpu> nohup python -m tads.eval \
    --config configs/experiments/7b_fullft_<run>.yaml \
    --ckpt ${OUTPUT_ROOT}/7b_fullft/<run>/epoch_last \  # epoch_3 (legacy)도 가능
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
    --out_dir ${EVAL_RESULTS_ROOT}/7b_fullft/<run> \
    >> logs/eval_7b_fullft_<run>.log 2>&1 &
```

---

## 7. 에이전트가 한 번의 tick에서 해야 할 일 (의사 코드)

**원칙 재강조**: 셀 1개 = GPU 1장 = `python -m tads.eval` 백그라운드 프로세스 1개.
bash wrapper 호출 금지. 빈 GPU가 있는 만큼만 한꺼번에 launch하고, 큐에 남은 셀은
다음 tick에서 또 빈 GPU가 생기면 launch.

> # 🚨 RE-CONFIRM — 출력 직전 PRE-FLIGHT CHECKLIST 12항목 다 통과했나? (문서 최상단 참조)
> 
> ```
> [ ] 01 ## (6) Current 헤더 존재
> [ ] 02 ## (7) Latest 헤더 존재         ← 자주 누락. 반드시 확인.
> [ ] 03 ## (8) Best 헤더 존재           ← 자주 누락. 반드시 확인.
> [ ] 04 ## (9) History 헤더 존재        ← 자주 누락. 반드시 확인.
> [ ] 05/06/07 (6)/(7)/(8)에 Params 컬럼 있음
> [ ] 08 (6) Current 80개 셀에 `-` / `err` / 공란 0개
> [ ] 09 (9) History 이번 tick 변경분 append 됨
> [ ] 10 학습 alive인 셀은 (6)에서 `학습중`, (7)/(8) 이전 값 유지
> [ ] 11 (8) Best는 max() 비교만 — 낮은 새 점수로 덮어쓰지 말 것
> [ ] 12 코드 펜스 + monospace + CJK width 2 + pipe table 금지 + bold 금지
> [ ] 13 (6) Status 컬럼 = 3-facet (`<history> · <오류> · <발산알람>`) ←
>       자주 facet 1만 적고 끝내는 위반. placeholder로라도 facet 3개 유지.
> ```
> 
> 하나라도 미통과 → atomic 교체 중단 → 누락 부분 §0-4 (6)/(7)/(8)/(9) 템플릿으로 재생성 → 다시 13개 체크 → 통과해야 사용자에게 보고.

```
1. cd /home/jieun/kms/tads
2. source scripts/setup_env.sh   # 매 쉘마다 한 번

2.5 log-tail polling pass (§0-7) — classify보다 먼저
   for model in {llama2, qwen25, mistral, deepseek}:
     for method in {full_100, random_10, data_agent_10, tads_10}:
         train_log = newest("${OUTPUT_ROOT}/main_7b/${model}/${method}/_latest/logs/train_*_r0.log")
         eval_log  = newest("${EVAL_RESULTS_ROOT}/${model}/${method}/_latest/logs/eval_*.log")
         # script append 로그도 fallback으로 검사
         train_log = train_log or "logs/main_7b_${model}_${method}.log"
         eval_log  = eval_log  or "logs/eval_main_7b_${model}_${method}.log"
         signals[(model, method)] = extract_signals(train_log, eval_log)  # §0-7 (b)
         # signals: sft_progress, anchor_progress, sealed_epoch_event,
         # eval_bench_progress, eval_sealed_event, errors, log_mtime_age
         if signals.hang_suspect:           # §0-7 (d): mtime > 30분 + alive
             mark_hang(model, method, signals)
         if signals.new_oom_or_error:
             append_history_md(model, method, signals)   # §0-6 (c)
         if signals.new_sealed_eval:
             sync_score_board_now(model, method)         # 다음 tick 안 기다리고 표 갱신
         update_status_column(model, method, signals)     # §0-5/§0-6 (a)

3. classify pass — 모든 80개 셀에 §5-4 표의 상태 토큰 부여
   for model in {llama2, qwen25, mistral, deepseek}:
     for method in {full_100, random_10, data_agent_10, tads_10}:
         ckpt_root = ${OUTPUT_ROOT}/main_7b/${model}/${method}
         eval_base = ${EVAL_RESULTS_ROOT}/${model}/${method}
         label     = "${model}_${method}"
         sig       = signals[(model, method)]   # 2.5 polling 결과

         # 1) ckpt 측 결정
         latest_run = resolve_latest_run(ckpt_root)
         if not latest_run:
             status[(model, method)] = "학습전"
             continue
         latest_epoch = largest_sealed_epoch(latest_run)   # path or None
         train_alive  = pgrep_alive(f"python.*tads.train.*{model}/{method}\\.yaml")
         cfg          = json.load(latest_run + "/cfg.json")
         target_epochs = int(cfg.get("train_epochs", 3))
         # sealed N = epoch_last/_complete sentinel content (single int),
         # or for legacy epoch_<N>/ layout the max N. See §3-1.
         sealed_n     = sealed_epoch_n(latest_epoch)   # 0 if None
         if train_alive or sealed_n < target_epochs:
             status[(model, method)] = "학습중"
             continue

         # 2) eval 측 결정 — §5-4 표의 5단계 우선순위 + log 기반
         eval_alive   = pgrep_alive(f"python.*tads.eval.*{model}/{method}\\.yaml")
         eval_latest  = resolve_latest_run(eval_base)
         summary      = eval_latest + f"/{label}-eval_summary.json" if eval_latest else None
         sentinel_ok  = eval_latest and exists(eval_latest + "/_complete")

         if eval_alive or sig.eval_log_mtime_age < 300:    # 5분
             status[(model, method)] = "eval중"
             continue
         if sentinel_ok and exists(summary) and mtime(summary) > mtime(latest_epoch):
             status[(model, method)] = f"{parse_score(summary):.2f}%"     # DONE
             continue
         if sig.recent_oom_or_traceback:
             append_history_md(model, method, "[eval] failure", sig)
             increment_fail_count(eval_base)
             status[(model, method)] = "eval대기"
             continue
         status[(model, method)] = "eval대기"

   ## 3.5 transitions pass — (7) Latest / (8) Best / (9) History 갱신 입력 생성
   # status[(m, k)] 가 바뀐 셀들에 대해 (6)/(7)/(8)/(9) 어느 표를 어떻게
   # 갱신할지 결정 (§0-4 (10) 표 흐름).
   for (model, method), new in status.items():
       prev = read_prev_value_from_experiments_md(model, method)   # (6) Current cell
       if prev == new:
           continue
       # 어떤 표를 갱신할지 (§0-4 (10) 표):
       updates[(model, method)]["current"] = new      # (6) 무조건 덮어씀
       if re.match(r"^\d+\.\d{2}%$", new):
           # 새 DONE NN.NN% 나왔음
           updates[(model, method)]["latest"]  = new      # (7) 덮어씀
           updates[(model, method)]["best_if_higher"] = new   # (8) max 비교
       # (7) Latest / (8) Best는 학습중 / eval중 / eval대기로 변할 때 안 건드림 (§0-4 (10))
       # (9) History에는 무조건 한 줄 append
       updates[(model, method)]["history_row"] = (
           f"{now}  cell={model}/{method}  prev={prev}  new={new}  "
           f"reason={infer_reason(model, method, sig)}  source={pick_source(model, method, sig)}"
       )

3.7 dispatch pass — 빈 GPU 만큼 eval 자동 launch (1 GPU = 1 (cell, bench), 단일 bench, BBH 제외)
   # 운영 가정 (사용자 명시): 학습은 모두 끝났고, 노드에는 eval만 돈다.
   # 에이전트가 (cell, bench) 큐를 매 tick continuous하게 dispatch — 사용자 개입 0.
   # 정책: --benchmarks는 항상 단일 bench. mmlu → gsm8k → humaneval → tydiqa
   # 순서대로 한 cell의 모든 bench가 끝나야 다음 단계로. BBH 제외 (사용자 직접).
   AUTO_BENCHES = ("mmlu", "gsm8k", "humaneval", "tydiqa")  # 순서 그대로

   def next_bench_for_cell(model, method):
       """이 cell의 4-bench 큐에서 아직 DONE 안 된 가장 앞 bench를 반환.
       모두 DONE이면 None."""
       eval_base = f"{EVAL_RESULTS_ROOT}/{model}/{method}"
       eval_latest = resolve_latest_run(eval_base)
       label = f"{model}_{method}"
       for b in AUTO_BENCHES:
           if eval_latest is not None and exists(f"{eval_latest}/_complete"):
               bj = f"{eval_latest}/{label}-{b}.json"
               if exists(bj) and parse_score_pct(bj) is not None:
                   continue   # 이 bench는 이미 DONE
           return b
       return None    # 4-bench 모두 DONE

   # (cell, bench) pair 큐 — 셀당 다음으로 돌아야 할 bench 1개만.
   pair_queue = []
   for (m, x), v in status.items():
       if v != "eval대기":
           continue
       b = next_bench_for_cell(m, x)
       if b is None:
           continue
       # 같은 cell이 이미 다른 GPU에서 running이면 그 cell 큐에서 제외
       # (같은 cell의 다음 bench는 현재 bench 끝난 후 다음 tick에 enqueue).
       if pgrep_alive(f"python.*-m tads.eval.*{m}/{x}\\.yaml"):
           continue
       pair_queue.append((m, x, b))

   # 빈 GPU 목록 = nvidia-smi에서 memory.used < 2GB.
   free_gpus = [
       idx for idx, mem_used in nvidia_smi_query("memory.used")
       if mem_used < 2048
   ]

   # 1 GPU = 1 (cell, bench). K개 빈 GPU면 K개 launch.
   for gpu in free_gpus:
       if not pair_queue:
           break
       m, x, b = pair_queue.pop(0)
       cmd = (
           # 단일 bench dispatch (2026-05-17 정책). 4개를 한 명령에 묶지 말 것.
           # bench별 log path 분리 — log race 방지.
           f'CUDA_VISIBLE_DEVICES={gpu} nohup python -m tads.eval '
           f'--config configs/experiments/main_7b/{m}/{x}.yaml '
           f'--benchmarks {b} '
           f'--out_dir ${{EVAL_RESULTS_ROOT}}/{m}/{x} '
           f'>> logs/eval_{m}_{x}_{b}.log 2>&1 &'
       )
       subprocess.Popen(cmd, shell=True)
       time.sleep(0.5)   # CUDA init race 회피
       log(f"[tick] dispatched {m}/{x}/{b} -> GPU {gpu}")
       # 즉시 (cell, bench) 상태를 eval중으로 표 표기 (다음 tick까지 안 기다림).
       status[(m, x, b)] = "eval중"
       updates[(m, x, b)]["current"] = "eval중"
       updates[(m, x, b)]["history_row"] = (
           f"{now}  cell={m}/{x}  bench={b}  prev=eval대기  new=eval중  "
           f"reason=auto-dispatch (GPU {gpu})  source=logs/eval_{m}_{x}_{b}.log"
       )
   # 매 tick continuous: 16 cells × 4 benches = 64 pairs 전부 DONE될 때까지
   # 매 tick 빈 GPU 만큼 dispatch. 사용자 개입 0 (BBH 제외).

4. report pass — experiments.md + dashboard + 사용자 액션 섹션 atomic 갱신
   #  eval auto-launch는 위 3.7에서 처리됨 (§0 / §4). 학습 auto-launch는 절대 금지.

   ## STRUCTURE — MUST 5개 섹션 모두 갱신 (§0-4 절대 비교섭 규칙):
   ## (6) Current      — 상태 토큰 + Status + Params, 80 cells
   ## (7) Latest       — NN.NN%/- + Last DONE + Params, 80 cells
   ## (8) Best         — NN.NN%/- + AVG + Best run + Params, 80 cells
   ## (9) History      — 그 tick의 셀 값 변경마다 한 줄 append (newest at bottom)
   ## 4개 표 + 1 로그 누락 시 → atomic 교체 실패로 간주 → 재생성 후 다시 검증.

   - §0-4 (1)-(5) 모델별 점수 표 + §0-5 dashboard 코드 펜스 안만 atomic 교체
   - §0-4 (6) 80-cell 표에 `-`가 단 하나도 남지 않도록 dash-self-check 수행
     (§0-4 셀프 체크 bash). 통과 못 하면 그 tick의 출력은 사용자에게
     보고하지 않고 분류를 재돌릴 것.
   - §0-4 절대 비교섭 STRUCT-OK self-check 수행 — (6)/(7)/(8)/(9) 헤더 4개
     모두 존재 + (6)/(7)/(8)에 Params 컬럼 존재 확인. STRUCT-FAIL이면 atomic
     교체 중단, 누락 섹션 생성 후 재시도.
   - 재학습 감지된 셀은 (6)만 학습중으로 덮어쓰고 (7)/(8) Latest/Best는
     이전 값 유지, (9) History에 prev=NN.NN% new=학습중 append (§0-4 (10)).
   - 학습전 / eval대기 셀이 1개 이상이면 experiments.md 상단에 "사용자 액션
     필요" 섹션 2개(학습 미실행, eval 미실행) 갱신.
   - 채팅 보고: 변경된 행만 1줄 요약(§0-7 (e) 형식). raw 표 dump 금지.

5. 종료. 다음 cron tick에서 1번부터 다시 시작.
   에이전트는 자기 자신을 무한 루프로 띄우지 말 것 — cron이 idempotent하게
   호출.
```

---

## 8. 금지 사항 / 함정

- **eval은 에이전트가 자동 launch한다** (§0 / §4 재정의, 2026-05-16). 매 tick `eval대기` 셀과 빈 GPU를 매칭해 셀별로 `python -m tads.eval`을 nohup background launch. 사용자 개입 0.
- **학습은 절대 자동 launch 금지** — `python -m tads.train`, `torchrun`, `run_main_7b.sh` 호출은 사용자만 가능. NEED-TRAIN 셀은 보고만.
- **eval은 1셀 = 1 GPU = 1명령으로만 실행** (§4 정책). batch wrapper(`run_eval_main_7b.sh`, `auto_eval_7b_fullft.sh`)는 에이전트도 사용자도 쓰지 말 것 — wrapper의 cascade fail / log race / `_latest` symlink race 문제. 셀 여러 개를 동시에 돌릴 때도 각 셀마다 별개 `python -m tads.eval` 명령을 다른 GPU에 핀해서 따로 실행. for 루프로 여러 셀을 한꺼번에 띄우는 패턴 금지.
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

### 9-1. 권장 간격 (2026-05-17 정책)

- **15분마다** (`*/15 * * * *`) — eval 셀 1개당 평균 5~15분이라 15분 폴링이면 빈 GPU가 생긴 직후 다음 (cell, bench) pair를 빠르게 dispatch. 매 tick continuous하게 진행하므로 16 cells × 4 benches = 64 pair가 빈 GPU 수에 따라 1~3일 안에 완료.
- 사용자가 BBH를 수동 launch 중인 경우에도 별 영향 없음 — auto dispatch는 BBH 제외 4-bench만 처리하며, 학습/BBH 점유 GPU는 free_gpus 필터에서 자동 제외.
- 학습이 동시에 안 돈다는 운영 가정 하에서 15분이 sweet spot. 만약 학습이 동시에 돈다면 30분으로 늦추면 학습 잡과의 메모리 경합 줄임.

### 9-2. crontab 예시

서버에 `crontab -e`로 등록:

```cron
# TADS auto-eval — 매 15분, eval대기 셀이 있으면 빈 GPU 만큼 자동 dispatch (4-bench, BBH 제외)
*/15 * * * * /home/jieun/kms/tads/scripts/auto_eval_tick.sh >> /home/jieun/kms/tads/logs/auto_eval_cron.log 2>&1
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
  # Return path of the most-recent sealed ckpt under <run_dir>, or empty.
  # Priority: (1) epoch_last/ (new layout, 2026-05-16~), (2) max sealed
  # numeric epoch_<N>/ (legacy).
  local run_dir=$1
  if [ -f "${run_dir}/epoch_last/_complete" ]; then
    echo "${run_dir}/epoch_last"
    return
  fi
  ls -1d "${run_dir}"/epoch_* 2>/dev/null | sort -V | while read -r p; do
    [ "$(basename "$p")" = "epoch_last" ] && continue
    [ -f "${p}/_complete" ] && echo "${p}"
  done | tail -n 1
}

sealed_epoch_n() {
  # Return the epoch number for a sealed ckpt path. For epoch_last/, the
  # number is the integer content of _complete. For legacy epoch_<N>/, it
  # is parsed from the directory name. Returns 0 when missing/malformed.
  local p="$1"
  [ -z "$p" ] && { echo 0; return; }
  local base
  base="$(basename "$p")"
  if [ "$base" = "epoch_last" ]; then
    cat "${p}/_complete" 2>/dev/null | head -1 | grep -oE '^-?[0-9]+' \
      | head -1 || echo 0
    return
  fi
  echo "${base}" | sed 's/^epoch_//' | grep -oE '^-?[0-9]+' || echo 0
}

# ---------------- §0-7: log-tail polling pass (classify보다 먼저) ---------------
# 각 셀의 가장 최근 train + eval 로그를 tail해서 Status / HISTORY.md / 점수표를
# 선제 동기화. 이 단계는 실패해도 아래 classify는 그대로 돈다 (best-effort).
poll_log_signals() {
  local model=$1 method=$2
  local ckpt_base="${OUTPUT_ROOT}/main_7b/${model}/${method}"
  local eval_base="${EVAL_RESULTS_ROOT}/${model}/${method}"

  # newest per-run log (없으면 script-side append 로그 fallback)
  local train_log eval_log
  train_log=$(ls -t "${ckpt_base}/_latest/logs/train_"*_r0.log 2>/dev/null | head -1 \
              || true)
  [ -z "$train_log" ] && train_log="${REPO}/logs/main_7b_${model}_${method}.log"
  eval_log=$(ls -t "${eval_base}/_latest/logs/eval_"*.log 2>/dev/null | head -1 \
             || true)
  [ -z "$eval_log" ] && eval_log="${REPO}/logs/eval_main_7b_${model}_${method}.log"

  # §0-7 (b) 7개 signal 추출 — 단순 grep + tail. 결과는 stdout 라인으로 모음.
  echo "=== signals: ${model}/${method} ==="
  if [ -f "$train_log" ]; then
    # 1. SFT 진행
    grep -E 'SFT \| epoch=[0-9]+ \| step=' "$train_log" 2>/dev/null | tail -1 \
      | sed 's/^/  [sft]   /'
    # 2. anchor / collect_episode 진행
    grep -E 'TrajectoryAnchor.update|collect_episode' "$train_log" 2>/dev/null | tail -3 \
      | sed 's/^/  [sel]   /'
    # 3. epoch sealed 이벤트
    grep -E 'Checkpoint saved \+ sealed|_latest -> runs/' "$train_log" 2>/dev/null | tail -2 \
      | sed 's/^/  [seal]  /'
  fi
  if [ -f "$eval_log" ]; then
    # 4. eval 벤치 진행 + 실패
    grep -E 'Progress:|Eval start|Benchmark .* failed' "$eval_log" 2>/dev/null | tail -5 \
      | sed 's/^/  [eval]  /'
    # 5. eval sealed 이벤트
    grep -E 'Eval run sealed:|_latest -> runs/.* under ' "$eval_log" 2>/dev/null | tail -2 \
      | sed 's/^/  [eseal] /'
  fi
  # 6. 오류 — train + eval 양쪽
  for L in "$train_log" "$eval_log"; do
    [ -f "$L" ] || continue
    grep -iE '(out of memory|OOM|CUDA error|RuntimeError|Killed|Traceback)' "$L" \
        2>/dev/null | tail -5 | sed "s|^|  [err]   ${L##*/}: |"
  done
  # 7. mtime age — hang 감지
  local now=$(date +%s)
  for kind in train eval; do
    local L
    [ "$kind" = train ] && L="$train_log" || L="$eval_log"
    [ -f "$L" ] || continue
    local mtime age proc_alive=0
    mtime=$(stat -c %Y "$L")
    age=$(( now - mtime ))
    if pgrep -af "python.*tads.${kind}.*${model}/${method}\.yaml" >/dev/null 2>&1; then
      proc_alive=1
    fi
    if [ "$proc_alive" = 1 ] && [ "$age" -ge 1800 ]; then
      echo "  [HANG]  ${kind} ${model}/${method} idle ${age}s (log=${L##*/})"
    elif [ "$proc_alive" = 1 ] && [ "$age" -ge 300 ]; then
      echo "  [slow]  ${kind} ${model}/${method} idle ${age}s"
    fi
  done
}

# 로그 polling 결과는 tick 로그에 그대로 dump — 에이전트가 그 다음 패스에서
# 읽어 Status / HISTORY.md / 점수표를 업데이트한다.
echo "[tick $(date -Is)] === log-tail polling pass ==="
for model in llama2 qwen25 mistral deepseek; do
  for method in full_100 random_10 data_agent_10 tads_10; do
    poll_log_signals "$model" "$method"
  done
done

# --------------------------------- §7-3 classify pass -------------------------
need=()
for model in llama2 qwen25 mistral deepseek; do
  for method in full_100 random_10 data_agent_10 tads_10; do
    ckpt_root="${OUTPUT_ROOT}/main_7b/${model}/${method}"
    latest_run=$(resolve_latest_run "$ckpt_root")
    [ -z "$latest_run" ] && continue
    latest=$(largest_sealed_epoch "$latest_run")
    [ -z "$latest" ] && continue
    eval_base="${EVAL_RESULTS_ROOT}/${model}/${method}"
    # eval 측도 같은 helper로 _latest 해석 (BASE 구조가 train과 동일).
    eval_latest=$(resolve_latest_run "$eval_base")
    label="${model}_${method}"
    summary="${eval_latest:+${eval_latest}/${label}-eval_summary.json}"
    sentinel="${eval_latest:+${eval_latest}/_complete}"
    # DONE 조건 (§5-3 의 3-조건):
    #   eval _latest 가 있고, _complete sentinel과 summary 둘 다 있으며,
    #   summary mtime > latest sealed epoch mtime.
    if [ -n "$eval_latest" ] \
       && [ -f "$sentinel" ] \
       && [ -f "$summary" ] \
       && [ "$summary" -nt "$latest" ]; then
      continue   # DONE
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
echo "[tick $(date -Is)] free GPUs (참고용): ${free_gpus[*]}  |  eval대기 큐: ${need[*]}"

# Dispatch pass — 빈 GPU 만큼 자동 launch (1 GPU = 1 cell). 2026-05-16
# 재정의: eval은 에이전트가 자동 실행 (운영 가정: 학습 동시 실행 없음).
# wrapper 호출 금지 — 셀별 `python -m tads.eval` 명령을 직접 nohup &.
launched_pids=()
LAUNCH_LOG="${LOG_DIR}/auto_dispatch_$(date +%Y%m%d).log"
for gpu in "${free_gpus[@]}"; do
  [ ${#need[@]} -eq 0 ] && break
  cell="${need[0]}"; need=("${need[@]:1}")
  model="${cell%/*}"; method="${cell#*/}"

  # 같은 셀이 이미 어디서 돌고 있으면 skip (중복 launch 방지).
  if pgrep -af "python -m tads.eval.*main_7b/${model}/${method}\.yaml" >/dev/null; then
    echo "[tick $(date -Is)] skip ${cell} — already running" | tee -a "$LAUNCH_LOG"
    continue
  fi

  cfg="configs/experiments/main_7b/${model}/${method}.yaml"
  out_dir="${EVAL_RESULTS_ROOT}/${model}/${method}"
  log_path="${LOG_DIR}/eval_${model}_${method}.log"
  echo "[tick $(date -Is)] dispatch ${cell} -> GPU ${gpu}" | tee -a "$LAUNCH_LOG"
  # 2026-05-17 정책: bench 하나씩 순차 dispatch. --benchmarks에 단일
  # bench만 (mmlu → gsm8k → humaneval → tydiqa 순). 4개를 한 명령에 묶지
  # 말 것. BBH는 자동 dispatch 제외 (사용자 직접).
  # $cell 큐 항목은 "<model>/<method>/<bench>" 형식 (single-bench pair).
  bench="${cell##*/}"          # 마지막 / 뒤 = bench 이름
  cellpath="${cell%/*}"         # 앞 부분 = <model>/<method>
  model="${cellpath%/*}"
  method="${cellpath#*/}"
  log_path="${LOG_DIR}/eval_${model}_${method}_${bench}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tads.eval \
      --config "$cfg" \
      --benchmarks "$bench" \
      --out_dir "$out_dir" \
      >> "$log_path" 2>&1 &
  pid=$!
  launched_pids+=("$pid:${model}/${method}/${bench}:gpu${gpu}")
  echo "  launched pid=$pid bench=$bench" | tee -a "$LAUNCH_LOG"
  sleep 0.5    # CUDA init race 회피
done

if [ ${#launched_pids[@]} -gt 0 ]; then
  echo "[tick $(date -Is)] launched: ${launched_pids[*]}" | tee -a "$LAUNCH_LOG"
fi
if [ ${#need[@]} -gt 0 ]; then
  echo "[tick $(date -Is)] queue remaining (다음 tick에 자동 dispatch): ${need[*]}"
fi

# 직전 tick에 에이전트가 띄운 eval 잡들의 종료 회수: 별도 pidfile 관리 없이
# §5-4 표 5단계로 매 tick 재판정해서 표 갱신한다. eval.py가 sentinel +
# _latest를 자체 처리하므로 외부 마커 작성 불필요. exit !=0인 잡은 다음
# 분류 pass에서 로그 tail에 OOM/Traceback이 잡혀 .fail_count 증가 (§10-3).
# 16셀 전부 DONE되면 큐가 영구적으로 비어 dispatch pass는 no-op.
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
# 새 run-layout: <ckpt_root>/_latest -> runs/<tag>/epoch_last/
ls -ld ${OUTPUT_ROOT}/main_7b/*/*/_latest 2>/dev/null | head
# 그 안의 sealed epoch_last (or legacy epoch_<N>)
for r in ${OUTPUT_ROOT}/main_7b/*/*/_latest; do
  # epoch_last 우선
  if [ -f "$r/epoch_last/_complete" ]; then
    n=$(cat "$r/epoch_last/_complete" 2>/dev/null | head -1)
    echo "OK   $r/epoch_last  (epoch=$n)"
  else
    # legacy fallback
    for e in "$r"/epoch_*; do
      [ "$(basename "$e")" = "epoch_last" ] && continue
      [ -f "$e/_complete" ] && echo "OK   $e  (legacy numeric)"
    done
  fi
done | head
# Legacy flat layout (옛날 학습용)도 확인
ls -d ${OUTPUT_ROOT}/main_7b/*/*/epoch_* 2>/dev/null | head
ls -d ${EVAL_RESULTS_ROOT}/*/* 2>/dev/null | head
ls -l ${EVAL_RESULTS_ROOT}/experiments.md 2>/dev/null || echo "experiments.md 아직 없음 — §0-4 템플릿으로 생성"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

네 줄 다 합리적인 출력이 나오면 §7 루프(또는 §9 cron tick)를 시작해도 된다.

---

## 13. 기준치 (사용자 입력) — 점수 검증용 reference

사용자가 직접 측정하거나 다른 환경에서 확인한 점수를 여기에 기록한다. 에이전트의 eval 결과가 이 값과 ±2%p 이상 차이나면 **노란불** (학습 / eval 환경 의심), ±5%p 이상이면 **빨간불** (재학습 권장). 점수 신뢰성 baseline 역할.

표 형식 = §0-4 표 작성 규칙(코드 펜스 + 고정폭 정렬). 값은 정확도(%) 둘째 자리. 셀이 비어있으면 (`-`) 사용자가 아직 측정하지 않았다는 뜻 (이 표는 에이전트가 자동 갱신하지 않으므로 `-` 사용 OK — 점수 표가 아니라 reference 표).

```
Model/Method               mmlu      gsm8k     humaneval tydiqa    bbh
========================== ========= ========= ========= ========= =========
llama2 / random_10         47.14%    14.13%    25.55%    44.16%    39.21%
llama2 / full_100          -         -         -         -         -
llama2 / data_agent_10     -         -         -         -         -
llama2 / tads_10           -         -         -         -         -

qwen25 / full_100          -         -         -         -         -
qwen25 / random_10         -         -         -         -         -
qwen25 / data_agent_10     -         -         -         -         -
qwen25 / tads_10           -         -         -         -         -

mistral / full_100         -         -         -         -         -
mistral / random_10        -         -         -         -         -
mistral / data_agent_10    -         -         -         -         -
mistral / tads_10          -         -         -         -         -

deepseek / full_100        -         -         -         -         -
deepseek / random_10       -         -         -         -         -
deepseek / data_agent_10   -         -         -         -         -
deepseek / tads_10         -         -         -         -         -
```

기록 시점 / 출처:
- `llama2 / random_10` — 2026-05-16 사용자 직접 입력 (이전 환경 재현 값).

사용법:
- 에이전트는 매 tick에서 새로 산출된 `NN.NN%`를 위 reference와 비교.
- delta ≥ +2%p 또는 ≤ −2%p → §0-6 (a) Status 컬럼에 `(ref ±X.XX)` 부기 (예: `47.56% (ref +0.42)`).
- delta ≥ +5%p 또는 ≤ −5%p → §0-6 (c) HISTORY.md `[eval]` 엔트리에 reference 불일치 명시 + 사용자에게 즉시 보고 (학습/eval 환경 차이 의심).
- reference 셀이 `-`이면 그 셀에 대한 비교는 건너뜀 (조용히 통과, 알람 없음).

추가 입력 방법: 사용자가 새 reference 값을 알려주면 위 표의 해당 셀을 `-`에서 `NN.NN%`로 교체 (atomic 편집). 가이드 문서의 표는 에이전트가 자동으로 안 건드린다 — 사용자 명시적 요청 시에만 갱신.
