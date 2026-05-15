# LLaMA-2-7B 튜닝 — NAIT 논문 수치 매칭 + 최대화

이 브랜치(`llama-tune`)는 **Llama-2-7B 실험만**(qwen/mistral/deepseek 영향 없음)
파라미터를 조정해 NAIT (ICLR 2026, Chen et al.) 논문 Table 2의 baseline 행에 더
가까이 가도록 튜닝한 버전이다.

다른 모델 config는 손대지 않았으므로, 이 브랜치에서 학습한 llama2 결과만이
영향 받음. 머지 후 다른 모델 결과는 dev/main과 동일.

---

## 1. 우리 실험 ↔ 논문 행 매핑

논문 Table 2 (Llama-2-7B, Alpaca-GPT4, Full FT) 발췌:

| Sys ID | Method | MMLU | MMLU-Pro | GSM | SVAMP | H-Eval | MBPP | TydiQA | XQuAD | BBH | **AVG** | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **01** | Alpaca-GPT4 (Peng et al., 2023) | **46.87** | 21.89 | **14.63** | 39.00 | **27.87** | 51.58 | **39.48** | 42.99 | **39.94** | 36.03 | — |
| 02 | LIMA (Zhou et al., 2023) | 45.20 | 23.04 | 15.76 | 37.67 | 27.75 | 46.56 | 44.92 | 44.72 | 39.91 | 36.17 | +0.39% |
| 03 | 01 + AlpaGasus (Chen et al., 2024) | 43.21 | 21.96 | 13.34 | 36.67 | 23.94 | 46.08 | 44.70 | 46.84 | 39.91 | 35.18 | −2.34% |
| 04 | 01 + Q2Q (Li et al., 2024) | 46.73 | 21.50 | 14.50 | 35.00 | 25.19 | 44.97 | 44.41 | 48.44 | 40.34 | 35.68 | −0.98% |
| 05 | 01 + SelectIT (Liu et al., 2024b) | 47.90 | 22.86 | 15.40 | 41.11 | 27.92 | 49.47 | 43.91 | 45.56 | 40.33 | **37.16** | +3.15% |
| **06** | 01 + Random Baseline | **47.14** | 21.43 | **14.13** | 35.67 | **25.55** | 47.35 | **44.16** | 46.56 | **39.21** | 35.69 | −0.94% |

**우리 실험과의 매핑**:

| 우리 config | 논문 행 | 비고 |
|---|---|---|
| `main_7b/llama2/full_100` | **01 Alpaca-GPT4** | Full Alpaca-GPT4 100% (paper-matching baseline, **upper bound**) |
| `main_7b/llama2/random_10` | **06 01 + Random Baseline** | 10% random 샘플링 (paper-matching baseline, **lower bound**) |
| `main_7b/llama2/data_agent_10` | (논문엔 없음) | TADS λ=0 동등 — random보다 ↑, full에 −2%p 이내 권장 |
| `main_7b/llama2/tads_10` | (논문엔 없음) | TADS proposed — random보다 ↑ 필수, SelectIT 수준(+3%p) 목표 |

우리 평가 벤치 5종 (MMLU, GSM8K, HumanEval, TyDiQA, BBH)으로 5-bench AVG 계산 시
**기대 수치**:

| 셀 | MMLU | GSM | H-Eval | TyDiQA | BBH | **5-bench AVG** |
|---|---|---|---|---|---|---|
| `llama2/full_100` (= row 01) | 46.87 | 14.63 | 27.87 | 39.48 | 39.94 | **33.76** |
| `llama2/random_10` (= row 06) | 47.14 | 14.13 | 25.55 | 44.16 | 39.21 | **34.04** |
| `llama2/data_agent_10` (target) | ≥ 47 | ≥ 14 | ≥ 26 | ≥ 44 | ≥ 39 | ≥ 34.0 |
| `llama2/tads_10` (target) | ≥ 47 | ≥ 14 | ≥ 27 | ≥ 44 | ≥ 39 | ≥ **34.5** (random 명확 초과 + alpaca 5-bench 근접/초과) |

(논문 9-bench 기준 random AVG가 alpaca보다 낮지만, 우리 5-bench 부분합으론 TyDiQA 영향으로
random AVG가 alpaca를 약간 상회. 이건 벤치 부분집합 효과지 알고리즘 우열 아님.)

---

## 2. 적용한 파라미터 변경 (llama 한정)

### 2-1. `configs/models/llama2-7b.yaml`

| 변경 | Before → After | 이유 |
|---|---|---|
| `prompt_style` | `llama_user_assistant` → `alpaca_default` | 논문 row 01은 **Stanford Alpaca template** (`### Instruction:` / `### Response:`). `<|user|>`/`<|assistant|>` 템플릿은 base Llama-2에 정의돼 있지 않은 special token이라 SFT 가치가 떨어짐 — 가장 큰 잠재 개선 포인트. |
| `attn_implementation` | (none) → `flash_attention_2` | 동일 품질에 ~30% 속도 향상. flash-attn 미설치 시 loader가 sdpa로 자동 fallback. |
| `learning_rate` | `2.0e-5` (그대로) | NAIT Table 8 spec — 변경 없음. |
| `max_seq_len` | `512` (그대로) | NAIT Table 8 spec — 변경 없음. |

### 2-2. `configs/experiments/main_7b/llama2/*.yaml` (모든 4개 셀)

| 변경 | Before → After | 이유 |
|---|---|---|
| `train_epochs` | `3` → `4` | 논문은 3 epoch이지만 우리 실측에서 epoch 3 종료 시점 loss가 아직 떨어지는 추세 → 1 epoch 추가로 마지막 ~5% 수렴분 회수. 5 이상은 overfit 위험. |
| `weight_decay` | `0.1` (그대로) | 변경 안 함 — 작은 SFT 데이터셋엔 0.0이 좋다는 견해도 있으나 기본 stick. |

### 2-3. `tads_10` 추가 변경 (proposed method 강화)

| 변경 | Before → After | 이유 |
|---|---|---|
| `anchor.max_samples_for_pca` | `1024` → `2048` | PCA 추정 분산 ↓, anchor 방향 안정성 ↑ → alignment 노이즈 감소 |
| `agent.entropy_coef` | `0.01` → `0.02` | PPO actor의 탐색량 살짝 ↑ — 수렴 후 한 쪽으로 너무 collapsed되는 패턴 회피 |

### 2-4. `data_agent_10` 추가 변경

| 변경 | Before → After | 이유 |
|---|---|---|
| `agent.entropy_coef` | `0.01` → `0.02` | tads_10과 동일 이유, baseline 비교 공정성 |

---

## 3. 다른 모델 (qwen/mistral/deepseek)에 영향 없음

- `configs/models/llama2-7b.yaml`만 수정 → 다른 모델 config는 그대로
- `configs/experiments/main_7b/llama2/*.yaml`만 수정 → 다른 model 디렉토리 영향 없음
- `configs/base.yaml`, `configs/methods/*`, `configs/modes/*`는 미변경 → 공통 디폴트 유지

---

## 4. 학습/평가 명령

새 run-layout (`feature/run-layout` 머지됨) 사용:

```bash
# 학습 — 자동 timestamp tag, 이전 결과 보존
torchrun --nproc_per_node=4 -m tads.train \
    --config configs/experiments/main_7b/llama2/full_100.yaml

torchrun --nproc_per_node=4 -m tads.train \
    --config configs/experiments/main_7b/llama2/random_10.yaml

torchrun --nproc_per_node=4 -m tads.train \
    --config configs/experiments/main_7b/llama2/data_agent_10.yaml

torchrun --nproc_per_node=4 -m tads.train \
    --config configs/experiments/main_7b/llama2/tads_10.yaml

# 평가 — _latest 자동 사용
python -m tads.eval \
    --config configs/experiments/main_7b/llama2/full_100.yaml \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh

# 또는 한 번에 4 개 모두
MODELS=llama2 bash scripts/run_eval_main_7b.sh --gpus 0
```

---

## 5. 결과 합격 기준

| 셀 | 합격 기준 (5-bench AVG) | Critical 알람 |
|---|---|---|
| `llama2/full_100` | ≥ 33.0 (논문 row 01의 33.76 대비 −0.8%p 이내) | < 32 → 학습 자체 의심 |
| `llama2/random_10` | ≥ 33.5 (논문 row 06의 34.04 대비 −0.5%p 이내) | < 33 → 셀렉션 시드/데이터 의심 |
| `llama2/data_agent_10` | ≥ random_10 (NAIT보다 ↑) | < random_10 → **빨간불** |
| `llama2/tads_10` | ≥ data_agent_10, 가능하면 SelectIT(+3.15%p)에 근접 | < random_10 → **빨간불** |

**기대 효과 합산** (대략):
- prompt_style 변경: AVG +1~3%p (가장 큰 영향)
- train_epochs 3→4: AVG +0.3~0.5%p
- agent/anchor 튜닝 (tads only): AVG +0.2~0.5%p
- 합계: 우리 baseline보다 **+1.5~4%p** 정도 상승 기대

---

## 6. 되돌리기 (튜닝 실패 시)

```bash
# 한 셀만 되돌리기 — 이전 run으로 _latest 재지정
ln -sfn runs/<old_tag> ${OUTPUT_ROOT}/main_7b/llama2/<method>/_latest

# 또는 git
git checkout dev -- configs/models/llama2-7b.yaml configs/experiments/main_7b/llama2/
```
