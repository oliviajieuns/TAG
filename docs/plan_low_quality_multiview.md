# TADS 재설계: Reliability × Learnability × Alignment Multi-View Fusion for Low-Quality Instruction Data

**Status:** 설계 문서. 이 문서의 내용은 아직 구현되지 않음.
**Target:** Elsevier *Information Fusion* SI — "Multi-view Fusion and Learning on Low-quality Data" (마감 2026-08-30).
**Supersedes:** 기존 layer-as-view eigengap fusion 계획. 이번 방향은 signal-level 3-view(신뢰도/학습가능성/정렬)로 전환하며, hidden-state geometry(정렬)는 현행 유지. eigengap 층별 가중은 이번 사이클에서 제외(선택적 future work).

핵심 전환 논리: entropy는 불확실성 지표라서 "깨끗하지만 어려운 데이터"와 "noisy 데이터"를 구분하지 못한다. 따라서 불확실성을 품질로 간주하지 않고 다음 세 관점을 분리한다.

1. **Reliability (Q)** — response가 instruction과 일치하고 완전한가 (노이즈 대응 관점)
2. **Learnability (D)** — 어렵지만 실제로 학습이 진행되는 샘플인가 (학습 난이도 관점)
3. **Alignment (A)** — 현재 checkpoint의 hidden-state geometry와 정렬되는가 (기존 유지)

세 view는 **서로 다른 정보원**에서 나온다: Q는 counterfactual instruction에 대한 별도 forward pass, D는 refresh 간 loss dynamics, A는 층별 hidden-state 기하. 기존 signal-as-view 안(loss/entropy/align)이 "한 softmax의 통계 두 개"라는 비판을 받았던 것과 달리, 이 구성은 multi-view 주장이 실제로 성립한다.

---

## 0. 현재 코드 상태 (검증 완료, 2026-08-11 기준 branch `claude/tads-multiview-fusion-journal-3q6wyt`)

- **점수식** (`tads/core/selector.py::collect_episode`, L272–301):
  `s_i = R_i · a_i · (1 + λ·Align_i)`
  - `R_i = w·r_loss_i + (1−w)·r_entropy_i`, `w = Var(r_loss)/(Var(r_loss)+Var(r_entropy)+ε)` — dataset-level로 1회 계산 (L259–270)
  - `a_i` = PPO actor action ∈ [0,1] (`tads/core/agent.py`), 매 epoch `agent.update`로 PPO 학습 (`tads/pipelines/selection.py` L174–181)
  - `Align_i` = 층별 스트리밍 정렬 `Σ_l ⟨Δh_l, v_l⟩` 후 min-max 정규화 (selector.py L180–201, L276–287)
- **Reward** (`tads/core/reward.py`): 샘플별 response-token 평균 CE(`r_loss`)와 평균 predictive entropy(`r_entropy`)를 per-sample 청크로 계산.
- **루프 구조** (`tads/train.py` L442–476): epoch(=refresh)마다 `select_indices` → top-K subset → `sft_one_epoch`. 기본 `train_epochs: 3`, `selection_ratio: 0.5` (main 실험은 10/50%).
- **체크포인트** (`tads/train.py` L491–518): epoch마다 model/optimizer/scheduler/agent/anchor 저장, auto-resume 지원. → loss history 저장 추가가 자연스럽게 붙는 구조.
- **데이터** (`tads/data/alpaca.py`, `sft_prompts.py`): raw record는 `{instruction, input, output}` 스키마, `ALPACA_DATA_FILES`로 로컬 json/parquet 주입 가능 → **오염 데이터를 파일로 생성해서 주입하면 학습 파이프라인 수정이 거의 불필요**.
- **중요한 기존 불일치**: `r_loss`는 매 epoch 전 샘플에 대해 이미 계산되고 있으나 epoch 간에 버려진다(learnability에 필요한 재료가 공짜로 존재). PPO action `a_i`는 논문의 deterministic scoring 식과 불일치 상태 — 이번 개정에서 반드시 통일해야 함(§3).

---

## 1. 새 점수 설계

### 1.1 Reliability gate Q_i (정적, base checkpoint에서 1회)

**Counterfactual instruction fidelity:**

```
ΔL_i = L(y_i | x_i⁻) − L(y_i | x_i)
Q_i  = rank(ΔL_i) ∈ [0, 1]
```

- `x_i⁻`: 같은 length-bucket(response 토큰 수 기준 quantile bucket) 내 derangement로 뽑은 **의미상 무관한 instruction**. length-bucket 매칭은 길이 confound(RECOST가 지적한 response-length 편향) 차단용.
- 해석: 올바른 instruction이 response 예측을 실질적으로 개선하면(ΔL 큼) 신뢰 가능. mismatch/무관/노이즈 response는 어떤 instruction을 줘도 loss가 비슷 → ΔL≈0 → Q 낮음.
- 비용: pool 전체에 대한 forward 1회 추가(= `collect_episode` 1회 분량, epoch 1에서만). 결과를 `output_dir/reliability_cache.pt`에 캐싱하고 checkpoint resume 시 재사용.
- 구현: counterfactual 쌍은 tokenize 단계에서 `tokenize_alpaca(instruction=x⁻, output=y_i)`로 두 번째 tokenized dataset을 만들면 기존 인프라 그대로 재사용 가능.

**Completeness 항 c_i (truncation 대응, 순수 데이터 레벨, forward 불필요):**

```
c_i = 1.0        response가 EOS로 끝나고 문장이 완결됨
c_i = c_trunc    (기본 0.2) labels가 max_seq_len에서 잘렸거나 문장 중간에서 끊김
```

판정: (a) 마지막 non-pad label 토큰이 EOS인가, (b) raw text가 종결 문장부호/코드블록 닫힘으로 끝나는가. 두 heuristic의 AND/soft 조합.

**최종 gate:**

```
G_i = (Q_i · c_i + ε)^γ      (기본 γ=1, ε=0.01)
```

### 1.2 Learnable difficulty D_i^t (동적, 매 refresh)

`L_i^t` = epoch t의 `all_r_loss` (이미 계산됨, selector.py L256).

```
t = 1:   D_i^1 = rank(L_i^1)                            # history 없음
t ≥ 2:   P_i^t = rank([L_i^{t−1} − L_i^t]_+)            # learning progress
         D_i^t = rank(L_i^t) · (η + (1−η)·P_i^t)        # 기본 η = 0.5
```

- 해석: loss가 높아도 **직전 refresh 대비 감소 중**이면 "어렵지만 학습 가능". loss가 계속 높고 진전이 없으면 noise 가능성 → 하향.
- `train_epochs=3`이면 progress는 2회만 계산되므로 EMA 없이 단순 epoch-to-epoch delta 사용(현재 설정에 맞는 최소 설계). 오염 실험용 light 구성은 `train_epochs: 5`로 늘려 dirty-fraction 곡선의 해상도를 확보 권장.
- entropy는 **주 점수에서 제거**하고 diagnostic 로그 + ablation arm으로만 유지.
- 구현: 직전 epoch loss 벡터를 `epoch_N/loss_history.pt`로 저장 (기존 checkpoint 저장 블록 train.py L491–에 1줄 추가), resume 시 로드.

### 1.3 Alignment A_i^t — 현행 유지

selector.py의 스트리밍 multi-layer 정렬 + min-max 정규화 그대로. 변경 없음.

### 1.4 최종 점수 (gated multiplicative fusion)

```
S_i^t = G_i · D_i^t · (1 + λ·A_i^t)
```

- **Q를 additive가 아닌 gate로**: 더하기 방식은 noisy sample의 높은 loss가 낮은 reliability를 상쇄한다. 곱하기 gate는 저신뢰 샘플의 진입 자체를 막는다.
- 논문 한 줄 요약: *"Rather than treating uncertainty as data quality, we disentangle sample reliability, learnable difficulty, and checkpoint-dependent representation alignment, and fuse them via a quality-gated multiplicative rule."*

### 1.5 Duplicate 처리 — reward 밖에서

중복은 점수로 풀지 않는다. instruction 텍스트에 MinHash/shingle 기반 near-duplicate clustering을 적용하고, **top-K 선택 시 cluster당 최대 1개**(greedy: cluster 내 최고 S 샘플만 허용). `tads/core/dedup.py` 신규, top-K 선택 직전에 삽입 (selector.py L305–307 자리).

---

## 2. PPO action 처리 (공저자 결정 필요, 권장안 명시)

현재 `a_i`가 점수에 곱해지지만 논문 서술은 deterministic scoring — 불일치. 권장:

- **주 경로: deterministic** — `score_mode: deterministic`에서 `S_i = G·D·(1+λA)`만 사용, PPO agent 미사용. multi-view fusion 서사가 깔끔해지고 코드/논문 불일치가 해소됨.
- **legacy 경로 보존** — `score_mode: legacy`가 현행 `R·a·(1+λA)`를 비트 동일하게 재현(회귀 테스트로 고정). old-TADS baseline arm이 이 경로를 그대로 사용.

---

## 3. 저품질 데이터 생성 파이프라인

신규 모듈 `tads/data/corruption.py` + 스크립트 `scripts/make_corrupted_pool.py`. raw Alpaca record(`{instruction, input, output}`)에 대한 결정적(seeded) 변환으로 오염 pool JSON과 **manifest**(`corruption_manifest.json`: index → {type, params, source, dup_cluster})를 생성. manifest가 ground-truth이므로 Dirty@K/AUPRC를 정확히 측정 가능. 생성된 파일은 `ALPACA_DATA_FILES`로 주입 → **학습 파이프라인 수정 불필요**.

| 유형 | 변환 | 탐지 담당 신호 |
|---|---|---|
| T1 instruction–response mismatch | 같은 length-bucket 내 response derangement swap | Q (counterfactual fidelity) |
| T2 noisy response | word dropout p≈0.15 + 인접 word swap + 확률적으로 타 response 문장 삽입 | Q 낮음 + D의 무진전(persistent high loss) |
| T3 truncated response | 단어 기준 U(30%,70%) 지점 절단, 종결부호 제거 | c_i (completeness) |
| T4 duplicate instruction | 5% instruction을 3–5배 복제(공백/대소문자 jitter 포함), manifest에 cluster id | dedup 모듈 (reward 외부) |
| T5 wrong answer | 검증 가능한 subset(GSM8K-train 1–2k 주입)의 최종 답 교란 | Q + D; 이 subset에서만 answer-accuracy 평가 |
| T6 source imbalance | Alpaca-GPT4 + 저품질 출처(Dolly-15k 등) 혼합, source별 오염률 상이, **seed마다 오염 source 교체** | 전체 파이프라인 (source-wise selection rate 분석) |

**Pool 구성:**
- `per-type-20` ×5 (T1–T3, T5 각 20% 단독; T4는 별도) — 유형별 진단용
- `composite-{10,20,40}` — T1/T2/T3/T5 균등 혼합
- `source-skewed-20` — T6

---

## 4. 실험 설계

### Phase A — Forward-only 진단 (저비용, 최우선)

base checkpoint에서 pool 전체 점수 계산만 수행(학습 없음). Qwen2.5-0.5B로 반복 → LLaMA-2-7B로 확인.

- 비교 신호: `r_entropy`, `r_loss`, old composite `R`, `Q`, `c`, `D^1`, `S^1`
- 지표: **Dirty@K** (K = 10%/50% top-K 내 오염 비율), **AUPRC**(clean vs dirty), 유형별 recall
- 목적: "entropy top-K가 오염 데이터를 얼마나 집어오는가"를 정량화 — motivation figure이자 새 reward의 direct 근거. RECOST의 entropy/NLL 한계 보고와 연결.

### Phase B — End-to-end (본 실험)

- **Composite-20**: {Random, Full-polluted, Data Agent(=old reward), TADS-old(legacy), TADS-new, Oracle-clean} × 3 seeds. 동일 corruption manifest 공유. LLaMA-2-7B(flagship) + Qwen2.5-0.5B(반복용).
- **Source-skewed-20**: {Random, TADS-old, TADS-new} × 3 seeds.
- 평가: 기존 8-task eval 그대로 + **refresh별 selected-dirty-fraction 곡선**(핵심 그림 — static quality filtering과 구별되는 training-adaptive 강점) + T5 subset answer-accuracy + duplicate-adjusted diversity.
- light 구성은 `train_epochs: 5`로 곡선 해상도 확보.

### Phase C — 현실성 보강 (게이트드, 일정 여유 시)

- 수동 검수 300–500 샘플(선택/기각 샘플의 실제 품질 라벨링) 또는 Donkii 류 실제 오류 annotation benchmark 1개.
- synthetic-only의 약점 보완. EMNLP 2025 "noise consistency + hidden-state diversity" 계열과의 차별점은 **online learning progress + checkpoint-dependent geometry**임을 명시.

---

## 5. 구현 계획 (파일 단위, 우선순위 순)

1. **점수 경로 정리 + 회귀 테스트** (~1일) — `selector.py`에 `score_mode: {legacy, deterministic}` 도입, legacy가 현행 선택 결과와 비트 동일함을 고정 seed로 assert (`tests/test_selector_regression.py`). 모든 후속 작업의 전제.
2. **`tads/data/corruption.py` + `scripts/make_corrupted_pool.py` + manifest + 단위 테스트** (~1.5일) — T1–T6 변환, seed 결정성 테스트.
3. **`tads/core/reliability.py`** (~1.5일) — counterfactual 쌍 생성(length-bucket derangement), forward 1회로 ΔL 계산, rank 정규화, completeness c_i, `reliability_cache.pt` 캐싱/resume.
4. **Learnability** (~0.5일) — selector.py에 loss-history 전달, `D_i^t` 계산(~30 LOC), `loss_history.pt` 저장/복원.
5. **새 점수 결합 + config** (~0.5일) — `configs/methods/tads_mvf.yaml`: `{score_mode: deterministic, gamma, eta, lam, c_trunc, eps}`.
6. **`tads/core/dedup.py`** (~1일) — MinHash near-dup clustering + cluster-constrained top-K.
7. **`scripts/score_pool.py`** (~1일) — Phase A forward-only 실행기: manifest를 읽어 Dirty@K/AUPRC/유형별 recall 산출.
8. **오염 실험 config** (~0.5일) — `configs/experiments/lowq/…` (light 0.5B 5-epoch + 7B flagship).

합계 ~7.5일 엔지니어링. Phase A는 5번까지 완료되면 즉시 실행 가능.

---

## 6. 리스크 및 주의점

- **Q의 길이 편향**: counterfactual 대조에서 length-bucket 매칭 + rank 정규화 필수. ablation으로 bucket 미적용 시 AUPRC 열화를 보여주면 오히려 설계 근거가 됨.
- **Q는 정적 gate** (base checkpoint 1회): mismatch 탐지는 정적이고, 적응성 서사는 D와 A가 담당. "static reliability gate + dynamic learnability/alignment fusion"으로 명시적으로 서술 — refresh마다 Q를 재계산하면 비용이 배가되고 얻는 것이 적음.
- **3 epochs 한계**: progress 신호가 2회뿐 → EMA 등 복잡한 설계 금지, 단순 delta 유지. 오염 실험 light 구성만 5 epochs.
- **clean-data 성능 overclaim 금지**: 기존 λ-ablation의 inverted-U가 보여주듯 현행 설계는 clean data에서 이미 잘 튜닝됨. 새 설계의 주장은 "low-quality pool에서의 강건성"이지 "clean에서 strict win"이 아님. clean Alpaca-GPT4에서 TADS-new ≈ TADS-old(동등성) 확인 실험 1개 필요.
- **T4/T6는 reward로 풀지 않음**: duplicate는 dedup 모듈, source imbalance는 분석 축. reward에 욱여넣으면 각 view의 담당이 흐려짐.
- **PPO 제거는 공저자 합의 사항**: deterministic 주 경로 권장이지만 방법의 정체성 변화이므로 착수 전 확정.
- **dual-submission**: CIKM 철회 결정은 여전히 선결 조건(기존 계획 문서와 동일).
