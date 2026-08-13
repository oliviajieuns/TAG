# TAG → Information Fusion SI 단일 실행 계획 (v3.1)

**Status:** ACTIVE — 유일한 실행 문서 (2026-08-13 v3.1: 원고 프레이밍을
TAG로 확정, §0·§2.0·§8·§11 갱신. 실험/이론/일정은 v3 그대로).
**Target:** Elsevier *Information Fusion* SI — "Multi-view Fusion and Learning
on Low-quality Data: Foundation Models in Theories, Algorithms and
Applications" (마감 **2026-08-30, D-18**). CIKM은 종료, dual-submission 제약
없음. 증거 위생 규칙: [`cikm-review-revision-audit.md`](cikm-review-revision-audit.md) §2.

**v2 → v3 개정 근거:** 12-에이전트 적대적 검증(스코프/융합/실험/이론 리뷰어
4인 전원 major-revision + 저널·게스트에디터·선행연구 정찰 5인 + 인용 검증 3인,
36편 전부 실존 확인). 핵심 발견 두 가지가 v3를 강제했다:

1. **v2 게이트는 기본값에서 수학적으로 깨져 있었다** (융합·이론 리뷰어가
   독립적으로 동일 계산): clean-보정된 σ 게이트의 억제비는 최대
   (0.5/0.9)^γ ≈ 0.56인데 D 인자의 스팬은 (1+ε)/ε ≈ 101× — 오염 고손실
   샘플이 깨끗한 쉬운 샘플을 ~57× 이겨 "non-compensatory"가 거짓이 된다.
   → §2의 re-zeroed gate + d_floor로 수정 (구현 완료).
2. **스코프 방어의 열쇠가 이미 저널 안에 있다**: *"Multi-view fusion for
   instruction mining of large language model"* (Information Fusion, vol.
   110, art. 102480, 2024) — 이 저널이 텍스트-only LLM instruction 선택의
   multi-view 프레이밍을 이미 게재했다. "최초 multi-view 선택" 주장은 폐기;
   이 논문의 **training-adaptive, corruption-robust, 비보상적 후속**으로
   포지셔닝한다.

**v3 구현 상태 (2026-08-12): §1 P0 전부, §2 점수 v3, §3 오염 확장(T1b/T7/
K-counterfactual), 표 파이프라인 v2, arm config 체계 — 코드 반영 및 테스트
green (core 59 + corruption 19 + table 12 pass).**

**v3 → v3.1 개정 근거 (2026-08-13):** 원고의 진입 프레임을 "multi-view
fusion 방법론"에서 **"training-adaptive selection이 저품질 pool에서 깨진다 →
reliability를 veto로"** 로 재배치. 방법의 셀링 포인트가 세 뷰의 존재가 아니라
**비보상성(non-compensatory) 그 자체**임이 v3 검증에서 드러났고(§7-1의 γ*
정리가 실질 기여), 저널 스코프는 fusion 어휘를 §0 그대로 유지하는 것으로
충분히 방어된다. 부수 결정 두 가지: (a) v3의 "trajectory 전면 삭제"를 **철회**
— dynamic 두 뷰가 trajectory-anchored selection에서 그대로 상속된다는 점이
"게이트만 추가했다"는 최소-변경 주장의 근거이므로 명칭을 유지하되 CIKM
Reviewer 1의 명칭-수식 불일치 지적(audit §1-1)을 §8 명명 규칙으로 봉쇄한다.
(b) 이름 확정: **TAG** = **T**raining-**A**daptive data selection with a
reliability **G**ate.

---

## 0′. 확정 제목·초록 (사전 등록 — 이 문서가 그 기록)

**Title:** *Training-Adaptive Data Selection with a Reliability Gate:
Multi-View Fusion for Low-Quality Instruction Data*

**Keywords:** Multi-view fusion; Low-quality data; Instruction tuning;
Training-adaptive data selection; Reliability; Trajectory anchoring;
Hidden-state representations; Foundation models

**Abstract (제출본 기준 문구, `\STATUS{}` 자리는 Gate A′/B 결과로 치환):**

> Data selection for instruction tuning is increasingly
> *training-adaptive*: an example's usefulness changes as the model moves
> through its fine-tuning trajectory, and dynamic selectors re-score
> candidates accordingly. But real instruction pools are low-quality —
> mismatched pairs, truncated responses, and subtly wrong answers are
> common — and the difficulty–uncertainty signals that drive dynamic
> selection actively *prefer* such corruption, because broken samples look
> hard. We propose TAG (**T**raining-**A**daptive data selection with a
> reliability **G**ate), which casts the selector as a non-compensatory
> fusion of three model-intrinsic evidence views. A static *Reliability*
> view classifies each sample once, before training: a counterfactual
> likelihood contrast — how much the true instruction improves the
> response's likelihood over an unrelated one — converted into a
> zero-anchored gate G_i ∈ [0,1] calibrated on a clean reference pool, and
> cached at no per-refresh cost. Two dynamic views are inherited unchanged
> from trajectory-anchored selection: a difficulty–uncertainty carrier
> R_i^(t), and a trajectory-anchor alignment factor derived from
> checkpoint-to-checkpoint shifts in hidden representations, supported by a
> local stability analysis. The fused score is simply
> s_i^(t) = G_i · R_i^(t) · (1 + λ·ã_i^(t)): because both dynamic factors
> are bounded, a zeroed gate cannot be compensated by any amount of
> difficulty or alignment evidence — reliability is a veto, not a vote —
> while selection among gated samples remains fully adaptive to the
> training trajectory. On instruction pools with typed,
> manifest-verified corruptions (mismatched, noisy, truncated,
> wrong-answer, duplicated), `\STATUS{detection and end-to-end SFT results
> pending: Dirty@K/AUPRC and seed-paired benchmark comparisons vs.
> entropy-, perplexity-, and IFD-based selection}`; in prior same-protocol
> comparisons the underlying training-adaptive selector already
> outperformed strong static and dynamic baselines while reducing
> selection-and-tuning cost by 33% relative to a Reward+PPO selector. TAG
> requires no external judges, reward models, or PPO-style selector
> training.

**초록에 걸린 검증 부채 3건 (제출 전 반드시 해소 — §11 리스크와 연동):**

1. `\STATUS{}` — Gate A′(§4)·Gate B(§9) 결과로 치환. 미해소 시 제출 불가.
2. **"33% cost reduction vs. Reward+PPO"** — 이 수치의 출처는 CIKM 원고다.
   audit §2.1의 "기존 CIKM 숫자 인용 금지"에 정면으로 걸린다. **선택지는 둘
   뿐**: (a) 새 프로토콜(make_table_v2, sealed run)로 cost를 재측정해 새 숫자로
   교체, (b) 문장 삭제. 옛 숫자를 그대로 싣는 경로는 없다. D5까지 결정.
3. **"inherited unchanged"** — 현재 코드의 dynamic carrier는 초록의 R와
   다르다(§2.0). 코드를 초록에 맞추거나 초록을 코드에 맞추거나, 둘 중 하나.

---

## 0. 포지셔닝 (스코프 게이트키퍼 종합)

**뷰 명명 — 원고 표층(초록/intro/method 서두)과 내부 기호의 대응:**

| 원고 명칭 (TAG) | v3 내부 명칭 | 기호 | 획득 시점 |
|---|---|---|---|
| Reliability view (static gate) | Consistency view | `G = (Q·c+ε)^γ` | 1회 (base ckpt, 캐시) |
| Difficulty–uncertainty carrier | Dynamics view | `R^(t)` / `D′` | refresh마다 |
| Trajectory-anchor alignment | Geometry view | `ã^(t)` | epoch마다 |

내부 문서·코드·config 키는 Q/D/A와 `score_mode: mvf`를 그대로 유지한다(개명
diff가 회귀 위험만 만든다). 원고에서만 위 좌열을 쓰고, method 서두에 이 대응
표를 1개 싣는다 — **CIKM Reviewer 1의 "명칭이 수식과 불일치" 지적(audit §1-1)
재발 방지선이 바로 이 표다.**

**한 문단 요약:** instruction–response pair는 저품질 조건(mismatch=뷰 간
정렬 오류, truncation=불완전 뷰, noise)이 실제로 발생하는 2-뷰 데이터다.
training-adaptive 선택기는 이 pool에서 **구조적으로 오작동한다** — 난이도·
불확실성 신호가 곧 "망가진 샘플일수록 높다"이기 때문이다(초록 1–2문장, F2가
그 실측). TAG는 선택을 세 개의 **model-intrinsic evidence view** —
Reliability(G = Q·c, counterfactual 교차-뷰 forward, **static**),
difficulty–uncertainty carrier(R^(t), refresh 간 loss 궤적), trajectory-anchor
alignment(ã^(t), 층별 hidden-state 기하) — 의 **보정된 비보상적
(log-opinion-pool형) 융합**으로 재기술한다. dynamic 두 인자가 유계이므로
G=0은 어떤 난이도·정렬 증거로도 되살아나지 않고(veto), 게이트를 통과한
샘플들 사이의 순위는 완전히 trajectory-adaptive하게 남는다. 게이트 강도는
측정된 pool 오염도와 뷰 안정성에 보정으로 연동된다.

**뷰 명명:** 원고 표층은 Reliability / difficulty–uncertainty carrier /
trajectory-anchor alignment, 내부 기호·코드는 Q(·c), D, A 유지 — 대응표는
위 §0 서두. 원고 안에서 두 어휘를 섞지 않는다.

**"What constitutes a view" 서브섹션 (필수, method 서두):**
(a) 데이터 레벨: instruction과 response는 한 태스크의 이질적 2-뷰 — T1
mismatch가 곧 CFP의 misaligned view, T3 truncation이 incomplete view;
(b) evidence 레벨: 세 뷰는 생성 과정이 서로 다른 정보원(별도 counterfactual
forward / 시간축 loss dynamics / hidden-state 기하)이고 **획득 스케줄까지
이질적**(1회 / refresh / epoch) — co-training 계열의 constructed-view 전통
인용; (c) in-journal 선례(Huang et al. 2024) 명시.

**CFP 토픽 주장 (이 순서·이 강도로, intro에 명시):**
- **topic 2** (foundation-model representations as views) — 최강 라이선스
- **topic 11** — 반드시 "**quality-adaptive fusion**"으로 표현: 게이트
  강도가 오염도에 보정으로 연동(clean에서 무력 — 실측 표로 증명) + λ_t
  anchor-stability 적응(구현 완료) + MVF vs MVF-static arm이 직접 증거.
  "adjusts the fusion mechanism"이라는 문구는 금지 (룰 형태는 고정이므로).
- **topic 7** (trusted/explainable) — 뷰별 기각 사유 attribution figure +
  per-type recall 표 + Q calibration(reliability diagram, Phase C)
- **topic 6** (imperfect supervision) — noisy response = noisy supervision
- **topic 3** (incomplete) — **partial-view availability arm을 추가한
  뒤에만** 주장 (§4; Wen의 홈필드라 증거 없는 주장은 역효과)
- **topic 4** — T1b를 cross-view correspondence corruption으로 1문장
- topic 5 (imbalanced): 주장하지 않음 — T6은 view가 아니라 source 불균형.
  taxonomy 매핑 테이블에서 연결만 언급.

**오염 → 저품질 분류 매핑 테이블 (intro 초반, 필수):** 게스트 에디터 Jie
Wen 공저 서베이 *"Multimodal Fusion on Low-quality Data"* (arXiv 2404.18947,
Inf. Fusion)의 4분류를 그대로 채택 — T2/T5/T7→noisy, T3→incomplete,
T1a/T1b→misaligned(cross-view correspondence), T6→imbalanced(분석 축),
학습 중 score drift→quality-varying.

---

## 1. P0 — 실험 전 수정 [구현 완료 2026-08-12]

1. **표 파이프라인 v2** — `scripts/make_table_v2.py` + 테스트 12개.
   1 행 = 1 sealed run = 1 checkpoint 구조적 강제(교차-run 집계가 표현
   불가능), seed-macro 집계, paired diff + t-CI + 정확 sign-flip permutation
   + Holm. max-of-each는 어떤 경로로도 재현 불가. 기존 CIKM 숫자 인용 금지
   유지.
2. **Completeness 버그** — `text_complete` 컬럼을 토크나이즈 단계에서 생성
   (`sft_prompts.text_is_complete`: 종결부호/닫힌 코드펜스/숫자 종결), EOS
   토큰 체크는 max_seq_len 절단 감지로 축소. T3 회귀 테스트 포함.
   기존 토크나이즈 캐시는 `TADS_FRESH_DATA_CACHE=1`로 재생성 필요.
3. **Legacy 회귀 테스트** — `tests/test_selector_regression.py`: 논문 수식
   재유도 대조로 legacy 경로 고정 (min-max 정규화 유지 포함).
4. **rank01 midrank** — tie를 pool 인덱스로 깨던 v1 동작이 데이터 순서를
   점수로 승격시키던 버그 수정. `ties="positional"`은 진단용으로만 잔존.
5. **캐시 부재 hard error** — epoch>1에서 reliability 캐시가 없으면
   조용히 잘못된 checkpoint에서 Q를 재계산하는 대신 RuntimeError.
   게이트 설정 변경 시에는 캐시된 loss에서 forward 없이 재계산.

---

## 2. 점수 설계 (MVF v3) [구현 완료]

### 2.0 초록 수식 ↔ 구현 대조 (v3.1 신규 — 제출 전 해소 필수)

초록이 제시하는 융합식과 `score_mode: mvf`가 실제로 계산하는 식:

```
초록:  s_i^(t) = G_i · R_i^(t) · (1 + λ·ã_i^(t))
구현:  S_i^t   = (Q_i·c_i + ε)^γ · (D′_i + ε) · (1 + λ_t·ã_i^t)
        D′_i   = d_floor + (1 − d_floor)·D_i^t
```

대응은 `G_i = (Q_i·c_i + ε)^γ`, `R_i^(t) = D′_i + ε`. 두 군데가 **문자 그대로
같지 않다** — 원고 심사에서 코드를 대조하는 리뷰어가 반드시 짚는 지점이므로
지금 결정한다.

| # | 초록 | 구현 | 상태 |
|---|---|---|---|
| A | `G_i ∈ [0,1]`, 0이면 veto | `(Q·c+ε)^γ`, ε=0.01 바닥 | **표기 문제만.** ε 바닥은 게이트 아래 순위를 정의된 상태로 두기 위한 것이고 억제비 61×는 §7-1에서 명시적으로 계산된다. 초록의 "cannot be compensated"는 γ>γ* 조건부 주장이므로 method에서 ε·γ*를 드러내면 정합. 초록 문구 수정 불필요, **method에 ε 1문장 필수** |
| B | `R^(t)` = difficulty–**uncertainty** carrier, trajectory-anchored selection에서 **inherited unchanged** | `D′` = difficulty-only (`rank01(L^t)`×progress, `[0.5,1]` 압축). entropy는 로깅만 — v3가 "uncertainty ≠ quality"를 근거로 **의도적으로 제거** | **실질 불일치.** 아래 둘 중 하나를 D5까지 선택 |

**B의 선택지 (하나만 고른다):**

- **B1 — 초록을 코드에 맞춘다 (권고).** `R^(t)`를 "difficulty carrier"로
  적고, "inherited unchanged"를 "inherited from trajectory-anchored
  selection, with entropy removed from the carrier"로 바꾼다. 근거는 이미
  있다: entropy는 uncertainty이지 quality가 아니며, 저품질 pool에서 오염
  샘플을 **선호**하는 신호가 바로 그것이다(초록 2문장이 스스로 그렇게
  말한다 — 그 신호를 carrier에 남겨두는 편이 오히려 서사와 충돌). 비용 0,
  회귀 위험 0. entropy-in-carrier는 §4 forward-only ablation 1행으로 측정해
  근거를 붙인다(`R` 신호가 이미 `score_pool.py`에 있으므로 무료).
- **B2 — 코드를 초록에 맞춘다.** `mvf` carrier를 `pool_reward`의
  `w·L+(1−w)·H`로 교체. 유계가 아니므로(§7-1의 비보상 여유가 R의 pool
  스케일에 의존) `rank01` 또는 min-max 유계화가 **반드시** 동반되어야 하고,
  그 순간 §7-1 γ* 정리의 상수와 골든 회귀 테스트를 다시 유도해야 한다.
  D-18 시점에 정리·테스트·전 arm 재실행을 다시 여는 선택이므로 권고하지 않음.

기본값은 **B1**. B2로 갈 경우 §7-1·§9 일정·§5.4 예산을 같은 커밋에서 갱신할 것.

### 2.1 Reliability Q (Consistency view) — zero-anchored calibrated gate (re-zeroed)

```
ΔL_i = L(y_i | x_i⁻) − L(y_i | x_i)
Q_i  = clip(2·(σ(ΔL_i / s) − 0.5), 0, 1)      # ΔL ≤ 0 → 0 (게이트 바닥)
s    = P10(ΔL_clean) / logit(0.8)              # backbone별 1회, clean 참조
```

- v2의 raw σ는 오염 샘플에 0.5를 줘 억제비가 ~2×에 그쳤다(치명 결함 1).
  re-zero 후 (수치는 post-rezero 기준 — raw σ ≥ 0.8은 **보정 목표**이고
  게이트에 들어가는 값은 Q = 2·(0.8−0.5) = 0.6): 오염(ΔL≤0) → ε 바닥,
  clean 90% → Q ≥ 0.6 → 게이트 인자 ≥ 0.61 → **억제 = 0.61/0.01 = 61×**
  vs 보상 최대 = D (1.01/0.51) × A 2 ≈ 3.96 — γ=1에서 여유 61/3.96 ≈
  **15.4×**로 non-compensation 성립.
- 보정: "clean 참조의 90%가 raw σ ≥ 0.8" — clean-equivalence가 구조적으로
  내장. `reliability_ref_file`(clean pool ΔL .pt) 또는 명시적
  `reliability_scale`. 미지정 시 in-pool fallback은 경고와 함께 진단 전용.
- **v1 rank 게이트와 v2 raw-σ 게이트는 ablation arm으로 보존**
  (`reliability_mode: rank`, `reliability_rezero: false`) — v2 역전은
  "multiplicative fusion에서 calibration과 suppression의 긴장" figure로
  판매(리뷰어 제안: 실패 모드 자체가 기여).
- **Evidential-lite (K>1 counterfactual):** `loss_cf (K,N)` →
  Q = mean_k gate 후 `·(1−2·std_k)` dispersion discount. 0.5B에서 K=3
  Phase A로 검증, 이득 있으면 7B는 appendix. heuristic임을 명시.
- 알려진 오탐(열린 지시 ΔL≈0)은 유지 — Phase A 유형별 진단으로 먼저 측정해
  공개 (v2 결정 유지).

### 2.2 Dynamics D — split progress + range compression

```
t ≥ 2:  P̂_i = rank01_{Selected(t−1)}([L^{t−1}−L^t]₊)   (선택군 내부만)
        P̂_i = 0.5                                       (미선택: 증거 없음)
        D_i = rank01(L^t_i) · (η + (1−η)·P̂_i)
융합 시: D'_i = d_floor + (1−d_floor)·D_i,  d_floor = 0.5
```

- split은 v2 결정 유지 (rich-get-richer 차단). `selected_prev`는
  `selected_indices_epoch{t-1}.json`에서 자동 로드.
- **d_floor = 0.5 (v3 신규):** D 인자의 dynamic range를 (1+ε)/(0.5+ε) ≈ 2×로
  압축 — D는 "신뢰 샘플들 사이의 순위 조정자"이지 게이트를 뒤집는 힘이
  아니다. d_floor=0이 v2 역전 ablation.
- midrank가 progress의 0-질량 tie를 값 기준으로 처리 (§1-4).
- `progress_mode: global`(v1)은 detection-레벨 ablation 전용.

### 2.3 Geometry A — pool-CDF 정규화 + 안정성 적응 λ

- MVF 경로의 A 정규화를 min-max → **rank01(pool-CDF)** 로 교체: min-max는
  극단 2개 샘플이 [0,1] 끝점을 고정해 outlier 하나가 전체를 압축(융합
  리뷰어 major). legacy 경로는 min-max 유지(비트 동일 보장).
- **adaptive λ (topic 11 실증, 구현 완료):** `adaptive_lam: true` 시
  λ_t = λ0 · stability_t, stability = eigengap-가중 |cos(v^t, v^{t−1})|.
  "융합은 Geometry 뷰를 그 anchor가 안정적인 만큼만 신뢰한다."
  adaptive-vs-fixed ablation은 캐시 재조합으로 무료.

### 2.4 융합 규칙 — weighted log-opinion pool로 재기술

```
S_i = (Q_i·c_i + ε)^γ · (D'_i + ε) · (1 + λ_t·Ã_i)
log S = γ·log(Qc+ε) + log(D'+ε) + log(1+λ_t·Ã)
```

- **원고 프레이밍 전환 (융합 리뷰어 major):** "multiplicative gate가
  novel"이라 쓰지 않는다 — Kittler 1998 product rule, Genest–Zidek
  log-opinion pool, PoE, Dempster's rule이 전부 곱셈형이다. 대신:
  (i) 융합을 가중 log-opinion pool로 명시하고 (γ,1,~λ)를 pool weight로
  해석, (ii) 기여는 **보정된 zero-anchored 게이트 설계 + γ* 조건 +
  이질적 뷰 획득 스케줄(Q 1회/D refresh/A epoch) + 뷰별 진단**에 건다,
  (iii) TMC/ETMC(대칭적 evidential, conflict 재분배)와 RCML(conflictive
  averaging = 보상적)과의 대비 — "선택 문제에서는 unreliability가
  down-weight가 아니라 **veto**여야 한다"가 인용 가능한 포지셔닝 문장.
- 각 인자의 확률적 독해와 pool-relative(D,A) vs absolute(Q) 구분을 본문에
  1문단 + limitation 2문장.

### 2.5 Dedup — 현행 유지 (v2와 동일)

---

## 3. 오염 생성 (v3) [구현 완료]

| 유형 | 변환 | 역할 |
|---|---|---|
| T1a mismatch (derangement) | 현행 | **sanity 전용** (탐지기와 동일 연산 — 본문에 명시) |
| T1b mismatch (cross-source) | 다른 소스(Dolly)의 response로 교체, length-bucket 매칭, donor 재사용 최소화 | 헤드라인 mismatch |
| T2 noisy / T3 truncated / T5 wrong answer | 현행 | 헤드라인 (T3는 §1-2 수정 후 유효) |
| **T7 fluent-wrong (신규)** | instruct 모델이 생성한 유창하지만 오답/공허한 response (~5%), 2-step: `--emit-fluent-wrong-targets` → `scripts/gen_fluent_wrong.py`(서버, ~3–5 GPU-h) → `--fluent-wrong-file` | **PPL 필터가 못 잡는 유일한 유형 — 융합 필요성의 실증** (실험 리뷰어 치명 지적 2 대응). 길이 보존으로 length-bucket 통계 유지 |
| T4 duplicate / T6 source imbalance | 현행 | dedup 모듈 / 분석 축 |

- composite-20 = T1b/T2/T3/T5/**T7** 균등 (T1a 제외).
- K-counterfactual: `--num-counterfactuals K` → `counterfactual_1..K.json`
  (k=1은 기존과 바이트 동일 — SHA256 회귀 테스트로 증명됨).
- Oracle pool: `scripts/make_oracle_pool.py`.

---

## 4. Phase A — Forward-only 진단 (수정)

비교 신호(전부 `scripts/score_pool.py`에 구현됨): `entropy`, `loss`, `R`,
`legacy_score`, `Q`(v3), `Q_rank`(v1), `gate`, `D`, `mvf_score`(v3),
`mvf_v2`(역전 ablation), `additive`(보상적 대조), **`ppl`, `ifd`**
(`--uncond-loss`, `scripts/compute_uncond_loss.py` 1회 forward로 둘 다).

- **PPL/IFD 행이 필수인 이유 (실험 리뷰어):** "PPL 필터로 충분하다"는
  공격을 가정으로 반박하지 않고 측정으로 반박(또는 유형별로 인정)한다.
  IFD는 Q의 가장 가까운 공개 친척(Li et al., NAACL 2024) — 표 없는 구분
  주장은 통하지 않는다. Superfiltering 행 = 0.5B 모델의 loss로 7B pool
  IFD 계산 (무료).
- **Partial-view availability arm (topic 3 근거, 신규):** 캐시된 Q/D/A
  재조합으로 Q 가용률 p ∈ {0, 25, 50, 100}%(결측은 중립 0.5) Dirty@K/AUPRC
  열화 곡선. e2e는 p=50 1 run만 (0.5B).
- **γ sweep {0.5, 1, 2, 3}** (이론 리뷰어: γ* 조건 검증 겸용), λ ∈ {0, 0.5, 1},
  s 보정 민감도 κ ∈ {0.5, 2} (§7 robustness corollary의 실증 짝).
- clean pool 유형별 Q 분포 진단 유지 (v2).
- **Gate A' (판정 기준 갱신):** composite-20에서 `mvf_score`가
  `entropy/loss/R` **그리고 `ppl`/`ifd`** 대비 Dirty@K·AUPRC 우위 (T7
  포함 시 ppl은 T7에서 실패해야 정상), `Q ≥ Q_rank`, `mvf_score ≥ additive`.
  실패 시 2일 재설계 1회, 재실패 시 중단.

---

## 5. Phase B — End-to-end (수정)

### 5.1 Arms (composite-20)

| # | Arm | 비고 |
|---|---|---|
| 1 | Random | 바닥 |
| 2 | Full-polluted | 무선별 reference |
| 3 | Oracle-clean | 천장 reference (`make_oracle_pool.py`, ratio 0.125) |
| 4 | TADS-legacy | 회귀 테스트로 고정 |
| 5 | TADS-MVF v3 | 제안 |
| 6 | TADS-MVF-static | `mvf.static: true` (구현 완료) — adaptive 분리 control |
| 7 | **IFD top-K** (신규) | static 외부 baseline #1 — uncond loss로 필터한 pool + full 학습. 가장 싸고 가장 요구될 baseline |
| 8 | AlpaGasus 또는 PPL top-K | 외부 judge 가능 여부 D3까지 확인, 불가 시 PPL로 대체 명시 |

- 전 arm이 `configs/experiments/lowq/_shared_light_05b.yaml` 하나에서
  optimizer/precision/scheduler/batch/epochs를 상속 — **arm 파일에는
  method 관련 키만 존재** (CIKM §2.3 재발 방지를 구조로 강제, 구현 완료).
- SelectIT(저장소에 구현 존재, fidelity 이슈 없음)는 여유 시 0.5B 3 runs.
- RobustFT/RECOST/D3/RAISE/LEAD/Data Agent(2026)/GradFiltering/ENTP는
  cite-and-differentiate (각 1–2문장, §8) — 재현 비용 대비 필수 아님.

### 5.2 통계 (실험 리뷰어 major 반영 — v2에서 강화)

- **primary endpoint 사전 등록 (이 문서가 그 기록):** 0.5B composite-20,
  **TADS-MVF vs TADS-MVF-static**, 9-bench macro, seed-paired.
  (가장 강한 주장인 "adaptive가 static을 이긴다"를 primary로.)
- **primary pair는 처음부터 5 seeds** — 조건부 확장(optional stopping) 금지.
  나머지 arm 3 seeds.
- 사전 등록된 3개 pair(MVF−static, MVF−legacy, MVF−Random)에
  **Holm–Bonferroni**; 그 외 전부 exploratory 표기, 우월 서술 금지.
- t-CI와 함께 **정확 sign-flip permutation p** (make_table_v2 구현 완료).
- **co-headline 지표: oracle-gap recovery** `R = (macro_MVF −
  macro_Random)/(macro_Oracle − macro_Random)` seed-paired CI와 함께 —
  "dirty 회피가 downstream 가치로 이어졌는가"에 대한 직접 방어.
  dirty-fraction 곡선 단독 헤드라인 금지 (detection ≠ utility 공격 봉쇄).
- **diversity 지표 확정:** 선택 집합의 distinct MinHash-cluster coverage +
  anchor hidden-state k-means bucket coverage (vs Oracle) — figure 1 panel.

### 5.3 평가·백본 (사전 등록)

- 0.5B: 9벤치(기존 8 + IFEval) 전부. 7B: BBH 제외 8벤치(사유: 셀당 ~15h;
  0.5B BBH 행을 같은 표에 병기해 무해함을 보임). mmlu_pro 미포함(사전 결정).
- IFEval 구현 ~1d (`tads/evals` registry).
- **7B flagship 결정 필요 (D3, 사용자):** 권고 = **Qwen2.5-7B**
  (`configs/experiments/main_7b/qwen25/` 존재, 비용 동일, 2026년 심사에서
  LLaMA-2의 "3년 묵은 백본" 감점 회피). LLaMA-2-7B 유지 시 사유를 원고에
  사전 등록. 0.5B LoRA vs 7B full-FT 모드 차이는 pre-registered fact로
  1문장 + (여유 시) 0.5B full-FT 브리지 1 run.
- 7B는 5 epochs가 곡선 해상도상 옳지만 비용 재계산 후 3 epochs 유지 시
  사유를 지금 문서화: **7B = 3 epochs, 곡선 해상도는 0.5B(5 epochs)가 담당,
  7B는 endpoint 확인** — 이것이 사전 등록이다.

### 5.4 Run 예산 (가정: A100급 ≥ 4장 — D3 실측 후 조정)

| 블록 | runs |
|---|---:|
| 0.5B composite-20: 8 arms × 3 + primary pair 2 arms × 2 추가 seed | 28 |
| 0.5B source-skewed 3 × 3 | 9 |
| 0.5B clean-equiv 2 × 3 | 6 |
| 0.5B ablation: gate-form(rank/v2) 2×3, dedup-off 1×3, knockout e2e 2×3, partial-view p=50 1×3, adaptive-λ on/off 1×3 | 27 |
| 7B core {Random, legacy, MVF, static} × 3 | 12 |
| 7B Oracle × 3 (Gate B 통과 시) | 3 |
| **계** | **85** |

**삭감 순서(뒤부터):** 7B Oracle → adaptive-λ e2e(Phase A로 대체) →
partial-view e2e → knockout e2e → dedup-off → source-skewed → SelectIT.
**자르지 않는 선:** 0.5B composite 8 arms + clean-equiv + 7B core 12.

---

## 6. Phase C — 실데이터 검증 (승격: 제2 헤드라인 pool)

- **원본 Alpaca 52K를 제2 실험 pool로 승격** (실험 리뷰어 제안): {Random,
  TADS-legacy, TADS-MVF} × 3 seeds, 0.5B — 실제 노이즈 pool에서의 e2e가
  synthetic-only 공격의 최선 방어. (§5.4 예산 외 +9 runs — Gate B 시점
  여유로 판단, 부족하면 forward-only 스코어링 + 수동 라벨만.)
- 수동 라벨 300–500 (점수 분위 층화 추출, 라벨 정의·단일/이중 평가자 공개
  프로토콜을 지금 문서에 고정: mismatch/noisy/truncated/wrong/clean 5류,
  단일 평가자 시 그 사실 명시) → G와 각 뷰의 precision/recall +
  **Q reliability diagram** (Q-bin별 실측 clean율) — topic 7의 캐논 figure.
- D8 시작, 학습과 완전 병렬.

---

## 7. 이론 (이론 리뷰어 종합 — v2 Proposition 폐기·대체)

1. **Parametric non-compensation theorem (γ*):** dirty i (ΔL≤0, Q_i=0)와
   clean j (ΔL≥δ, c_j=1)에 대해, 모든 D,A 값에서 S_i < S_j ⟺
   `γ > γ* = log[(1+ε)(1+λ)/(d_floor+ε)] / log[(σ(δ/s)·2−1)·c+ε 관련 명시항]`
   — 정확한 형태는 구현(`mvf_score`의 c, ε, d_floor 포함)과 일치하게 유도.
   기본값 대입 (post-rezero): 보상 최대 = (1.01/0.51)·2 ≈ 3.96, 억제 =
   (0.61/0.01) = 61 → γ=1에서 여유 61/3.96 ≈ 15.4×. **γ* 조건-체크 행(δ̂,
   d0, γ*, shipped γ)을 결과 표에 포함**, v2 파라미터의 역전을 motivating
   counterexample로 수록.
   **c_trunc 상호작용 명시:** clean-but-incomplete (c = 0.2) 샘플은 Q <
   ~0.15에서 dirty-complete 최고점 아래로 내려갈 수 있다 — 의도된 soft
   penalty지만 정리 서술에 c_j = 1 가정을 명기하고, incomplete-clean의
   순위 하락을 한계로 1문장 기술.
2. **Dirty@K theorem (2-성분 ΔL mixture):** clean 질량 P(ΔL≥δ)≥1−β_c,
   dirty 질량 P(ΔL≤0)≥1−β_d + γ>γ* 하에서 union bound로
   Dirty@K ≤ g(π, β_c, β_d, K/N). Phase A 캐시에서 (π, δ̂, β̂_c, β̂_d) 추정
   → **predicted-vs-measured Dirty@K figure** (무료; "이론이 측정 곡선을
   예측한다"가 이 venue 최고의 이론 artifact).
3. **Calibration-robustness corollary:** s가 κ배 오추정 시 억제 여유의
   명시적 열화 + Phase A κ∈{0.5,2} 실측 짝 — CFP "robustness analysis"
   문구 직격.
4. **Additive-impossibility proposition:** 임의 고정 가중 additive 융합에는
   bounded-view 구성에서 ΔL≤0 샘플이 ΔL≥δ 샘플을 이기는 배치가 존재
   (class-level). QMF/PDF의 보상적 계열과의 형식적 대비.
5. DK lemma → appendix, "conditional stability of the Geometry view" 1문장.
6. **"Scope of theoretical claims" 문단:** 이론은 selection 단계까지;
   downstream은 §5 control로만. (여유 시 noisy-label stability 차용의
   conditional excess-risk corollary — D14까지 안 되면 문단만.)
7. D view의 regret 분석은 명시적으로 하지 않음 (static arm이 대신 방어).

---

## 8. 원고 계획 (스코프 리뷰어 required changes 반영)

- **제목 확정 (v3.1, Gate B 재론 없음):** *"Training-Adaptive Data Selection
  with a Reliability Gate: Multi-View Fusion for Low-Quality Instruction
  Data"* — 부제가 fusion + low-quality를 모두 담으므로 스코프 요건 충족,
  주제목이 방법의 실제 기여(비보상 게이트)를 말한다. v3의 후보 (a)/(b)는
  폐기. **"trajectory 전면 삭제" 방침도 철회** — trajectory-anchor alignment는
  뷰 이름으로 유지하되, §0 대응표와 method의 수식-명칭 일치 규칙으로 CIKM
  Reviewer 1 지적을 봉쇄한다.
- **약칭 사용 규칙:** TAG는 초록·intro·결과 표에서 method 이름으로만 쓰고,
  method 본문의 수식 서술은 기호(G, R, ã)로 한다. arm 표기는 코드와 1:1로
  — `TAG` = `score_mode: mvf`, `TAG-static` = `mvf.static: true`,
  `TADS-legacy` = `score_mode: tads`(게이트 없음). 표 캡션에 이 대응을 각주로.
- **Abstract:** §0′에 확정 문구 등록 완료. 구조는 training-adaptive 전제
  1문장 → 저품질 pool에서 difficulty–uncertainty가 오염을 **선호**한다는
  실패 서술 → 세 뷰(획득 스케줄 차이 포함) → 융합식 + veto-not-vote →
  `\STATUS{}` 결과. **§0′의 검증 부채 3건(STATUS 치환 / 33% 재측정 또는 삭제 /
  carrier 문구 B1·B2)을 제출 전 전부 닫을 것.**
- **Intro:** CFP 토픽 명시 문단(§0 순서) + taxonomy 매핑 테이블 + Huang et
  al. 2024를 둘째 문단에 인용·확장 선언. 첫 문단은 초록과 같은 순서
  (training-adaptive → 저품질에서의 역선택 → veto)로 전개하고, F2(entropy/
  PPL/IFD top-K의 오염률)를 그 문단에서 바로 참조한다.
- **Related work 3단:** ① MVL on low-quality data — TMC(ICLR'21),
  ETMC(TPAMI'23), QMF(ICML'23), PDF(ICML'24), RCML(AAAI'24), 저품질 서베이
  (2404.18947), Kittler'98/log-opinion pools/PoE/EDL(NeurIPS'18); 게스트
  에디터 인용은 역할 있는 것만 5–6편(Hu Neurocomputing'21 리뷰, Hu PR'20
  dynamic auto-weighted, Wen TSMC'23 서베이, Li Inf.Fusion'24, Wen+Wong
  AAAI'26 quality-aware(보상적 대조), 선택적으로 Hu IB 서베이 — Q를
  정보 관련성 검정으로 읽는 1문장을 넣을 때만). ② instruction 데이터 선택
  — LIMA, AlpaGasus, IFD/Cherry, Superfiltering, DEITA, SelectIT, LESS,
  MATES, RHO-loss(비보상적 AND의 정당화 인용), D3(IJCAI'25, 최근접 3-뷰
  경쟁자 — 보상적·정적과 대비), RAISE, LEAD, Data Agent(ICML'26), survey
  2402.05123. ③ noisy instruction data — RECOST(동기 무기: "IFD/PPL은
  dirty를 **더 높은** 확률로 선택"을 서두에), RobustFT, DS2, FiNE, ENTP·
  GradFiltering(concurrent 표기), Li&Sen 2026(A/D 뷰의 독립 근거),
  DataInf(detection 지표 선례). **전부 실존 검증 완료 — 이 목록 밖 인용은
  추가 검증 필수.**
- 참고문헌 53 → 커트 후 신규 ~15편, 최종 ≤ 50. D5까지 1차 커트.
- **Figure 계획:** F1 3-뷰 융합 schematic(= graphical abstract), F2
  motivation(entropy/PPL/IFD top-K의 오염), F3 dirty-fraction 곡선(MVF vs
  static vs legacy), F4 뷰별 기각 attribution(log-분해, topic 7), F5
  predicted-vs-measured Dirty@K, F6 Q reliability diagram(Phase C), F7
  partial-view 열화 곡선.
- 집필 병렬화: method/related/intro는 결과 무관 → D5–10 초고. GenAI
  선언 Elsevier 문구, kotex 제거, elsarticle 이관 D3–4 시험 컴파일.

---

## 9. 일정 (v3 — D1–2를 구현에 소진, 계획대로)

| 일자 | 작업 | 게이트 |
|---|---|---|
| D1–2 (8/11–12) | ✅ 적대적 검증 워크플로, §1 P0 + §2 v3 + §3 오염 + 표 v2 + arm config 구현·테스트 | |
| D3 (8/13) | GPU 가용량 실측, AlpaGasus API 확인, 7B 백본 확정, elsarticle 시험 컴파일, clean-ref ΔL 계산(`scripts/calibrate_reliability.py` → 보정 s), pool 재생성(T1b/T7 targets), T7 생성. **주의:** tokenize 코드 변경으로 HF map fingerprint가 무효화됨 — 서버 첫 실행에서 전 pool 재토크나이즈(~2분/pool)가 정상이며, 만약 재토크나이즈가 일어나지 않으면(stale cache) `text_complete` 컬럼이 없어 completeness가 token-only로 후퇴함(경고 로그 확인, 필요시 `TADS_FRESH_DATA_CACHE=1`) | |
| D4 (8/14) | Phase A 실행 (0.5B+7B forward-only, 전 신호) | |
| D5 (8/15) | Phase A 분석. **+ §0′ 검증 부채 결정 3건 마감**: carrier 문구 B1/B2(§2.0), 33% cost 재측정 vs 삭제, `\STATUS{}` 치환 계획 | **Gate A′** (§4). 실패 → D7 재시도 1회 |
| D5–11 | Phase B: 0.5B 그리드(§5.4), 7B core. 병렬: 이론(§7 1·2·4), 원고 초고, IFEval, 참고문헌 커트 | |
| D8– | Phase C 라벨링 병렬 시작 | |
| D11–12 (8/21–22) | 7B core + clean-equiv 취합 | **Gate B**: §5.4 삭감 적용 (제목은 §8에서 확정 완료 — 재론 없음), clean-equiv 실패 시 "저품질 강건 + 소폭 clean tax 명시" 서사 전환(사전 결정) |
| D12–16 | 잔여 run, Phase C 제2 pool(여유 시), 전체 figure/table(v2 파이프라인), 결과 집필 | |
| D16–18 | 전체 초고, §10 체크리스트, 공저자 검토 | |
| D18 (8/30) | 제출 | |

**D3 사용자 결정 3건:** ① GPU 가용량(→ §5.4 조정), ② AlpaGasus judge API,
③ 7B flagship (권고: Qwen2.5-7B). 공저자: PPO 경로 제거 확정.

---

## 10. Elsevier 제출 체크리스트 (v2에서 유지)

1. elsarticle 시험 컴파일 D3–4 (tikz/algorithm/threeparttable/tabularx,
   double-column, double-blind 옵션 vs CFP anonymized review 대조)
2. ACM-Reference-Format → elsarticle-num
3. 참고문헌 최종 ≤ 50
4. GenAI 선언 제목 교체 ("Declaration of generative AI and AI-assisted
   technologies in the manuscript preparation process")
5. Highlights 3–5 bullets ≤ 85자
6. **Graphical abstract = 3-뷰 융합 schematic** (531×1328 px; dirty-fraction
   곡선 아님 — 데스크 에디터가 2초 안에 스코프를 읽어야 함)
7. CRediT / 8. Funding / 9. Competing interests / 10. Data availability
   (corruption manifest + 코드 공개 결정 포함) / 11. Acknowledgements /
   12. Keywords (v3.1 확정, §0′와 동일): *Multi-view fusion; Low-quality
   data; Instruction tuning; Training-adaptive data selection; Reliability;
   Trajectory anchoring; Hidden-state representations; Foundation models*
   (8개 — Elsevier 권장 상한 확인 후 초과 시 *Hidden-state representations*
   부터 컷. partial-view arm 착지 시 *Incomplete multi-view learning* 추가)
13. 전체 소스 제출 / 14. 분량 재확인 / 15. 포털 SI 트랙("VSI: Low-quality
    Data: Foundation Models") + 마감 시간대 확인

---

## 11. 리스크 (v3.1 갱신)

- **초록의 검증 부채 3건 (v3.1 신규, 최우선)**: `\STATUS{}` 미치환 = 제출
  불가. **"33% vs Reward+PPO"는 CIKM 숫자이므로 audit §2.1 위반** — 새
  프로토콜 재측정 또는 문장 삭제, D5까지 결정(재측정 경로를 택하면 §5.4에
  cost-측정 run이 추가된다). "inherited unchanged"는 §2.0-B의 실질 불일치
  — 기본값 B1(초록 문구 수정). 세 건 모두 §0′에 체크 항목으로 등록됨.
- **명칭 리스크 (재발 경계)**: CIKM Reviewer 1의 reject 사유 1번이
  "`trajectory anchor` 명칭이 실제 수식과 불일치"였다(audit §1-1). v3.1이
  trajectory를 되살렸으므로 이 지적이 되살아날 수 있다 — §0 대응표 +
  method의 기호-명칭 1:1 서술 + 표 캡션 각주(§8 약칭 규칙)가 방어선이며,
  셋 중 하나라도 빠지면 초록에서 trajectory를 다시 뺀다.
- **일정**: D1–2 구현은 끝났지만 버퍼는 여전히 0. Gate A′ 재시도 소진 시
  0.5B 표 우선 제출 원칙(7B는 D16까지 연장, 부분 7B 숫자 게재 금지 —
  공저자와 D3 사전 합의).
- **calibration-vs-suppression 긴장**: re-zero 게이트는 clean pool 하위
  ~10%(열린 지시 포함)를 강하게 누른다 — clean-equivalence 실험이 이걸
  측정한다. 실패 시 γ<1 완화 + 서사 전환 fork는 §9 Gate B에 사전 결정됨.
- **스코프**: Huang et al. 2024 인용 + taxonomy 테이블 + 뷰 정의 서브섹션
  이 세 개가 방어선. 원고에서 하나라도 빠지면 데스크 리젝 리스크 복귀.
- **T7 생성 품질**: instruct 모델이 "그럴듯한 오답"을 못 만들면 T7이
  약해짐 — D3 생성 후 50개 수동 검수, 미달 시 프롬프트 조정 1회.
- **PPL/IFD가 composite에서 이겨버리는 경우**: T7 비중을 늘리고 유형별
  분해로 정직하게 보고 — "어느 유형에 어느 뷰가 필요한가"가 논문의 표.
- 기타 v2 리스크(합성 중심, AlpaGasus API, elsarticle, GPU 가용량) 유지.
