# TADS CIKM 2026 리뷰 기반 수정 감사 및 실행 계획

> 상태: 내부 작업 문서  
> 작성일: 2026-08-11  
> 범위: 기존 TADS 논문의 정의, 구현, 실험 공정성, 통계적 증거 복구  
> 원칙: 아래 P0--P2가 해결되기 전에는 low-quality/multi-view 확장을 진행하지 않는다.

## 1. 결론

CIKM 리뷰는 TADS의 연구 주제 자체를 부정하지 않았다. Reviewer 1과 3은 weak accept였고, Reviewer 2의 weak reject와 meta-review는 다음 세 문제를 중심으로 reject를 권고했다.

1. 핵심 개념인 `trajectory anchor`의 정의와 실제 수식이 일치하지 않음
2. 1 pp 안팎의 성능 차이에 multi-seed 불확실성 분석이 없음
3. baseline, optimizer, token budget, cost 비교의 공정성과 재현성이 충분히 입증되지 않음

따라서 우선순위는 새로운 데이터나 문제 설정이 아니라 **현재 결과가 어떤 코드와 설정에서 생성됐는지 복구하고, 공정한 TADS--CO--Data Agent 비교를 다시 세우는 것**이다.

## 2. 리뷰 대응안의 판정 수정

초기 내부 revision plan의 `Not true`와 `Disagree`는 대부분 수정해야 한다.

| 리뷰 지적 | 기존 대응 | 수정된 판정 |
|---|---|---|
| PCA는 delta, candidate alignment는 mean activation | Agree; 구현은 둘 다 delta | Agree. 단, 현재 구현도 mean activation을 사용하므로 제안된 수정은 사실과 반대 |
| 제안 direction의 이론적 근거 부족 | Not true | Partly agree. 현재 theorem은 estimator stability만 보이고 utility는 보이지 않음 |
| representation extraction/PCA 비용 | Not true | Agree. 매 optimizer step이 아니라 epoch당 한 번임을 정정하되 overhead 자체는 인정 |
| 성능 향상이 작음 | Disagree | Agree on uncertainty. 단일 seed point estimate만으로 반박 불가 |
| `lambda=1`이 평가 benchmark에서 선택됐을 가능성 | Not true | Requires provenance. 전역값 사용은 확인되나 선택 chronology를 증명해야 함 |
| dynamic protocol이 불명확 | Not true | Agree. 구현에 정보가 있는 것과 원고가 충분히 설명한 것은 다름 |
| cost table이 불완전 | Disagree | Agree. controlled measurement와 dependency comparison을 분리해야 함 |
| 최근 baseline과 instruction-following 평가 부족 | Agree | Agree. 단, 기존 baseline fidelity audit 이후 추가 |
| baseline tuning detail이 충분하지 않음 | Not true | Agree. 현재 shared-setting 주장과 실제 config가 충돌 |
| scalability가 불명확 | Agree | Agree. 단계별 cost, memory, pool-size profile이 필요 |

## 3. P0: 새 실험 전에 해결할 결과 provenance

현재 저장소만으로 논문의 `40.70`, `38.64`, `39.83` 등이 정확히 어떤 코드, config, seed, checkpoint에서 생성됐는지 완전히 재구성할 수 없다.

각 결과 행에 다음 manifest를 연결해야 한다.

```text
checkpoint path
git SHA
resolved config snapshot
training and selection seed
candidate statistic: delta or mean
reward transform: raw or z-score
selected_indices_epoch*.json
evaluation outputs from one final checkpoint
timing_breakdown.json
world size and GPU type
```

### 3.1 Delta/mean 결과 생성 정의

현재 HEAD의 정의는 다음과 같다.

- Anchor PCA: first-to-last contextualization delta  
  [`tads/core/trajectory_anchor.py`](../tads/core/trajectory_anchor.py)
- Candidate alignment: sequence-mean hidden activation  
  [`tads/core/selector.py`](../tads/core/selector.py)

과거 candidate alignment는 delta였고 commit `1889eaf`에서 mean activation으로 변경됐다. 원고 표의 각 결과가 변경 전후 어느 구현에서 생성됐는지 먼저 확인해야 한다. 이를 확인하지 않고 원고 Eq.만 delta로 바꾸면 현재 코드 및 결과와 불일치한다.

### 3.2 Raw reward/z-score 불일치

`configs/experiments/main_7b/llama2/tads_10.yaml` 주석은 z-scored reward를 설명하지만 현재 ranking 구현은 raw non-negative reward를 사용한다.

- Config: [`configs/experiments/main_7b/llama2/tads_10.yaml`](../configs/experiments/main_7b/llama2/tads_10.yaml)
- Runtime score: [`tads/core/scorer.py`](../tads/core/scorer.py)

Commit `335963f` 전후 중 어느 구현이 결과를 만들었는지 확인하고, 최종 정의·코드·config·원고를 하나로 통일해야 한다.

### 3.3 CO와 표 생성 경로

현재 tracked config에는 독립적인 `co_10` 또는 `lam: 0` 실험이 없다. 또한 table script는 `data_agent_10`을 `Composite-reward only (lambda=0)` 행에 매핑한다.

- [`scripts/make_table.sh`](../scripts/make_table.sh)

같은 method/benchmark의 여러 결과가 있으면 task별 maximum을 선택해 평균하기 때문에, 한 행이 동일 checkpoint의 결과가 아닐 수도 있다. 다음 조치가 필요하다.

1. CO와 PPO Data Agent의 config/output path를 분리한다.
2. 한 row는 반드시 하나의 final checkpoint에서 나온 모든 benchmark 결과만 사용한다.
3. multiple seeds는 seed별 row를 먼저 만든 뒤 seed-level macro를 평균한다.
4. 회수할 수 없는 기존 결과는 재현 불가로 표시하고 재실행한다.

## 4. P0: 현재 main comparison의 공정성 문제

원고는 같은 backbone 안에서 모든 selector가 같은 hyperparameter를 사용하고 selected subset만 다르다고 주장한다. 현재 설정은 이 주장과 다르다.

### 4.1 TADS 전용 optimizer 설정

LLaMA-2-7B TADS만 다음 설정을 사용한다.

- fp32 AdamW
- warmup `0.06` (기본 `0.03`)
- gradient clip `0.5` (기본 `1.0`)

근거: [`configs/experiments/main_7b/llama2/tads_10.yaml`](../configs/experiments/main_7b/llama2/tads_10.yaml)

따라서 현재의 `TADS 40.70 vs CO 38.64`를 anchor만의 `+2.06 pp` 효과라고 해석할 수 없다. 동일 optimizer 설정으로 TADS와 CO를 다시 실행해야 한다.

### 4.2 Data Agent SFT protocol 차이

TADS는 하나의 optimizer와 scheduler를 세 epoch 동안 유지한다. Data Agent baseline은 매 epoch optimizer와 one-epoch scheduler를 다시 생성한다.

- TADS: [`tads/train.py`](../tads/train.py)
- Data Agent: [`baselines/data_agent/train.py`](../baselines/data_agent/train.py)

Data Agent와 TADS의 차이를 selector 효과로 해석하려면 optimizer state, scheduler, effective batch, world size를 통일한 controlled SFT loop가 필요하다.

### 4.3 Baseline fidelity

- Q2Q 구현은 원 논문의 precursor SFT를 생략한 simplified version이다: [`configs/methods/q2q.yaml`](../configs/methods/q2q.yaml)
- NAIT의 seed 구성과 hidden-state layer indexing이 원 논문과 맞는지 다시 확인해야 한다: [`configs/methods/nait.yaml`](../configs/methods/nait.yaml), [`baselines/nait/direction.py`](../baselines/nait/direction.py)

새 baseline을 추가하기 전에 현재 표의 baseline부터 faithful implementation인지 검증한다.

## 5. 방법 정의 수정

### 5.1 `Trajectory` 명칭

현재

\[
\Delta h_l = h_l^{(K)} - h_l^{(1)}
\]

은 checkpoint-to-checkpoint movement가 아니라 동일 sequence 내부의 contextualization delta다. Training adaptivity는 이 통계를 새 checkpoint에서 다시 계산하는 데서 나온다.

권장 제목:

> **TADS: Training-Adaptive Data Selection via Online Hidden-State Geometry for Instruction Tuning**

원고 전체에서 다음 표현을 통일한다.

- `trajectory-derived` -> `checkpoint-adaptive` 또는 `recomputed during training`
- `trajectory anchor` -> `online hidden-state geometry anchor`
- `representation-space movement induced by training` 삭제
- `the trajectory itself is the signal` 삭제

### 5.2 Delta와 mean 중 최종 설계 선택

수식만 먼저 수정하지 않는다. 다음 세 변형을 비교한다.

1. Delta-PCA -> mean candidate: 현재 방식
2. Delta-PCA -> delta candidate
3. Mean-PCA -> mean candidate

먼저 scoring-only 분석으로 rank correlation과 top-10% overlap을 측정한다. subset이 실제로 달라질 때 Qwen2.5-0.5B, 3 seeds downstream SFT를 실행한다.

- 현재 방식이 명확히 우수하면 PCA statistic과 candidate statistic의 서로 다른 역할을 설명하고 ablation을 보고한다.
- 성능이 비슷하면 개념적 일관성이 높은 delta-to-delta를 우선한다.
- delta-to-delta가 우수하면 방법을 변경하고 main result를 다시 생성한다.

## 6. 이론의 역할 수정

현재 Davis--Kahan 결과가 보이는 것은 covariance drift와 eigengap 조건 아래 PCA anchor의 국소적인 estimator stability다. 다음은 보이지 않는다.

- contextualization delta가 capability direction이라는 것
- alignment가 높은 샘플이 더 유용하다는 것
- 안정적인 anchor가 downstream 성능을 높인다는 것
- min--max normalization 또는 전체 ranking이 안정적이라는 것

따라서 theorem을 utility justification이 아니라 conditional stability lemma로 낮춘다. Utility는 다음 control로 검증한다.

- mean vs delta
- refreshed vs fixed-initial anchor
- shuffled/random anchor
- sign calibration 제거

Raw non-negative carrier를 최종 정의로 확정하면 다음 bounded-refinement proposition을 추가할 수 있다.

\[
R_i \le s_i \le (1+\lambda)R_i.
\]

따라서 `R_i > (1+lambda) R_j`이면 anchor가 두 샘플의 순위를 뒤집을 수 없다. 이는 anchor가 carrier를 대체하지 않고 carrier 값이 가까운 후보를 제한적으로 재정렬한다는 주장을 직접 뒷받침한다.

반면 `learning-rate-scale perturbation of the base ranking`처럼 현재 theorem이 ranking까지 보장하는 표현은 삭제한다.

## 7. 필수 재실험

### 7.1 Core comparison: 최소 12 runs

현재 main setting에서 다음 네 방법을 동일 protocol로 3 seeds 실행한다.

| Method | Seeds | Runs |
|---|---:|---:|
| Random | 3 | 3 |
| CO | 3 | 3 |
| TADS | 3 | 3 |
| Data Agent | 3 | 3 |
| **Total** |  | **12** |

통일할 항목:

- optimizer 종류와 precision
- epoch 간 optimizer state 유지
- 하나의 global scheduler
- learning rate, warmup, clip, weight decay
- world size와 effective batch
- 3 epochs, refresh `T=3`
- 매 epoch top-10% replacement, no accumulation
- total optimizer updates
- actual input/response-token exposure
- 동일 final checkpoint evaluation

3 seeds는 variance 보고의 최저선이다. TADS와 baseline의 차이가 1 pp 미만이거나 paired CI가 0을 포함하면 핵심 네 방법만 5 seeds로 확장한다.

보고 순서:

1. seed별 8-task macro 계산
2. seed-level mean, SD, 95% CI
3. 같은 seed의 TADS--CO, TADS--Data Agent, TADS--Random paired difference
4. 결과 방향과 무관하게 모든 seed 공개

Full FT를 multi-seed로 재실행하지 않으면 `upper bound`가 아니라 single-run `full-data reference`로만 사용하고, 통계적 우월성 주장은 하지 않는다.

### 7.2 최소 추가 평가

- IFEval 등 instruction-following 평가 1개
- TADS가 CO 대비 승격·강등한 예시
- epoch 간 selected-subset Jaccard와 unique exposure
- response length, domain, loss, entropy, alignment 분포
- fixed/random/shuffled anchor control

## 8. Cost 및 scalability 재측정

현재 `TADS 2.98h vs Data Agent 4.45h`는 raw timing artifact 없이 표의 숫자만 남아 있어 독립 검증할 수 없다. Data Agent timer에는 outer episode와 내부 forward/PPO phase가 같은 category로 중첩 계측될 가능성도 있다.

재측정 시 다음을 분리한다.

- probe forward
- PCA
- full-pool scoring forward
- projection/top-k
- SFT
- setup/data/checkpoint를 포함한 true total wall-clock
- aggregate GPU-hours
- peak GPU memory
- peak host RAM

같은 hardware, world size, cache policy에서 seed별 값의 median과 range를 보고한다. Wall-clock과 A100-hours를 혼용하지 않는다.

추가 full training 없이 pool size 10K/25K/52K의 selection-only throughput을 측정하면 scalability 지적에 대한 최소 증거가 된다.

## 9. 원고에서 즉시 낮춰야 할 주장

새 결과가 나오기 전에는 다음 표현을 삭제하거나 제한한다.

- `consistently outperforms across model families, data settings, and benchmarks`
- `only the selected subset differs`
- `lambda optimum is stable across tasks and backbones`
- `recovers CO to within noise` (반복 실험 없음)
- Full FT를 `upper bound`로 부르는 표현
- theorem이 base ranking stability를 보장한다는 표현
- 측정 범위를 넘는 general scalability claim

대신 다음 수준으로 쓴다.

> TADS obtains the highest point-estimate average in the main setting and exceeds Data Agent's point estimate in the reported model--dataset settings; statistical reliability is evaluated with matched multi-seed runs.

`validation-free`는 hyperparameter 개발이 전혀 없다는 뜻이 아니라, **selection 시 target validation label이나 external quality label이 필요하지 않다**는 뜻으로 제한한다.

## 10. 실행 순서와 gate

### P0 -- 학습 전 감사

1. 실제 CIKM 제출 PDF와 실제 rebuttal 확보
2. 모든 표 셀의 result manifest 작성
3. mean/delta 및 raw/z-score 결과 생성 정의 확정
4. CO와 PPO Data Agent의 config/output path 분리
5. 동일 checkpoint 단위 table generation 확정
6. baseline fidelity와 raw timing log 확인

### P1 -- 기존 TADS를 성립시키는 필수 증거

1. 공정한 12-run core comparison
2. delta/mean 설계 ablation
3. actual token exposure와 selection-cost profiling
4. instruction-following 평가
5. qualitative selection analysis

### P2 -- 원고 수정

1. trajectory를 training-adaptive hidden-state geometry로 변경
2. theorem을 conditional stability 범위로 축소
3. refresh/replacement/steps/tokens protocol 표 추가
4. controlled cost와 dependency 표 분리
5. 결과에 맞게 abstract/conclusion claim 재작성
6. 명시적 Limitations 추가

### P3 -- 이후 확장

P0--P2를 통과한 뒤에만 최근 baseline 1--2개와 작은 scaling study를 추가한다. Low-quality/multi-view 특별호 확장은 이 기존 논문의 정합성과 증거가 복구된 다음 별도로 판단한다.

