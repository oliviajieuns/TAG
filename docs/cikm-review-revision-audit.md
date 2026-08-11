# [ARCHIVE] CIKM 2026 리뷰에서 이월된 교훈

> **상태: 참고 자료. 실행 문서 아님.**
> CIKM 2026은 reject 확정이며 재제출하지 않는다(내년 사이클 포함).
> 이 문서는 리뷰 대응 계획이 아니라, **그 리뷰에서 배운 것 중 Information
> Fusion SI 원고에 그대로 적용되는 항목만** 남긴 기록이다.
> 실행 계획은 [`plan_low_quality_multiview.md`](plan_low_quality_multiview.md).
> 최종 정리: 2026-08-11.

## 1. 왜 reject됐는가 (한 문단)

Reviewer 1·3은 weak accept, Reviewer 2와 meta-review가 reject를 주도했다.
연구 주제 자체는 부정당하지 않았다. 문제는 세 가지였다.

1. `trajectory anchor`라는 명칭·서술이 실제 수식과 일치하지 않음
2. 1 pp 안팎 차이에 multi-seed 불확실성 분석이 없음
3. baseline·optimizer·token budget·cost 비교의 공정성과 재현성 미입증

즉 **아이디어가 아니라 증거 위생(evidence hygiene)에서 떨어졌다.** 새 논문에서
같은 실수를 반복하면 결과가 아무리 좋아도 같은 자리에서 막힌다.

---

## 2. 새 원고에 그대로 이월되는 항목 (반드시 지킬 것)

### 2.1 표 숫자 생성 규칙 — 최우선

[`scripts/make_table.sh`](../scripts/make_table.sh)는 벤치마크별로 발견된
**모든 결과의 최댓값을 평균**한다(L23 `W-AVG: unweighted mean of the per-bench
MAX accuracies`, L542 헤더 라벨 `W-AVG (max-of-each)`). eval 레이아웃이
`runs/<timestamp>/`로 이력을 보존하므로 재평가가 많은 셀일수록 값이 올라간다.

→ **새 원고의 모든 표는 "한 행 = 한 final checkpoint = 그 checkpoint에서 나온
전 벤치 점수"로만 만든다.** max-over-runs는 어떤 형태로도 쓰지 않는다.
이건 논쟁거리가 아니라 integrity 문제다.

### 2.2 벤치마크 목록은 사전 고정

CIKM 매트릭스는 9벤치를 돌리고([`AUTO_EVAL_AGENT.md`](../AUTO_EVAL_AGENT.md) §0-1)
표에는 8개만 실었다(`mmlu_pro` 누락, `make_table.sh` L137–146). 돌려놓고 뺀
벤치는 첫 번째 공격 지점이다. 새 원고는 macro에 들어갈 벤치 목록을 실험 전에
문서에 고정하고, 뺀 게 있으면 이유를 같이 적는다.

### 2.3 method별 튜닝 비대칭 금지

[`AUTO_EVAL_AGENT.md`](../AUTO_EVAL_AGENT.md) L336은
`data_agent_10 > tads_10 + 1%p` → "TADS 하이퍼파라미터 재검토 필요"를 운영
규칙으로 명시했다. baseline에는 대응 규칙이 없다. LLaMA-2-7B TADS만
fp32 AdamW / warmup 0.06 / clip 0.5를 쓴 것도 이 규칙의 산물로 보인다.

→ 새 실험은 **모든 arm이 동일 optimizer·scheduler·precision·effective batch·
world size·epoch 수**를 쓰고, 어떤 arm도 결과를 보고 재튜닝하지 않는다.
selected subset만 다르다는 주장을 하려면 실제로 그것만 달라야 한다.

### 2.4 Full FT는 upper bound가 아니라 reference

multi-seed로 재실행하지 않은 단일 run을 `upper-bound baseline`이라 부르지
않는다(런북 L321 표현). `full-data reference`로 표기한다.

### 2.5 통계 보고 최소선

- seed별 macro를 먼저 계산 → seed-level mean / SD / 95% CI
- 같은 seed 내 paired difference와 그 CI
- 결과 방향과 무관하게 전 seed 공개
- 차이가 1 pp 미만이거나 paired CI가 0을 포함하면 "우세"라고 쓰지 않는다

3 seeds가 최저선. 핵심 비교에서 갈리지 않으면 5로 늘린다.

### 2.6 낮춰야 할 표현

`consistently outperforms across model families / data settings / benchmarks`,
`only the selected subset differs`(실제로 그렇게 만든 뒤에만), `lambda optimum is
stable across tasks and backbones`, `recovers CO to within noise`,
Full FT = `upper bound`, theorem이 ranking 안정성을 보장한다는 서술,
측정 범위를 넘는 general scalability 주장.

`validation-free`는 "하이퍼파라미터 개발이 전혀 없었다"가 아니라 **"선택 시
target validation label이나 external quality label이 필요 없다"**로 제한한다.

### 2.7 명칭 — 해결됨

`Δh_l = h_l^(K) − h_l^(1)`은 checkpoint 간 이동이 아니라 **같은 시퀀스 내부의
contextualization delta**다. training adaptivity는 이 통계를 새 checkpoint에서
다시 계산하는 데서 나온다. "trajectory"는 오칭이었다.

→ 새 원고는 약어 TADS를 유지하되 확장을 `Training-Adaptive Data Selection`으로
바꾼다(최종 문안은 plan §8). `trajectory-derived`, `representation-space movement
induced by training`, `the trajectory itself is the signal` 전부 삭제.

### 2.8 이론의 위치

Davis–Kahan 결과가 보이는 것은 covariance drift와 eigengap 조건 하의 **PCA anchor
국소 안정성**뿐이다. 다음은 보이지 않는다: contextualization delta가 capability
direction이라는 것, alignment가 높으면 유용하다는 것, 안정적 anchor가 downstream
성능을 높인다는 것, min–max 정규화나 전체 ranking이 안정적이라는 것.

→ 새 원고에서 이 정리는 **alignment 뷰 하나의 conditional stability lemma**로
격하되고, 주 이론 기여는 fusion 쪽에서 새로 만든다
(plan §7). Utility는 control 실험으로만 주장한다.

### 2.9 비용 회계

`TADS 2.98h vs Data Agent 4.45h`는 raw timing artifact 없이 표 숫자만 남아 독립
검증이 불가능했다. 새 원고는 probe forward / null 추정 forward / PCA /
full-pool scoring / projection·top-k / SFT / true total wall-clock / aggregate
GPU-hours / peak GPU mem / peak host RAM을 분리 기록하고, seed별 median과 range를
보고한다. wall-clock과 A100-hours를 섞지 않는다.

### 2.10 baseline fidelity

Q2Q 구현은 원 논문의 precursor SFT를 생략한 simplified version이고
([`configs/methods/q2q.yaml`](../configs/methods/q2q.yaml)), NAIT의 seed 구성과
hidden-state layer indexing은 원 논문 대비 재확인이 필요하다. **SI 원고에서 이
둘은 쓰지 않으므로 이월 대상 아님.** 다시 쓸 일이 생기면 먼저 감사할 것.

---

## 3. 폐기된 항목 (더 이상 하지 않음)

- 리뷰 지적별 Agree/Disagree 판정표와 rebuttal 문안
- CIKM 재제출을 전제로 한 12-run core comparison (TADS/CO/Random/Data Agent × 3 seeds)
- 기존 main table 셀별 result manifest 소급 복구 — 새 실험으로 대체되므로 불필요.
  단, **max-of-each로 만들어진 기존 숫자는 어떤 새 문서에도 인용하지 않는다.**
- CO(λ=0) 계열 config/output path 분리 — 새 점수식에 λ가 없으므로 소멸
- delta vs mean candidate statistic 설계 ablation — alignment 뷰는 현행 유지로 확정
- dual-submission 게이트 — CIKM reject로 해소. Elsevier 제출에 제약 없음.
