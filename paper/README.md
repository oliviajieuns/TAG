# CIKM 2026 — Self-Anchored Alignment

## 디렉토리 구조

```
cikm2026/
├── main.tex                          # 메인 (compile here)
├── references.bib                    # bib (검증 필요한 항목 [VERIFY] 표시)
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related.tex                # 동료 담당, outline 상태
│   ├── 03_method.tex
│   ├── 04_experiments.tex            # 결과는 \TBD / \TODO 처리
│   ├── 05_discussion.tex
│   ├── 06_conclusion.tex
│   └── 07_genai_disclosure.tex
└── README.md (이 파일)
```

## Overleaf 업로드 방법

1. Overleaf에서 **New Project → Upload Project (zip)** 선택
2. 이 디렉토리 전체를 zip으로 압축해서 업로드
3. 또는 **New Project → Blank** 만든 뒤 파일 하나씩 업로드
4. **Compiler를 pdfLaTeX**로 설정 (Overleaf 기본값 OK)
5. **Main document를 main.tex**로 설정

## 컴파일 옵션

현재 main.tex는 제출용 옵션:

```latex
\documentclass[sigconf, anonymous, review]{acmart}
```

- `anonymous`: 저자 정보 자동 익명화 (double-blind 요건 충족)
- `review`: line numbers 표시 (reviewer 참조용)

작성 중 author 정보 보고 싶으면 일시적으로:
```latex
\documentclass[sigconf, authordraft]{acmart}
```
로 바꿨다가 제출 직전 원복.

## 페이지 한도 (CIKM 2026)

- 본문 + figures + tables + appendices: **최대 10 pages**
- References 전용: 추가 **최대 2 pages**
- GenAI Disclosure 섹션 (`07_genai_disclosure.tex`): page limit 외

## TODO 체크리스트 (5/23 마감)

### 즉시 (오늘~내일 아침)
- [ ] §1 thesis 다듬기 (이미 거의 완성)
- [ ] §3 Theorem 1 proof 본문에서 검증 (Davis-Kahan 인용 정확성)
- [ ] §4.2 (NAIT seed bias) — 수치 채워짐. NAIT-MMLU/TydiQA 결과 들어오면 6×6 / 6-row 확장
- [ ] references.bib `[VERIFY]` 항목 검증 (NAIT, Data Agent, CPQS, SelectIT)

### 5/8 아침 (학습/평가 완료 후)
- [ ] §4.3 main result table 채우기 (`summarize_results.py` 출력)
- [ ] 게이트 2 판정 (Ours vs NAIT-Mix avg)
- [ ] §4 narrative paragraph 작성

### 5/9~5/16 (Abstract 마감)
- [ ] §3 Theorem 1 full proof appendix 작성
- [ ] §4.4 cost comparison wall-clock 측정
- [ ] §4.5 ablations (λ, layer, dynamic vs static)
- [ ] §4.6 anchor stability figure (Theorem 1 empirical)
- [ ] Abstract 200~250 words 압축
- [ ] Abstract EasyChair 제출 (5/16)

### 5/16~5/23 (Full paper)
- [ ] §2 Related Work 동료 작성 분 통합
- [ ] §5 Limitations 최종 다듬기
- [ ] References double-check
- [ ] GenAI disclosure 모델/사용처 명시
- [ ] PDF 컴파일 + 익명화 확인 (저자 정보 누락 / self-citation 3인칭)
- [ ] CCS concept 정확히 선택 (현재 placeholder)
- [ ] EasyChair Full Paper 제출 (5/23)

## 자주 쓸 LaTeX 팁

### TODO/TBD 시각화
`main.tex`에 정의된 매크로:
- `\TODO{내용}` — 빨간색 `[TODO: 내용]`
- `\TBD` — 빨간색 `TBD`

제출 전 일괄 검색:
```bash
grep -rn "TODO\|TBD\|VERIFY" .
```

### 익명화 검증
double-blind 위반 자주 발생하는 곳:
1. acks 환경 (지금은 비어있음, OK)
2. 자기 인용 — `\cite{self2024}`을 1인칭 ("our prior work")으로 쓰면 안 됨, 3인칭으로
3. PDF metadata — `\title`, `\author`만 익명이면 OK
4. 그림 캡션이나 코드에 GitHub URL — `https://github.com/anonymous` 형태로

## 컴파일 실패 시

ACM 클래스에 흔한 에러:

| 에러 | 원인 | 해결 |
|------|------|------|
| `acmart not found` | 패키지 없음 | Overleaf는 자동 설치, 로컬은 `tlmgr install acmart` |
| `Bibliography not found` | bib 파일 누락 | references.bib 같은 폴더에 있는지 확인 |
| `Undefined control sequence \TODO` | 매크로 정의 누락 | main.tex preamble의 `\newcommand{\TODO}{...}` 살아있는지 |
| `Theorem unknown` | amsthm 누락 | `\usepackage{amsmath, amssymb, amsthm}` 확인 |

## 마지막 점검 (제출 직전)

```latex
% 제출 직전 main.tex 첫 줄 확인
\documentclass[sigconf, anonymous, review]{acmart}  % ← anonymous, review 필수
```

```bash
# TODO 잔여 확인
grep -rn "TODO\|TBD\|VERIFY" sections/ main.tex references.bib | wc -l
# 0이어야 함
```

```bash
# 최종 PDF 페이지 수 확인
pdfinfo main.pdf | grep Pages
# 본문 ≤ 10, references 추가 ≤ 2 = 총 ≤ 12
```

