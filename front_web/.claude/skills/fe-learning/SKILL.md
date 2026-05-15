---
name: fe-learning
description: "프론트엔드 학습 오케스트레이터. 학습 계획 작성, 개념 설명, 예제 설계, React/Next.js/TanStack Query/Tailwind 아키텍처 구성, 테스트 코드 작성, 배포까지 전체 프론트엔드 학습 여정을 조율한다. 다음 상황에서 반드시 이 스킬을 사용할 것: (1) '프론트엔드 배워줘', '리액트 시작하고 싶어' 등 학습 시작 요청, (2) 'Next.js 설명해줘', 'useState가 뭐야' 등 개념 질문, (3) '투두앱 만들어줘', '예제 코드 보여줘' 등 실습 요청, (4) '테스트 코드 짜줘', '린터 설정해줘' 등 품질 도구 요청, (5) 'Vercel에 배포해줘', 'CI/CD 구성해줘' 등 배포 요청, (6) '다음에 뭐 배워야 해', '학습 계획 세워줘' 등 커리큘럼 요청, (7) '다시 실행', '이어서 진행', '이전 예제 수정' 등 후속 작업 요청."
---

# 프론트엔드 학습 오케스트레이터

## 역할

5명의 전문 에이전트 팀을 조율하여 JS 초중급 수준의 학습자가 프론트엔드 개발 전체 과정을 배울 수 있도록 안내한다.

**팀 구성:**
- `curriculum-planner`: 학습 로드맵 설계, 단계별 계획 수립
- `concept-explainer`: 개념 원리 설명 (WHY 중심, 비유 활용)
- `workshop-guide`: 아키텍처 구성, 실습 예제, 개발 도구 설정
- `quality-guide`: 테스트 코드 작성, ESLint/Prettier 설정
- `deploy-guide`: Vercel 배포, GitHub Actions CI/CD

## Phase 0: 컨텍스트 확인

워크플로우 시작 시 기존 상태를 먼저 확인한다.

```
_workspace/curriculum/roadmap.md 존재?
├── Yes + 사용자가 부분 요청 → 부분 재실행 (해당 에이전트만 호출)
├── Yes + 새 주제 요청      → 기존 로드맵 참조하여 이어서 진행
└── No                    → 초기 실행 (curriculum-planner부터)
```

## Phase 1: 요청 분석 및 실행 모드 결정

사용자 요청을 분석하여 실행 모드를 결정한다.

### 빠른 요청 (서브 에이전트 1명)
단일 개념 질문이나 단순 설정은 해당 에이전트만 단독 호출한다.

| 요청 유형 | 호출 에이전트 |
|----------|------------|
| 개념 설명 ("useState가 뭐야?") | concept-explainer |
| 빠른 코드 예제 ("Counter 컴포넌트 예제") | workshop-guide |
| 린터/포맷터 설정 | quality-guide |
| 배포 설정 파일 생성 | deploy-guide |

### 전체 학습 세션 (에이전트 팀)
새 학습 단계 시작, 실전 프로젝트 구축, 전체 절차가 필요할 때 팀을 구성한다.

| 요청 유형 | 팀 구성 |
|----------|--------|
| 학습 시작/재시작 | curriculum-planner + concept-explainer + workshop-guide |
| 실전 프로젝트 빌드 | workshop-guide + concept-explainer + quality-guide |
| 완전한 프로젝트 셋업 | 5명 전체 |

## Phase 2: 에이전트 팀 실행

### 전체 학습 세션 워크플로우

```
1. curriculum-planner
   → 사용자 수준 파악 (없으면 3~5개 질문)
   → 로드맵 생성/업데이트
   → 현재 단계 목표 확정
   → 출력: _workspace/curriculum/roadmap.md

2. concept-explainer (curriculum-planner와 병렬 가능)
   → 이번 단계 핵심 개념 WHY 설명
   → 비유 + 최소 코드 예제

3. workshop-guide
   → concept-explainer 설명 완료 후 실행
   → 실습 예제 코드 작성 (실행 가능)
   → 아키텍처 설정 (필요 시)
   → 출력: 실제 프로젝트 파일

4. quality-guide (workshop-guide 완료 후)
   → 핵심 컴포넌트 테스트 코드 작성
   → ESLint 설정 (미설정 시)

5. deploy-guide (quality-guide 완료 후, 배포 요청 시)
   → 배포 설정 파일 생성
   → 배포 진행
```

### 팀 생성 예시

```
팀 모드: 에이전트 팀 (전체 학습 세션)
팀원: curriculum-planner, concept-explainer, workshop-guide

역할 분담:
- curriculum-planner: 로드맵 생성 → workshop-guide에 단계 전달
- concept-explainer: 개념 설명 → workshop-guide와 병렬 실행 가능
- workshop-guide: 실습 예제 구현 → quality-guide에 파일 전달
```

## Phase 3: 결과 통합 및 전달

각 에이전트 산출물을 사용자에게 통합하여 전달한다:

1. **학습 요약**: 이번 세션에서 배운 것 (1~3줄)
2. **생성된 파일 목록**: 실습 코드, 설정 파일 경로
3. **다음 단계**: 로드맵에서 다음 학습 항목
4. **연습 과제**: 스스로 해볼 수 있는 확장 과제 1~2개

## 데이터 흐름

```
_workspace/
├── curriculum/
│   └── roadmap.md          ← curriculum-planner 생성
├── examples/
│   └── {topic}/            ← workshop-guide 생성
│       ├── README.md       (단계별 설명)
│       └── 실습 파일들
└── session-log.md          ← 각 세션 요약 기록
```

## 에러 핸들링

- 에이전트 응답 실패 시: 1회 재시도, 재실패 시 해당 단계 건너뛰고 보고
- 코드 실행 오류 시: workshop-guide에 오류 원인 분석 요청
- 학습자가 막혔을 때: concept-explainer에 다른 방식으로 재설명 요청

## 테스트 시나리오

**정상 흐름:**
1. "프론트엔드 처음 배우려고 해. React부터 시작하고 싶어."
   - curriculum-planner가 수준 파악 질문 → 로드맵 생성
   - concept-explainer가 "컴포넌트란 무엇인가" WHY 설명
   - workshop-guide가 `Counter` 예제 코드 생성

**에러 흐름:**
1. workshop-guide가 생성한 코드에 빌드 오류 발생
   - 오류 로그 분석 → 수정 방법 제시 → 재생성

**후속 작업:**
1. "저번에 만든 Counter 예제에 테스트 코드 추가해줘."
   - `_workspace/examples/counter/` 확인 → quality-guide 단독 호출

## 참고 자료

- 학습 로드맵 상세: `references/learning-phases.md`
- 기술 스택 결정 이유: `references/tech-stack-guide.md`
