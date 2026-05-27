# 프론트엔드 학습 환경

## 하네스: 프론트엔드 학습

**목표:** JS 초중급 수준의 학습자가 React/Next.js/TanStack Query/Tailwind를 원리부터 배포까지 체계적으로 학습하도록 지원한다.

**트리거:** 프론트엔드 학습, 개념 설명, 예제 생성, 프로젝트 설정, 테스트, 배포 관련 요청 시 `fe-learning` 스킬을 사용하라. 단순 질문은 직접 응답 가능.

**학습자 수준:** JS 코드를 일부 이해하는 초입자. 전문 용어 첫 등장 시 한국어 설명 병기. 비유와 최소 코드 예제 활용.

## 예제 생성 필수 규칙

예제를 생성할 때 아래 항목을 **빠짐없이** 이행한다.

| # | 규칙 | 담당 에이전트 |
|---|------|-------------|
| 1 | `concepts.md` 반드시 생성 — 개념 설명은 README가 아닌 concepts.md에 작성 | workshop-guide, concept-explainer |
| 2 | `README.md`에 원본 파일명과 `cp` 명령어 명시 | workshop-guide |
| 3 | 예제 생성 후 `_workspace/my-first-app`에 자동 복사 | workshop-guide |
| 4 | 파일은 `_workspace/my-first-app/app/<full-slug>/page.tsx` 경로에 배치 (루트 app/ 오배치 금지, 슬러그 축약 금지) | workshop-guide |
| 5 | concepts.md에서 학습자가 몰랐던 개념에 ⚠️ 표기 | concept-explainer |
| 6 | `README.md`에 Q&A 기록 섹션 포함 — 확장 과제 중 나온 질문·해결책·장단점 기록 | workshop-guide |

> 변경 이력 → [docs/changelog.md](./docs/changelog.md)
