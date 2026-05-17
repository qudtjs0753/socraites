---
name: rag-study-orchestrator
description: "RAG(Retrieval Augmented Generation) 학습 스터디 오케스트레이터. RAG 공부, 커리큘럼, RAG 실습, 구현, 배포, 벡터DB, Chroma, Elasticsearch, 임베딩, 청킹, 하이브리드 검색, K8S RAG 배포, Kind 클러스터, 랭체인, 라마인덱스, 학습 계획 관련 요청 시 반드시 이 스킬을 사용하라. 후속 작업: RAG 커리큘럼 업데이트, 다음 레벨로 진행, 특정 주제 다시 설명, 실습 코드 요청, 이전 학습 이어서 등 모든 RAG 학습 관련 요청에도 이 스킬을 사용."
---

# RAG 학습 스터디 오케스트레이터

Python 개발자 / AIOps(K8S+Kind) / DA(Chroma, Elasticsearch) 배경을 가진 학습자를 위한 RAG 학습 팀을 조율하는 통합 스킬.

## 실행 모드: 하이브리드

- **커리큘럼 초기 생성 (Phase 2-3)**: 에이전트 팀 — curriculum-designer + 전문 튜터들이 협업하여 종합 커리큘럼 작성
- **학습 세션 (Phase 4)**: 서브 에이전트 — 해당 주제 전문 튜터만 호출

## 에이전트 구성

| 팀원 | 에이전트 파일 | 역할 | 담당 레벨 |
|------|-------------|------|----------|
| curriculum-designer | agents/curriculum-designer.md | 학습 로드맵 및 커리큘럼 설계 | 전체 |
| rag-tutor | agents/rag-tutor.md | Python RAG 구현 가이드 | Level 1-3 |
| infra-tutor | agents/infra-tutor.md | K8S/Kind RAG 배포 가이드 | Level 4 |
| vectordb-tutor | agents/vectordb-tutor.md | 벡터DB/검색 엔진 가이드 | Level 2-4 |

## 워크플로우

### Phase 0: 컨텍스트 확인

1. `_workspace/` 디렉토리 존재 여부 확인
2. 실행 모드 결정:
   - **`_workspace/` 미존재** → 초기 실행. Phase 1로 진행
   - **`_workspace/` 존재 + 특정 주제/레벨 요청** → 부분 재실행. 해당 튜터만 호출
   - **`_workspace/` 존재 + 새 학습 목표** → 새 실행. 기존 백업 후 Phase 1 진행
3. 기존 `_workspace/00_learning_roadmap.md` 있으면 현재 학습 단계 파악

### Phase 1: 사용자 분석

1. 요청 내용 파악:
   - 커리큘럼 요청인가? → Phase 2-3 전체 실행
   - 특정 레벨/주제 학습 요청인가? → Phase 4 직접 실행
   - 이전 학습 이어서인가? → `_workspace/` 읽고 Phase 4 실행
2. `_workspace/` 디렉토리 생성 (없으면)
3. `_workspace/00_input.md`에 사용자 목표 기록

### Phase 2: 팀 구성 (커리큘럼 생성 시)

커리큘럼 초기 생성일 때만 실행. 에이전트 팀 모드로 진행.

팀 생성:
```
TeamCreate(
  team_name: "rag-study-team",
  members: [
    { name: "curriculum-designer", model: "opus",
      prompt: "RAG 학습 커리큘럼 설계자. curriculum-overview.md 참조: .claude/skills/rag-study-orchestrator/references/curriculum-overview.md" },
    { name: "rag-tutor", model: "opus",
      prompt: "Python RAG 구현 전문가. 각 레벨별 Python 실습 예제 섹션 작성" },
    { name: "vectordb-tutor", model: "opus",
      prompt: "벡터DB/검색 엔진 전문가. Chroma, Elasticsearch 실습 섹션 작성" },
    { name: "infra-tutor", model: "opus",
      prompt: "K8S/Kind 인프라 전문가. Level 4 배포 섹션 및 전체 인프라 가이드 작성" }
  ]
)
```

작업 등록:
```
TaskCreate(tasks: [
  { title: "전체 로드맵 설계", assignee: "curriculum-designer",
    description: "4개 레벨 커리큘럼 개요 작성. references/curriculum-overview.md 참조" },
  { title: "Level 1-2 Python 실습", assignee: "rag-tutor",
    description: "기초/향상 RAG Python 코드 예제 작성. references/level1-basic-rag.md, level2-enhanced-rag.md 참조" },
  { title: "Level 2-3 벡터DB 가이드", assignee: "vectordb-tutor",
    description: "Chroma + Elasticsearch 실습 가이드. references/level2-enhanced-rag.md 참조" },
  { title: "Level 3 Python 실습", assignee: "rag-tutor",
    description: "고급 RAG(Self-RAG, CRAG, HyDE) 코드 작성. references/level3-advanced-rag.md 참조",
    depends_on: ["Level 1-2 Python 실습"] },
  { title: "Level 4 인프라 가이드", assignee: "infra-tutor",
    description: "Kind + K8S RAG 배포 가이드. references/level4-production-k8s.md 참조" },
  { title: "커리큘럼 통합 및 로드맵 완성", assignee: "curriculum-designer",
    description: "각 팀원 산출물 통합하여 최종 로드맵 작성",
    depends_on: ["전체 로드맵 설계", "Level 1-2 Python 실습", "Level 2-3 벡터DB 가이드", "Level 4 인프라 가이드"] }
])
```

### Phase 3: 팀 협업 및 커리큘럼 생성

**실행 방식:** 팀원 자체 조율

각 팀원이 담당 섹션을 작성하고 파일로 저장한다:

| 팀원 | 출력 경로 |
|------|----------|
| curriculum-designer | `_workspace/01_roadmap_overview.md` |
| rag-tutor | `_workspace/02_python_guide_level1.md`, `_workspace/02_python_guide_level2.md`, `_workspace/02_python_guide_level3.md` |
| vectordb-tutor | `_workspace/04_vectordb_guide_chroma.md`, `_workspace/04_vectordb_guide_elasticsearch.md` |
| infra-tutor | `_workspace/03_infra_guide_kind.md`, `_workspace/03_infra_guide_k8s.md` |

통합 산출물: `_workspace/00_learning_roadmap.md` (curriculum-designer가 최종 통합)

팀 통신 규칙:
- rag-tutor는 Chroma 코드 작성 시 vectordb-tutor에게 API 확인 요청 가능
- infra-tutor는 Level 4 K8S 환경에서 Python 앱 설정 관련 rag-tutor에게 확인 가능
- 모든 팀원은 작업 완료 시 리더에게 SendMessage로 알림

### Phase 4: 학습 세션 (서브 에이전트 모드)

특정 주제 학습 요청 시 해당 튜터를 서브 에이전트로 호출:

```
주제별 에이전트 매핑:
- Python 코드 / RAG 구현 / LangChain / RAGAS → rag-tutor
- Chroma / Elasticsearch / 벡터DB / BM25 / 하이브리드 검색 → vectordb-tutor
- K8S / Kind / Helm / 배포 / 모니터링 → infra-tutor
- 커리큘럼 / 학습 계획 / 레벨 평가 → curriculum-designer
```

Agent 호출:
```
Agent(
  subagent_type: "general-purpose",
  model: "opus",
  prompt: "
    에이전트 정의: .claude/agents/{agent-name}.md
    참조 문서: .claude/skills/rag-study-orchestrator/references/{level-file}.md
    
    사용자 요청: {user_request}
    현재 학습 단계: {current_level}
    이전 산출물: _workspace/ (있으면 참조)
    
    산출물을 _workspace/{prefix}_{topic}.md에 저장하라.
  "
)
```

### Phase 5: 결과 통합 및 보고

1. 생성된 산출물 파일 목록 확인
2. 사용자에게 결과 요약:
   - 생성된 커리큘럼/가이드 목록
   - 다음 학습 단계 추천
   - 실습 시작 방법
3. 팀 사용 시: TeamDelete로 팀 정리
4. `_workspace/` 보존 (다음 세션 이어서 사용)
5. 사용자에게 피드백 요청: "어떤 부분을 더 깊이 다루고 싶으신가요?"

## 데이터 흐름

```
[오케스트레이터]
    │
    ├─ 커리큘럼 생성 → TeamCreate → [팀원 자체 조율]
    │                                    │
    │   curriculum-designer ←────────────┤
    │   rag-tutor           ←────────────┤
    │   vectordb-tutor      ←────────────┤
    │   infra-tutor         ←────────────┘
    │           │
    │           ↓ _workspace/ 파일들
    │           ↓
    │   [curriculum-designer 통합]
    │           ↓
    │   _workspace/00_learning_roadmap.md
    │
    └─ 학습 세션 → Agent(해당 튜터) → _workspace/{topic}.md
```

## 환경 제약 — 에이전트 필수 준수 사항

**폐쇄망**: `pip install`은 가능. 외부 API(OpenAI 등) 사용 불가 가능. HuggingFace 모델·Docker 이미지는 외부 다운로드 필요.

모든 에이전트는 코드 예제 작성 시 다음을 준수한다:

1. **LLM/임베딩 초기화** — 반드시 아래 표준 패턴 사용. 특정 API에 하드코딩 금지:
   ```python
   import os
   from langchain_openai import ChatOpenAI, OpenAIEmbeddings
   # .env에서 사내 LLM 설정 → 엔드포인트만 바꾸면 어디서나 동작
   llm = ChatOpenAI(
       base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
       api_key=os.getenv("LLM_API_KEY"),
       model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
       temperature=0,
   )
   embeddings = OpenAIEmbeddings(
       base_url=os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1"),
       api_key=os.getenv("LLM_API_KEY"),
       model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
   )
   ```

2. **HuggingFace 모델** — 오프라인 절차 블록 추가:
   ```
   > **오프라인 다운로드 필요**
   > 외부 PC: `huggingface-cli download {모델명} --local-dir ./모델명`
   > 폴더를 압축 후 메일로 전달받아 동일 경로에 압축 해제
   ```

3. **Reranking** — Cohere API(외부망 가능 시)와 BGE 리랭커(오프라인) 양쪽 모두 제시

4. **Docker 이미지** (Level 4) — `docker save`/`docker load` 절차 필수 포함:
   ```bash
   # 외부 PC (인터넷 가능)
   docker pull {이미지}:{태그} && docker save -o {파일명}.tar {이미지}:{태그}
   # 내부 PC (메일 전달 후)
   docker load -i {파일명}.tar
   ```

5. **Kind 이미지** — `kindest/node` 이미지 오프라인 로드 절차 포함

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| 특정 튜터 응답 실패 | 해당 섹션 없이 진행, 보고서에 누락 명시 후 단독 재호출 제안 |
| _workspace 파일 없음 | Phase 0에서 초기 실행으로 분기 |
| 학습 레벨 불명확 | curriculum-designer에게 레벨 평가 요청 |
| 코드 예제 실행 오류 (사용자 보고) | rag-tutor 재호출, 환경 설정 체크리스트 포함 |

## 참조 문서

상세 커리큘럼 내용은 다음 references/ 파일을 단계별로 로드:
- `references/curriculum-overview.md` — 전체 커리큘럼 개요 및 레벨별 목표
- `references/level1-basic-rag.md` — Level 1: 기초 RAG (2주)
- `references/level2-enhanced-rag.md` — Level 2: 향상된 RAG (3주)
- `references/level3-advanced-rag.md` — Level 3: 고급 RAG (3주)
- `references/level4-production-k8s.md` — Level 4: 프로덕션 K8S (4주)

## 테스트 시나리오

### 정상 흐름 (커리큘럼 초기 생성)
1. 사용자: "RAG 공부를 시작하고 싶어. 커리큘럼 만들어줘"
2. Phase 0: `_workspace/` 없음 → 초기 실행
3. Phase 1: 커리큘럼 요청 확인
4. Phase 2: 4명 팀 구성 + 6개 작업 등록
5. Phase 3: 팀원 자체 조율하며 각 섹션 작성
6. Phase 5: `_workspace/00_learning_roadmap.md` 생성, 팀 정리
7. 예상 결과: 12주 RAG 학습 커리큘럼 생성

### 정상 흐름 (학습 세션)
1. 사용자: "Level 2 하이브리드 검색 Chroma + BM25 코드 보여줘"
2. Phase 0: `_workspace/` 존재 → 부분 재실행
3. Phase 4: vectordb-tutor + rag-tutor 서브 에이전트 호출
4. 예상 결과: `_workspace/04_vectordb_guide_hybrid.md` 생성

### 에러 흐름
1. Phase 3에서 infra-tutor가 응답 실패
2. 리더가 해당 섹션 없이 나머지 통합
3. 보고서에 "Level 4 인프라 가이드 누락" 명시
4. 사용자에게 "infra-tutor만 재호출할까요?" 제안
