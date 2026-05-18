# RAG 학습 스터디 프로젝트

## 하네스: RAG 학습 스터디

**목표:** Python 개발자 / AIOps(K8S+Kind) / DA(Chroma, Elasticsearch) 배경으로 RAG 기초→프로덕션까지 12주 단계별 학습

**트리거:** RAG 학습, 커리큘럼, 실습, 구현, 배포, 벡터DB, 검색, LangChain, Kind 관련 요청 시 `rag-study-orchestrator` 스킬을 사용하라. 단순 개념 질문은 직접 응답 가능.

## 학습자 프로필

- **Python 개발자**: 파이썬 경험 있음, LangChain 학습 중
- **AIOps**: K8S/Kind 운영 경험 있음
- **DA**: Chroma DB, Elasticsearch 운영 경험 있음

## 환경 제약

**폐쇄망**: 외부 인터넷 접근 제한. 에이전트는 아래 규칙을 반드시 준수하라.

| 구분 | 가능 여부 | 비고 |
|------|---------|------|
| `pip install` | 가능 | 내부 PyPI 미러 사용 |
| 사내 LLM API 호출 | 가능 | OpenAI 호환 엔드포인트 사용 |
| OpenAI / Cohere 외부 API | **불가** | 폐쇄망으로 외부 API 접근 불가 |
| HuggingFace 모델 가중치 | **불가** | 외부 PC에서 다운로드 → 메일로 전달 필요 |
| Docker 이미지 pull | **불가** | 외부 PC에서 `docker save` → 메일로 전달 필요 |

**LLM/임베딩 표준 초기화 패턴** — 모든 코드 예제에 이 패턴을 사용하라:

```
# .env
LLM_BASE_URL=http://사내-llm-서버/v1
LLM_API_KEY=사내키
LLM_MODEL=모델명
EMBED_MODEL=BAAI/bge-m3
EMBED_CACHE_DIR=./models   # 상대경로 기준: python 명령을 실행하는 디렉토리
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    temperature=0,
)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    cache_folder=os.getenv("EMBED_CACHE_DIR", "./models"),
)
```

LLM은 사내 OpenAI-호환 엔드포인트(`LLM_BASE_URL`)에 연결하고, 임베딩은 `EMBED_CACHE_DIR`에 수동 설치된 HuggingFace 모델을 로드한다.

**에이전트 지침:**
- 모든 LLM/임베딩 초기화는 위 표준 패턴 사용 — `ChatOpenAI(model="gpt-4o-mini")` 같은 하드코딩 금지
- HuggingFace 모델이 필요한 실습에는 `> 오프라인 다운로드 필요` 블록과 다운로드 방법을 함께 제시
- Docker 이미지가 필요한 실습에는 `docker save`/`docker load` 절차를 포함
- Reranking: BGE 리랭커(오프라인, `EMBED_CACHE_DIR`에 수동 설치) 사용 — Cohere 외부 API 불가

## 에이전트 팀

| 에이전트 | 역할 |
|---------|------|
| `curriculum-designer` | 커리큘럼 설계 및 학습 계획 |
| `rag-tutor` | Python RAG 구현 가이드 |
| `infra-tutor` | K8S/Kind 인프라 가이드 |
| `vectordb-tutor` | Chroma/Elasticsearch 가이드 |

## 커리큘럼 개요

| 레벨 | 주제 | 기간 |
|------|------|------|
| Level 1 | 기초 RAG (LangChain + Chroma) | 2주 |
| Level 2 | 향상된 RAG (Elasticsearch + 하이브리드) | 3주 |
| Level 3 | 고급 RAG (Self-RAG, RAGAS) | 3주 |
| Level 4 | 프로덕션 K8S (Kind + 모니터링) | 4주 |

**변경 이력:**

| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-05-15 | 초기 구성 | 전체 | RAG 학습 스터디 하네스 신규 구축 |
| 2026-05-18 | 폐쇄망 환경 반영 | 환경 제약, 표준 패턴, 에이전트 지침 | 외부 API 불가 확정, 임베딩을 HuggingFace 로컬(`cache_folder`)로 전환, `load_dotenv()` 추가 |
