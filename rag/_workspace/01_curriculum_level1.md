# Level 1: 기초 RAG 2주 학습 플랜

> **작성일**: 2026-05-15
> **대상 기간**: 2주 (총 14일)
> **선수 지식**: Python, K8S/Kind, Chroma DB, Elasticsearch 운영 경험
> **학습 모드**: 한국어, 실습 중심

---

## 목차

1. [Level 1 목표](#1-level-1-목표)
2. [주차별 학습 계획](#2-주차별-학습-계획)
3. [학습자 배경 연결](#3-학습자-배경-연결-이미-아는-것-→-rag로-잇기)
4. [준비물 체크리스트](#4-준비물-체크리스트)
5. [Level 1 완료 기준](#5-level-1-완료-기준)

---

## 1. Level 1 목표

2주 후 다음을 **설명하고 + 직접 구현**할 수 있게 된다.

### 1.1 개념 목표 (설명할 수 있어야 함)

- RAG가 왜 필요한지 (LLM의 지식 한계, fine-tuning 대비 장점)
- RAG의 두 단계 구조: **Indexing(오프라인)** vs **Retrieval+Generation(온라인)**
- 임베딩(Embedding)이 무엇이며, 왜 검색에 쓰이는지 (코사인 유사도)
- 청킹(Chunking)의 필요성과 `chunk_size`, `chunk_overlap`의 trade-off
- Chroma의 "컬렉션"이 Elasticsearch "인덱스"와 어떻게 다른지

### 1.2 구현 목표 (직접 만들 수 있어야 함)

- LangChain으로 **PDF → 청킹 → 임베딩 → Chroma 저장 → 검색 → LLM 답변** 풀파이프라인 구현
- `persist_directory`로 Chroma DB 영속화 (재실행 시 임베딩 재계산 없음)
- "모르면 모른다고" 답하는 커스텀 프롬프트 템플릿 작성
- 답변과 함께 **근거 문서(source)** 출력

### 1.3 결과물

- `01_basic_rag.py` — 30줄 내외의 PDF Q&A 봇
- `chroma_db/` — 영속화된 Chroma 컬렉션
- `notes/level1_review.md` — 본인이 정리한 RAG 개념 노트

---

## 2. 주차별 학습 계획

### 2.1 1주차 (Day 1~7) — RAG 개념 + LangChain 기초

#### Day 1 — RAG가 왜 필요한가? (이론 100%)

**학습 주제**
- LLM의 한계 3가지: ① 지식 컷오프, ② 환각(hallucination), ③ 도메인 특화 부족
- 해결책 비교: Fine-tuning vs RAG vs Prompt만 사용
- RAG = "오픈북 시험" 비유

**학습 자료**
- [Anthropic — Retrieval Augmented Generation](https://docs.anthropic.com/) 개요 섹션
- LangChain 공식 RAG 튜토리얼 첫 페이지

**실습**
- 손으로 그려보기: 종이에 RAG 데이터 흐름도 그리기 (Indexing 단계 / Retrieval+Generation 단계)
- 결과물: `notes/day1_rag_overview.md` (자기 말로 정리)

**소요 시간 가이드**: 1~2시간

---

#### Day 2 — 임베딩(Embedding) 직접 만져보기

**학습 주제**
- 텍스트 → 숫자 벡터 변환 원리
- 코사인 유사도(cosine similarity) 수식과 의미
- OpenAI `text-embedding-3-small`의 1536 차원이 의미하는 것

**실습**
```python
# practice/day2_embedding.py
from openai import OpenAI
import numpy as np

client = OpenAI()
texts = [
    "RAG는 외부 문서를 활용하는 기법입니다",
    "검색 증강 생성은 LLM에 지식을 추가합니다",  # 비슷한 의미
    "오늘 날씨가 맑습니다",                       # 무관
]

emb = [client.embeddings.create(input=t, model="text-embedding-3-small").data[0].embedding for t in texts]

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"1-2 (유사): {cos_sim(emb[0], emb[1]):.3f}")  # ~0.85
print(f"1-3 (무관): {cos_sim(emb[0], emb[2]):.3f}")  # ~0.15
```

**검증 질문**
- 한국어 ↔ 영어 문장 임베딩 유사도는 어떻게 나오는가? (multilingual 특성 체감)
- 짧은 문장 vs 긴 문장 유사도 안정성?

**소요 시간 가이드**: 2시간

---

#### Day 3 — LangChain Document Loader

**학습 주제**
- `Document` 객체 구조: `page_content` + `metadata`
- `PyPDFLoader`, `TextLoader`, `WebBaseLoader` 비교
- 메타데이터에 어떤 정보를 담아야 검색이 풍성해지는가? (source, page, section)

**실습**
- 본인이 관심 있는 PDF 1개 준비 (예: 회사 매뉴얼, 기술 블로그 PDF 변환본)
- 페이지 수 / 첫 페이지 200자 / 메타데이터 출력
- TextLoader로 `.md` 파일도 로딩해보기

**소요 시간 가이드**: 1.5시간

---

#### Day 4 — Text Splitter (청킹 전략)

**학습 주제**
- `RecursiveCharacterTextSplitter` 동작 원리 (separators 우선순위)
- `chunk_size`(문자 기준 ≠ 토큰 기준) vs `chunk_overlap` trade-off
- 청크가 너무 작으면? 너무 크면?

**실습**
- 같은 PDF에 `chunk_size = 200 / 500 / 1500` 세 가지로 청킹 후 비교
- "한 청크 안에 완결된 의미가 있는가?"를 사람 눈으로 확인
- `chunks[0].metadata`에 자동으로 `source`/`page`가 보존되는지 확인

**DA 배경 연결 포인트**: 청킹은 ES에서 문서를 어떻게 인덱싱할지 결정하는 것과 같은 의사결정이다. ES의 `analyzer` 선택 = RAG의 `splitter` 선택.

**소요 시간 가이드**: 2시간

---

#### Day 5 — Chroma DB에 임베딩 저장

**학습 주제**
- `Chroma.from_documents()` 내부 동작 (각 청크 임베딩 → 컬렉션 저장)
- `persist_directory`의 중요성 — 재실행 시 임베딩 비용 절약
- LangChain이 자동 생성하는 컬렉션명: `"langchain"`

**실습**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)
print(vectorstore._collection.count())
```

**DA 배경 연결 포인트**: Chroma 컬렉션 ≒ ES 인덱스. 다만 schema-less에 가깝고, 임베딩 벡터 + 메타데이터 + 원문이 함께 저장된다.

**소요 시간 가이드**: 1.5시간

---

#### Day 6 — Retriever와 첫 RAG 체인

**학습 주제**
- `vectorstore.as_retriever(search_kwargs={"k": 3})` — 상위 k개 가져오기
- `RetrievalQA` 체인의 `chain_type="stuff"` 의미 (검색된 청크를 한 번에 프롬프트로 합치기)
- `return_source_documents=True`로 근거 추적

**실습**
- 첫 풀파이프라인 작성 (`practice/day6_first_rag.py`)
- 동일 질문에 `k=1`, `k=3`, `k=5`로 답변 품질 비교

**소요 시간 가이드**: 2시간

---

#### Day 7 — 1주차 통합 회고

**해야 할 일**
- 1주차 코드를 깔끔하게 정리 → `practice/week1_summary.py`
- 회고 노트 `notes/week1_review.md` 작성:
  - 가장 헷갈렸던 개념 1가지
  - 가장 의외였던 결과 1가지
  - 2주차에 시도하고 싶은 것 1가지
- (선택) 본인 도메인 PDF로 동일 파이프라인 한 번 더 돌려보기

**소요 시간 가이드**: 1.5시간

---

### 2.2 2주차 (Day 8~14) — Chroma 심화 + 프롬프트 + 프로젝트 완성

#### Day 8 — Chroma DB 재로딩 + 검색 타입

**학습 주제**
- 기존 DB 재로딩: `Chroma(persist_directory=..., embedding_function=...)`
- 검색 방식 3종: `similarity_search`, `similarity_search_with_score`, `max_marginal_relevance_search`
- MMR이 왜 다양성을 보장하는지 (이미 선택된 청크와의 유사도를 페널티로)

**실습**
```python
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

query = "RAG의 한계는?"
for doc, score in vectorstore.similarity_search_with_score(query, k=3):
    print(f"{score:.4f} | {doc.page_content[:80]}")
```

**소요 시간 가이드**: 1.5시간

---

#### Day 9 — Chroma 컬렉션 직접 다루기 + 메타데이터 필터

**학습 주제**
- `chromadb.PersistentClient`로 raw API 접근
- 메타데이터 필터: `where={"page": {"$gte": 1}}`
- LangChain `as_retriever`에서도 `search_kwargs={"filter": {...}}`로 동일 동작

**실습**
- "5페이지 이후만" 검색하기
- "특정 source 파일만" 검색하기 (PDF 여러 개일 때)

**DA 배경 연결 포인트**: ES의 `filter` 컨텍스트와 동일한 개념. 점수 계산 없이 빠르게 후보를 좁힌다.

**소요 시간 가이드**: 1.5시간

---

#### Day 10 — Prompt Engineering (기본형의 함정)

**학습 주제**
- 기본 RAG의 두 가지 문제:
  1. 검색이 틀려도 LLM이 그럴듯하게 답한다 (환각)
  2. 근거를 명시하지 않으면 답변을 신뢰할 수 없다
- 시스템 프롬프트로 행동 제약하기

**실습**
- "관련 정보 없으면 '문서에서 찾을 수 없습니다'라고 답하라"는 제약 추가
- 의도적으로 PDF와 무관한 질문 던지기 → 제대로 거절하는지 검증

**소요 시간 가이드**: 2시간

---

#### Day 11 — 커스텀 PromptTemplate 적용

**학습 주제**
- `PromptTemplate(input_variables=["context", "question"])` 구조
- `chain_type_kwargs={"prompt": PROMPT}`로 체인에 주입
- 한국어 답변 강제, 인용 형식 지정

**실습**
```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""다음 컨텍스트만 사용하여 한국어로 답하세요.
컨텍스트:
{context}

질문: {question}

규칙:
- 컨텍스트에 없으면 "문서에서 찾을 수 없습니다."
- 답변 끝에 [출처: 페이지 번호] 형식으로 근거 표시

답변:""",
    input_variables=["context", "question"]
)
```

**소요 시간 가이드**: 2시간

---

#### Day 12 — 프로젝트 통합: "나만의 PDF Q&A 봇" v1

**해야 할 일**
- `01_basic_rag.py` 한 파일에 통합 (30줄 목표)
- 요구사항 충족:
  - [ ] PDF 1개 이상 지원
  - [ ] Chroma 영속화 (`persist_directory="./chroma_db"`)
  - [ ] 근거 페이지 번호 표시
  - [ ] "모르면 모른다"고 답하는 프롬프트
- CLI로 질문 입력받는 간단한 `while` 루프 추가 (선택)

**소요 시간 가이드**: 3시간

---

#### Day 13 — 한계 체감 실험 (Level 2 진입을 위한 빌드업)

**학습 주제**
- 의도적으로 검색이 잘 안 되는 질문 만들기
  - 동의어 (예: "코드 → 소스 코드 → 스크립트")
  - 한국어 조사·어미 변형
  - 숫자 / 고유명사 검색
- "왜 벡터 검색이 이런 케이스에 약한가?" 직관 키우기

**실습**
- 실패 케이스 5개 수집 → `notes/level1_failures.md`에 기록
- 각 실패에 대해 "왜 실패했는가?" 가설 1줄씩

**다음 레벨 연결**: 이 실패들이 Level 2의 BM25 + Hybrid Search 학습 동기가 된다.

**소요 시간 가이드**: 2시간

---

#### Day 14 — Level 1 회고 및 마무리

**해야 할 일**
- 체크리스트(섹션 5) 전 항목 자가 점검
- `notes/level1_review.md`에 다음 정리:
  - 2주 동안 배운 것 5가지 (한 줄씩)
  - 가장 자신 있는 것 / 아직 흐릿한 것
  - Level 2에서 가장 기대되는 것
- 코드 정리 + (선택) GitHub에 푸시

**소요 시간 가이드**: 1.5시간

---

## 3. 학습자 배경 연결 (이미 아는 것 → RAG로 잇기)

학습자는 **Python 개발 + K8S/Kind + Chroma/ES 운영** 경험이 모두 있다. 이 자산을 Level 1에서 최대로 활용한다.

### 3.1 Python 개발자 경험 → RAG 코드

| 이미 아는 것 | Level 1에서 만나는 것 | 연결 포인트 |
|---|---|---|
| 가상환경, `pip` | `pip install langchain ...` | 새 패키지는 많지만 설치 방식은 동일 |
| 환경변수 관리 | `.env` + `python-dotenv` | `OPENAI_API_KEY` 관리에 그대로 적용 |
| 함수 합성 / 데코레이터 | LangChain 체인 | `RetrievalQA`는 결국 함수 합성의 추상화 |
| typing / pydantic | LangChain의 `Document`, `BaseModel` | 데이터 스키마 익숙함이 그대로 전이 |

**학습 가속 팁**: LangChain 객체를 처음 만나면 **`type(x).__mro__`나 `dir(x)`로 직접 까보기**. Python 개발자에게는 docs보다 빠르다.

### 3.2 Chroma DB 운영 경험 → RAG 검색

| 이미 아는 것 | Level 1에서 만나는 것 | 연결 포인트 |
|---|---|---|
| Chroma 컬렉션 생성/조회 | `Chroma.from_documents()` | LangChain은 Chroma의 wrapper일 뿐 |
| `PersistentClient` | `persist_directory` | 동일한 디스크 영속화 메커니즘 |
| `collection.query()` | `vectorstore.similarity_search()` | API 이름만 다를 뿐 같은 동작 |
| 메타데이터 필터 | `search_kwargs={"filter": {...}}` | `where` 절 그대로 사용 가능 |

**핵심**: Chroma 운영을 해본 사람이라면 **Day 5~9는 매우 빠르게 진행 가능**. 대신 "임베딩 모델 선택이 검색 품질에 미치는 영향"에 시간을 더 투자할 것.

### 3.3 Elasticsearch 운영 경험 → RAG 검색 (Level 2 빌드업)

ES 경험은 Level 1에서는 **개념 비교 도구**로 쓰고, 본격 활용은 Level 2부터.

| ES 개념 | RAG/Chroma 대응 | 차이점 |
|---|---|---|
| 인덱스(Index) | 컬렉션(Collection) | Chroma는 schema-less |
| `_doc` 매핑 | 메타데이터 dict | 자유 형식 JSON |
| `analyzer`(Nori 등) | `TextSplitter` | 토큰화 vs 청킹 (입자 크기 차이) |
| BM25 점수 | 코사인 유사도 점수 | 렉시컬 vs 시맨틱 |
| `bool` 쿼리 `filter` | `search_kwargs={"filter": ...}` | 거의 동일한 의미 |

**Day 13 실패 케이스 실험에서 ES 경험이 빛난다**: "이건 BM25라면 잡았을 텐데..."라는 직관이 Level 2 학습 동기로 직결.

### 3.4 K8S/Kind 운영 경험 → 인프라 (Level 4 빌드업)

Level 1에서는 **로컬 Python으로만 동작**한다. K8S는 Level 4의 영역.
다만 Day 7 / Day 14에 다음을 미리 의식해두면 좋다.

- "이 코드를 컨테이너화한다면 무엇이 환경변수로 빠져야 하는가?" (`OPENAI_API_KEY`, `CHROMA_PERSIST_DIR`)
- "Chroma의 `persist_directory`는 K8S에서 `PersistentVolume`이 되겠구나."
- "ES를 Level 2에서 띄울 때 docker-compose로 시작 → Level 4에서 ECK로 가는 길."

---

## 4. 준비물 체크리스트

### 4.1 필수 — 시작 전 반드시

- [ ] **Python 3.10+** 설치 확인 (`python --version`)
- [ ] **가상환경 생성**: `python -m venv .venv && source .venv/bin/activate`
- [ ] **OpenAI API Key** 발급 및 결제 수단 등록 (Level 1 예상 비용: $1~3)
  - `text-embedding-3-small` 사용 시 매우 저렴
- [ ] **`.env` 파일 작성**
  ```
  OPENAI_API_KEY=sk-...
  ```
- [ ] **패키지 설치**
  ```bash
  pip install \
    langchain \
    langchain-openai \
    langchain-community \
    chromadb \
    pypdf \
    python-dotenv \
    numpy
  ```
- [ ] **실습용 PDF 1~2개 준비** (본인 도메인 문서 권장. 없으면 Python 공식 튜토리얼 PDF)
- [ ] **에디터/IDE**: VSCode 또는 PyCharm (Jupyter도 가능)

### 4.2 권장 — 학습 효율을 높이는 것

- [ ] **`requirements.txt`** 작성 후 버전 고정 (재현성)
- [ ] **`pre-commit`으로 `black`/`ruff`** 설정 (개발자 본능 유지)
- [ ] **Git 저장소** 초기화 — 매일 커밋으로 진행 가시화
- [ ] **노트 도구**: Obsidian / Notion / 그냥 `.md` 파일 — 본인이 쓰던 것 OK

### 4.3 불필요 — Level 1에서는 아직 안 써도 됨

- Docker / Kind / kubectl  (Level 4)
- Elasticsearch / Nori  (Level 2)
- RAGAS / LangSmith  (Level 3)
- GPU  (OpenAI API 사용 시 로컬 GPU 불필요)

### 4.4 디렉토리 구조 (권장)

```
rag/
├── .env
├── .venv/
├── requirements.txt
├── 01_basic_rag.py            # 최종 산출물
├── chroma_db/                 # 영속화 디렉토리
├── data/
│   └── your_document.pdf
├── practice/                  # 일자별 연습 코드
│   ├── day2_embedding.py
│   ├── day6_first_rag.py
│   └── week1_summary.py
└── notes/                     # 회고 / 정리
    ├── day1_rag_overview.md
    ├── week1_review.md
    ├── level1_failures.md
    └── level1_review.md
```

---

## 5. Level 1 완료 기준

다음 **모든 항목**에 체크할 수 있어야 Level 2로 진입한다.

### 5.1 개념 이해 (말로 설명 가능)

- [ ] RAG = **Indexing + Retrieval + Generation** 3단계 구조를 그림으로 그리고 말로 설명할 수 있다
- [ ] 임베딩이 **숫자 벡터**임을 이해하고, 코사인 유사도를 직접 계산할 수 있다
- [ ] **청크 크기(`chunk_size`)와 겹침(`chunk_overlap`)의 trade-off**를 설명할 수 있다
- [ ] Chroma 컬렉션과 ES 인덱스의 **공통점과 차이점**을 3가지 이상 말할 수 있다
- [ ] 기본 RAG의 두 가지 한계(검색 실패 / 환각)를 본인 실험 예시로 설명할 수 있다

### 5.2 구현 능력 (코드 직접 작성 가능)

- [ ] PDF 로드 → 청킹 → Chroma 저장까지 보지 않고 작성 가능 (10~15분 내)
- [ ] 기존 Chroma DB를 **재로딩**할 수 있다 (임베딩 재계산 없이)
- [ ] 메타데이터 필터로 특정 페이지/소스만 검색할 수 있다
- [ ] `PromptTemplate`을 커스텀하여 `RetrievalQA` 체인에 적용할 수 있다
- [ ] 답변과 함께 **근거 문서(source_documents)** 를 출력할 수 있다

### 5.3 실습 산출물

- [ ] `01_basic_rag.py` — 30줄 내외, 동작하는 PDF Q&A 봇
- [ ] `chroma_db/` — 재실행 시 임베딩 재계산 없이 동작 확인
- [ ] 무관한 질문 → "문서에서 찾을 수 없습니다." 응답 확인
- [ ] 답변에 [출처: 페이지 N] 표기 확인

### 5.4 다음 레벨 진입 준비 (Level 2 동기 확보)

- [ ] **벡터 검색이 약한 케이스 5개 이상** 수집 (`notes/level1_failures.md`)
- [ ] 각 실패에 대한 가설을 작성했다
- [ ] "BM25라면 / 하이브리드 검색이라면 잡지 않았을까?" 라는 질문이 자연스럽게 떠오른다

### 5.5 진입 불가 시 처방

| 미달 항목 | 추가 학습 |
|---|---|
| 개념 이해 부족 | Day 1, Day 2를 반복하며 노트에 자기 말로 다시 쓰기 |
| 구현 미숙 | Day 12 프로젝트를 **빈 파일에서 처음부터** 1회 더 작성 |
| 실패 케이스 못 모음 | Day 13을 하루 더, 다양한 도메인 질문으로 |

---

## 부록: 매일 30분이 부족할 때의 우선순위

시간이 빡빡한 날에는 다음 우선순위로 자른다.

1. **실습 코드 실행** (최우선 — 손이 기억하게)
2. **회고 노트 1줄** (메타인지)
3. **읽기 자료**
4. **추가 실험 / 변형**

> "안 한 것보다 짧게라도 한 게 훨씬 낫다." — 14일 연속성이 Level 1의 진짜 성공 요인.
