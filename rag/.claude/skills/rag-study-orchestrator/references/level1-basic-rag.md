# Level 1: 기초 RAG (2주)

## 목차
- [학습 목표](#학습-목표)
- [1주차: RAG 개념 + LangChain 기초](#1주차-rag-개념--langchain-기초)
- [2주차: Chroma DB + 기초 RAG 완성](#2주차-chroma-db--기초-rag-완성)
- [핵심 실습 프로젝트](#핵심-실습-프로젝트)
- [체크리스트](#체크리스트)

---

## 학습 목표

2주 후 달성 목표:
- RAG가 왜 필요한지, 어떻게 동작하는지 설명 가능
- LangChain으로 PDF → 청킹 → Chroma → 검색 → 생성 파이프라인 구현
- 30줄 이내의 동작하는 RAG Q&A 봇 완성

---

## 1주차: RAG 개념 + LangChain 기초

### Day 1-2: RAG가 왜 필요한가?

**문제 상황 이해:**
- LLM은 학습 데이터 기준일 이후 정보를 모른다
- Fine-tuning은 비싸고 느리다
- RAG = LLM에게 "참고 문서"를 실시간으로 제공하는 방법

**RAG 동작 원리:**
```
[오프라인 단계 — Indexing]
문서 → 청킹(Chunking) → 임베딩(Embedding) → 벡터 DB 저장

[온라인 단계 — Retrieval + Generation]
질문 → 임베딩 → 벡터 DB에서 유사 청크 검색 → LLM에 "질문 + 검색 결과" 전달 → 답변 생성
```

**임베딩(Embedding)이란?**
- 텍스트를 숫자 벡터로 변환: "RAG는 검색 기반 생성 기법입니다" → [0.12, -0.34, 0.56, ...]
- 의미가 비슷한 텍스트 = 벡터 거리가 가깝다
- OpenAI `text-embedding-3-small` → 1536차원 벡터

**실습 1: 임베딩 직접 확인**
```python
# pip install openai python-dotenv
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("LLM_API_KEY"),
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# 두 문장의 임베딩 유사도 비교
texts = [
    "RAG는 외부 문서를 활용하는 기법입니다",
    "검색 증강 생성은 LLM에 지식을 추가합니다",  # 비슷한 의미
    "오늘 날씨가 맑습니다",                      # 다른 의미
]

embeddings = [
    client.embeddings.create(input=t, model=EMBED_MODEL).data[0].embedding
    for t in texts
]

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"문장 1-2 유사도: {cosine_similarity(embeddings[0], embeddings[1]):.3f}")  # ~0.85
print(f"문장 1-3 유사도: {cosine_similarity(embeddings[0], embeddings[2]):.3f}")  # ~0.15
```

### Day 3-4: LangChain 기본 컴포넌트

**설치:**
```bash
pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv
```

**Document Loader — 문서 읽기:**
```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

# PDF 로딩
loader = PyPDFLoader("document.pdf")
pages = loader.load()
print(f"페이지 수: {len(pages)}")
print(f"첫 페이지 내용: {pages[0].page_content[:200]}")
print(f"메타데이터: {pages[0].metadata}")  # {'source': 'document.pdf', 'page': 0}
```

**Text Splitter — 청킹:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 청크 최대 크기 (토큰 기준 아님, 문자 기준)
    chunk_overlap=50,   # 청크 간 겹치는 문자 수 (컨텍스트 연속성 유지)
    separators=["\n\n", "\n", ".", " "]  # 분리 우선순위
)

chunks = splitter.split_documents(pages)
print(f"총 청크 수: {len(chunks)}")
print(f"청크 예시: {chunks[0].page_content}")
```

**왜 청크 겹침(overlap)이 필요한가?**
- 청크 경계에서 중요한 문장이 잘릴 수 있다
- overlap이 있으면 두 청크에 걸친 개념도 검색 가능
- 실험 권장: overlap = chunk_size의 10%

### Day 5-7: 첫 RAG 파이프라인 완성

```python
# pip install langchain langchain-openai chromadb pypdf python-dotenv
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# 사내 LLM / 임베딩 설정 — .env만 수정하면 어떤 OpenAI 호환 서버에도 연결
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

# 1. 문서 로딩 및 청킹
loader = PyPDFLoader("your_document.pdf")
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
).split_documents(loader.load())

# 2. 벡터 DB 구축 (첫 실행 후 persist_directory 지정으로 재사용)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # 검색된 모든 청크를 하나의 프롬프트에 넣기
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True  # 근거 문서 반환
)

# 4. 질문
result = qa_chain.invoke({"query": "RAG의 핵심 구성 요소는 무엇인가요?"})
print(f"답변: {result['result']}")
print(f"근거 문서: {[d.metadata for d in result['source_documents']]}")
```

---

## 2주차: Chroma DB + 기초 RAG 완성

### Day 8-9: Chroma DB 심화

**기존 Chroma DB 재사용 (임베딩 비용 절약):**
```python
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 이미 만들어진 vectorstore 로딩 (임베딩 재계산 없음)
embeddings = OpenAIEmbeddings(
    base_url=os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# 검색 타입 비교
query = "RAG의 한계는?"

# 1. 기본 유사도 검색
sim_results = vectorstore.similarity_search(query, k=3)

# 2. 점수 포함 검색 (거리값 확인)
sim_with_score = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in sim_with_score:
    print(f"점수(낮을수록 유사): {score:.4f} | {doc.page_content[:100]}")

# 3. MMR (Maximal Marginal Relevance) — 다양성 고려
mmr_results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
```

**Chroma 컬렉션 직접 접근:**
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("langchain")  # LangChain이 사용하는 기본 컬렉션명

print(f"총 문서 수: {collection.count()}")

# 메타데이터로 필터링
results = collection.query(
    query_texts=["RAG 한계"],
    n_results=3,
    where={"page": {"$gte": 1}}  # 1페이지 이상만
)
```

### Day 10-12: RAG 개선 — Prompt Engineering

**기본 RAG의 문제:**
- 검색된 청크가 질문과 관련 없어도 무조건 사용
- 답변을 모를 때도 "생성"해버림

**개선된 프롬프트 템플릿:**
```python
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """당신은 주어진 컨텍스트만 사용하여 질문에 답하는 전문가입니다.

컨텍스트:
{context}

질문: {question}

답변 규칙:
- 컨텍스트에 관련 정보가 있으면 그것만 사용하여 답변하세요.
- 컨텍스트에 관련 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
- 답변에 근거 문장을 인용하세요.

답변:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatOpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
```

### Day 13-14: 실습 프로젝트 완성 + 회고

---

## 핵심 실습 프로젝트

**"나만의 PDF Q&A 봇" 완성**

요구사항:
1. PDF 파일 1개 이상 지원
2. Chroma DB 영속성 (재실행 시 임베딩 재계산 없음)
3. 근거 페이지 번호 표시
4. "모르면 모른다고" 하는 프롬프트

```
추천 문서: 파이썬 공식 문서 한국어 번역, 기술 블로그 PDF
```

---

## 체크리스트

**개념 이해**
- [ ] RAG = Indexing + Retrieval + Generation 구조 설명 가능
- [ ] 임베딩이 숫자 벡터임을 이해, 코사인 유사도 계산 가능
- [ ] 청크 크기와 overlap의 trade-off 설명 가능

**구현 능력**
- [ ] PDF → Chroma DB 인덱싱 코드 직접 작성 가능
- [ ] 기존 Chroma DB 재로딩 가능 (임베딩 비용 절약)
- [ ] 커스텀 프롬프트 템플릿 적용 가능

**실습 완성**
- [ ] 동작하는 PDF Q&A 봇 완성
- [ ] 근거 문서와 함께 답변 반환 확인
- [ ] 관련 없는 질문 시 "모른다" 응답 확인

**다음 단계 준비**
- [ ] "검색 결과가 항상 좋지 않다"는 한계 인식 → Level 2로 진행
