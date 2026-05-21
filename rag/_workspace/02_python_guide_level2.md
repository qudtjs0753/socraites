# Level 2: 향상된 RAG — 청킹 전략 + 하이브리드 검색 + Reranking

> **환경**: Python 3.10+, LangChain, Chroma, BM25, Ollama 로컬 LLM  
> **기간**: 3주  
> **목표**: 청킹 전략 실험 + Chroma+BM25 하이브리드 검색 + BGE Reranking

---

## 목차

1. [Level 2 시작 전 — Level 1의 한계](#1-level-2-시작-전--level-1의-한계)
2. [청킹 전략 심화 이론](#2-청킹-전략-심화-이론)
3. [청킹 전략 4가지 구현 비교](#3-청킹-전략-4가지-구현-비교)
4. [BM25 — 키워드 검색 이론](#4-bm25--키워드-검색-이론)
5. [BM25 + 벡터 하이브리드 검색 구현](#5-bm25--벡터-하이브리드-검색-구현)
6. [RRF(Reciprocal Rank Fusion) 이론 및 구현](#6-rrfreciprocal-rank-fusion-이론-및-구현)
7. [Reranking — 검색 후 재순위](#7-reranking--검색-후-재순위)
8. [청킹 전략 Recall 실험](#8-청킹-전략-recall-실험)
9. [전체 파이프라인 통합](#9-전체-파이프라인-통합)
10. [테스트 작성](#10-테스트-작성)
11. [체크리스트 및 다음 단계](#11-체크리스트-및-다음-단계)

---

## 1. Level 2 시작 전 — Level 1의 한계

### 1.1 Level 1 파이프라인의 문제점

Level 1에서 구현한 기본 RAG는 다음과 같은 한계가 있다.

**문제 1: 키워드 정확 일치 실패**

```
문서에 저장된 내용: "서버 503 오류가 오전 9:23에 발생했습니다."
사용자 질문:       "503 에러 언제 났어?"

벡터 검색의 한계:
- "503 오류"와 "503 에러"는 의미가 같지만 표현이 다르다.
- 더 심각한 문제: "9:23"이라는 숫자나 "503"이라는 코드는
  임베딩 공간에서 다른 숫자들과 잘 구분되지 않는다.
- 결과: 관련 없는 청크가 상위에 검색됨
```

**문제 2: 청크 경계에서 맥락 단절**

```
원본 문서:
  "RRF(Reciprocal Rank Fusion)는 여러 검색 결과를
   통합하는 방법입니다. 각 문서의 순위를 역수로 변환하여..."

고정 크기 500자 청킹 결과:
  청크 A: "RRF(Reciprocal Rank Fusion)는 여러 검색 결과를"
  청크 B: "통합하는 방법입니다. 각 문서의 순위를 역수로..."

문제: "RRF란?"이라는 질문에 청크 A는 정의를 담고 있으나,
      청크 B는 설명을 담고 있다. 청크 A만 검색되면 불완전한 답변.
```

**문제 3: 검색 다양성 부족**

```
k=3으로 검색하면 유사한 내용의 청크 3개가 반환될 수 있다.
→ 실제로는 하나의 청크와 동일한 정보를 LLM에 3번 전달하는 것과 같다.
→ 토큰 낭비 + 다각도 정보 누락
```

### 1.2 Level 2에서 해결하는 방법

| 문제 | Level 2 해결책 |
|------|--------------|
| 키워드 정확 일치 실패 | BM25 + 벡터 하이브리드 검색 |
| 청크 경계 맥락 단절 | 의미 단위 청킹, Parent-Child 청킹 |
| 검색 결과 중복 | MMR 검색, Reranking |
| 한국어 형태소 무시 | 한국어 분리자 최적화 |

---

## 2. 청킹 전략 심화 이론

### 2.1 청킹이 검색 품질에 미치는 영향

검색 시스템의 성능은 다음 공식으로 생각할 수 있다:

```
RAG 품질 = 검색 품질 × 생성 품질

검색 품질의 결정 요소 (대략적 가중치):
  청킹 전략:        30~40%
  임베딩 모델:      30%
  검색 알고리즘:    20%
  재순위(Reranking): 10~20%
```

청킹이 잘못되면 아무리 좋은 LLM을 사용해도 좋은 답변을 생성할 수 없다. 검색에서 놓친 정보는 생성 단계에서 복구할 수 없기 때문이다.

### 2.2 청크 크기와 검색 품질의 수학적 관계

**Precision(정밀도)와 Recall(재현율)의 트레이드오프:**

```
Precision@k = (검색된 k개 중 관련 있는 청크 수) / k
Recall@k    = (관련 있는 청크 중 검색된 청크 수) / (전체 관련 청크 수)

청크가 작을수록: Precision 높음, Recall 낮음
  → 관련 부분만 정확히 가져오지만, 분산된 정보는 못 가져옴

청크가 클수록: Precision 낮음, Recall 높음
  → 많은 정보를 가져오지만, 노이즈도 많이 포함
```

**최적 청크 크기 실험 방법:**

```python
# 개념적 실험
chunk_sizes = [100, 200, 300, 500, 800, 1000]
for size in chunk_sizes:
    # 각 크기로 청킹 후 Recall@5 측정
    recall = measure_recall(splitter(size), test_queries, ground_truth)
    print(f"chunk_size={size}: Recall@5={recall:.3f}")

# 일반적인 결과 (도메인에 따라 다름):
# chunk_size=100: 0.62
# chunk_size=200: 0.71
# chunk_size=300: 0.78  ← 자주 최적점
# chunk_size=500: 0.75
# chunk_size=800: 0.68
# chunk_size=1000: 0.61
```

### 2.3 청킹 전략별 특성 비교

```
┌──────────────────┬─────────────────────────────────────────────┐
│ 전략             │ 설명 및 적합한 문서 유형                      │
├──────────────────┼─────────────────────────────────────────────┤
│ 고정 크기        │ 단순, 빠름. 대부분의 경우 시작점.               │
│ (Fixed Size)     │ 구조화되지 않은 긴 문서에 적합.                │
├──────────────────┼─────────────────────────────────────────────┤
│ 의미 단위        │ 임베딩 유사도로 의미가 끊기는 지점 탐지.         │
│ (Semantic)       │ 의미가 풍부한 문서에 적합. API 비용 발생.        │
├──────────────────┼─────────────────────────────────────────────┤
│ 부모-자식        │ 작은 청크로 검색, 큰 청크를 컨텍스트로 제공.     │
│ (Parent-Child)   │ 정밀한 검색 + 풍부한 컨텍스트가 모두 필요할 때. │
├──────────────────┼─────────────────────────────────────────────┤
│ 문서 구조 기반   │ 헤더/섹션/단락 기준으로 분리.                   │
│ (Structure-aware)│ Markdown, 보고서 등 구조화된 문서에 최적.       │
└──────────────────┴─────────────────────────────────────────────┘
```

---

## 3. 청킹 전략 4가지 구현 비교

### 3.1 준비: 공통 유틸리티

```python
# 파일: chunking_strategies.py
# 목적: 4가지 청킹 전략 구현 및 비교
# 실행: python chunking_strategies.py

import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Level 1에서 작성한 모듈
from ollama_llm import create_embeddings
from loaders import load_directory

load_dotenv()

# 샘플 문서
SAMPLE_DOCUMENTS = [
    Document(
        page_content="""
RAG(Retrieval-Augmented Generation) 시스템 설계 문서

1. 개요

RAG는 LLM의 한계를 극복하기 위한 기법입니다. LLM은 학습 데이터 이후의 정보를 알지 못하며,
사내 문서에 대한 지식도 없습니다. RAG는 질문에 관련된 문서를 실시간으로 검색하여
LLM의 컨텍스트에 제공하는 방식으로 이 문제를 해결합니다.

2. 핵심 구성 요소

2.1 Document Loader
다양한 파일 형식(CSV, Excel, TXT, MD, LOG, 이미지)을 읽어 Document 객체로 변환합니다.
각 Document는 page_content(텍스트)와 metadata(출처 정보)를 포함합니다.

2.2 Text Splitter
긴 문서를 검색 가능한 크기의 청크로 분할합니다.
RecursiveCharacterTextSplitter를 사용하며, 단락 > 문장 > 단어 순서로 분리를 시도합니다.

2.3 Embedding Model
텍스트를 고차원 숫자 벡터로 변환합니다.
의미가 비슷한 텍스트는 벡터 공간에서 가깝게 위치합니다.
Ollama의 nomic-embed-text 모델은 768차원 벡터를 생성합니다.

2.4 Vector Store (Chroma)
임베딩 벡터를 저장하고 유사도 검색을 제공합니다.
로컬 디렉토리에 영속적으로 저장됩니다.

3. 성능 지표

3.1 Recall@k
정답 문서가 상위 k개 검색 결과에 포함되는 비율입니다.
k=5 기준으로 0.7 이상을 목표로 합니다.

3.2 Faithfulness
생성된 답변이 검색된 문서에 근거하는 비율입니다.
RAGAS 프레임워크로 자동 측정 가능합니다.
""",
        metadata={"source": "rag_design.md"}
    ),
    Document(
        page_content="""
장애 대응 매뉴얼

1. 서버 응답 지연 (503 오류)

증상:
- API 응답 시간 500ms 초과
- 503 Service Unavailable 응답

원인 분석:
- 데이터베이스 커넥션 풀 고갈
- 메모리 부족 (OOM)
- CPU 사용률 100% 지속

대응 절차:
1. 모니터링 대시보드에서 병목 지점 확인
2. 커넥션 풀 상태 점검: SHOW PROCESSLIST
3. 필요 시 애플리케이션 재시작
4. 커넥션 풀 크기 조정 (기본값 50 → 200)

재발 방지:
- 커넥션 풀 사용률 80% 알람 설정
- 주간 성능 리뷰 정례화

2. 디스크 용량 부족

임계값: 디스크 사용률 85% 초과 시 알람

대응:
1. 오래된 로그 파일 압축/삭제 (30일 이상)
2. Chroma DB 벡터 파일 정리 (사용하지 않는 컬렉션 삭제)
3. 필요 시 디스크 증설 요청
""",
        metadata={"source": "incident_manual.md"}
    ),
]
```

### 3.2 전략 1: 고정 크기 청킹 (Fixed Size)

```python
# 전략 1: 가장 단순한 방식
def strategy_fixed_size(docs: List[Document], chunk_size: int = 300) -> List[Document]:
    """
    고정 크기 청킹.

    장점: 단순하고 빠름. 대부분의 경우 시작점으로 좋음.
    단점: 문장/단락 경계를 무시할 수 있음.

    권장 사용:
        - 청킹 전략 실험의 baseline
        - 문서 구조가 단순한 경우
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,  # overlap = 10%
        separators=["\n\n", "\n", ".", "。", " ", ""],
    )
    return splitter.split_documents(docs)
```

### 3.3 전략 2: 의미 단위 청킹 (Semantic Chunking)

```python
# 전략 2: 임베딩 유사도 기반 의미 분리
# pip install langchain-experimental
def strategy_semantic(docs: List[Document]) -> List[Document]:
    """
    의미가 급격히 변하는 지점에서 청크를 분리한다.

    동작 원리:
    1. 각 문장을 임베딩한다.
    2. 인접 문장 간 코사인 유사도를 계산한다.
    3. 유사도가 급격히 낮아지는 지점(95 percentile)에서 분리한다.

    예시:
    문장1: "RAG는 검색 증강 생성입니다." → 벡터 A
    문장2: "임베딩은 텍스트를 벡터로 변환합니다." → 벡터 B
    문장3: "오늘 날씨가 맑습니다." → 벡터 C

    sim(A, B) = 0.82 → 연속 (같은 청크)
    sim(B, C) = 0.12 → 분리 지점 → 새 청크 시작

    장점: 의미 단위로 분리되어 청크의 응집도 높음.
    단점: 임베딩 API 호출 비용 발생, 청크 크기 불균일.

    주의: Ollama 임베딩으로 모든 문장을 임베딩하므로 시간이 걸림.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = create_embeddings()

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # 상위 5% 변화점에서 분리
    )

    all_chunks = []
    for doc in docs:
        # SemanticChunker는 텍스트 문자열 입력을 받음
        chunks = semantic_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata],
        )
        all_chunks.extend(chunks)

    return all_chunks
```

### 3.4 전략 3: 부모-자식 청킹 (Parent-Child)

```python
# 전략 3: 검색은 작은 청크로, 컨텍스트는 큰 청크로
def strategy_parent_child(
    docs: List[Document],
    vectorstore: Chroma,
) -> "ParentDocumentRetriever":
    """
    부모-자식 청킹 전략.

    핵심 아이디어:
    - 작은 청크(200자): 정확한 검색을 위해 벡터 DB에 저장
    - 큰 청크(2000자): 풍부한 컨텍스트를 위해 별도 저장소에 보관
    - 검색 결과: 작은 청크로 찾고, 부모(큰) 청크를 반환

    왜 이 방식이 효과적인가?
      질문: "RRF의 수식은?"
      작은 청크 검색: "RRF = Σ 1/(k+r_i)" → 정확히 찾음
      반환하는 청크: 부모(큰 청크) → "RRF는... 수식은... 예시는..." → 풍부한 컨텍스트

    주의: InMemoryStore는 재시작 시 내용이 사라짐.
           영속 저장이 필요하면 LocalFileStore 사용.
    """
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import InMemoryStore

    # 부모: 검색 결과로 반환할 큰 청크
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
    )
    # 자식: 검색에 사용할 작은 청크
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )

    store = InMemoryStore()  # 부모 청크 보관소

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,    # 자식 청크가 저장될 벡터 DB
        docstore=store,             # 부모 청크 저장소
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 문서 추가 (자식 청크 → 벡터 DB, 부모 청크 → store)
    retriever.add_documents(docs)

    return retriever


def demo_parent_child():
    """부모-자식 검색 동작 확인"""
    from langchain.storage import InMemoryStore

    embeddings = create_embeddings()
    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=embeddings,
    )

    retriever = strategy_parent_child(SAMPLE_DOCUMENTS, vectorstore)

    # 검색: 작은 청크로 찾지만, 부모(큰) 청크를 반환
    query = "503 오류 대응 방법"
    results = retriever.invoke(query)

    print(f"검색 쿼리: '{query}'")
    print(f"반환된 청크 수: {len(results)}")
    for doc in results:
        print(f"\n[{doc.metadata.get('source')}] ({len(doc.page_content)}자)")
        print(doc.page_content[:300] + "...")
```

### 3.5 전략 4: 마크다운 구조 기반 청킹

```python
# 전략 4: 문서 구조(헤더)를 기반으로 청킹
def strategy_markdown_aware(docs: List[Document]) -> List[Document]:
    """
    마크다운/구조화 문서의 헤더 기준 청킹.

    MD/TXT 파일에서 # 헤더를 기준으로 섹션을 분리한다.
    각 섹션이 하나의 의미 단위이므로 응집도가 높다.

    예시 입력:
    # 장애 대응 매뉴얼
    ## 503 오류
    대응 절차: ...
    ## 디스크 부족
    임계값: ...

    예시 출력:
    청크 1: "# 장애 대응 매뉴얼\n## 503 오류\n대응 절차: ..."
    청크 2: "## 디스크 부족\n임계값: ..."
    """
    from langchain.text_splitter import MarkdownHeaderTextSplitter

    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # 헤더를 청크에 포함
    )

    # 이후 고정 크기로 추가 분리 (너무 긴 섹션 처리)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    all_chunks = []
    for doc in docs:
        # 마크다운 구조로 1차 분리
        md_chunks = md_splitter.split_text(doc.page_content)

        # 각 청크에 부모 문서의 메타데이터 추가
        for md_chunk in md_chunks:
            merged_metadata = {**doc.metadata, **md_chunk.metadata}

            # 청크가 너무 크면 추가로 분리
            if len(md_chunk.page_content) > 500:
                sub_chunks = char_splitter.split_documents([
                    Document(
                        page_content=md_chunk.page_content,
                        metadata=merged_metadata,
                    )
                ])
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(Document(
                    page_content=md_chunk.page_content,
                    metadata=merged_metadata,
                ))

    return all_chunks


# ─────────────────────────────────────────────
# 비교 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    docs = SAMPLE_DOCUMENTS

    print("=== 청킹 전략 비교 ===")

    # 전략 1: 고정 크기
    fixed_chunks = strategy_fixed_size(docs, chunk_size=300)
    print(f"\n[고정 크기 (300자)]")
    print(f"  청크 수: {len(fixed_chunks)}")
    print(f"  평균 크기: {sum(len(c.page_content) for c in fixed_chunks)/len(fixed_chunks):.0f}자")

    # 전략 2: 의미 단위 (Ollama 임베딩 사용)
    print(f"\n[의미 단위 청킹]")
    print("  임베딩 계산 중... (시간이 걸립니다)")
    semantic_chunks = strategy_semantic(docs)
    print(f"  청크 수: {len(semantic_chunks)}")
    sizes = [len(c.page_content) for c in semantic_chunks]
    print(f"  크기 범위: {min(sizes)} ~ {max(sizes)}자")

    # 전략 4: 마크다운 구조 기반
    md_chunks = strategy_markdown_aware(docs)
    print(f"\n[마크다운 구조 기반]")
    print(f"  청크 수: {len(md_chunks)}")
    for chunk in md_chunks[:3]:
        print(f"  헤더 정보: {chunk.metadata}")
        print(f"  내용: {chunk.page_content[:80]}...")
```

**직접 작성해보세요 — 실습 과제 1:**

```python
# 실습 1-1: 청킹 전략별 청크 특성 분석
# 위 4가지 전략으로 동일한 문서를 청킹하고
# 각 청크의 길이 분포를 히스토그램으로 출력하는 코드를 작성하라.

import statistics

def analyze_chunks(chunks: List[Document], strategy_name: str):
    """청크 목록의 통계 정보를 출력한다."""
    sizes = [len(c.page_content) for c in chunks]

    print(f"\n=== {strategy_name} ===")
    print(f"  청크 수:     {len(chunks)}")
    print(f"  최솟값:      {min(sizes)}자")
    print(f"  최댓값:      {max(sizes)}자")
    print(f"  평균:        {statistics.mean(sizes):.0f}자")
    print(f"  중앙값:      {statistics.median(sizes):.0f}자")
    # TODO: 표준편차도 출력하라
    # 표준편차가 크면 청크 크기가 불균일하다는 의미
    pass

# 실습 1-2: 동일 쿼리로 청킹 전략별 검색 결과 비교
# 같은 질문을 각 청킹 전략으로 만든 Chroma DB에 검색하고
# 상위 3개 결과를 출력하여 품질을 직접 비교하라.

test_query = "서버 장애 대응 절차"

# TODO: 각 청킹 전략별 Chroma DB를 구축하고 동일 쿼리로 검색하라
# 힌트: Chroma(collection_name="fixed", embedding_function=embeddings)
```

---

## 4. BM25 — 키워드 검색 이론

### 4.1 TF-IDF에서 BM25로

BM25(Best Match 25)는 정보 검색 분야의 오랜 표준 알고리즘이다. TF-IDF를 개선한 버전이다.

**TF-IDF 기본 개념:**

```
TF(t, d)  = 문서 d에서 단어 t가 등장하는 횟수 / 문서 d의 총 단어 수
IDF(t)    = log(전체 문서 수 / 단어 t가 등장하는 문서 수)

TF-IDF(t, d) = TF(t, d) × IDF(t)

직관:
- TF: 문서 내에서 단어가 많이 나올수록 중요
- IDF: 모든 문서에 공통으로 나오는 단어는 덜 중요 (예: "는", "이", "의")
```

**BM25의 개선점:**

TF-IDF의 두 가지 문제를 수정했다:

1. **TF 포화(saturation) 문제**: 단어가 10번 나오는 문서가 1번 나오는 문서보다 10배 관련성이 높은 것은 아니다.

```
BM25에서 TF 포화 적용:
  tf_bm25 = tf × (k1 + 1) / (tf + k1)

  k1 = 1.5 (포화 속도 조절)

  tf=1 → tf_bm25 = 1×2.5 / (1+1.5) = 1.0
  tf=3 → tf_bm25 = 3×2.5 / (3+1.5) = 1.67  (3배가 아닌 1.67배)
  tf=10 → tf_bm25 = 10×2.5 / (10+1.5) = 2.17 (10배가 아닌 2.17배)
```

2. **문서 길이 정규화**: 긴 문서는 자연히 단어가 더 많이 등장한다.

```
BM25 최종 수식:

              tf × (k1 + 1)
score(t,d) = ─────────────────────────────── × IDF(t)
              tf + k1 × (1 - b + b × dl/avgdl)

k1 = 1.5 (TF 포화 파라미터)
b  = 0.75 (문서 길이 정규화 파라미터)
dl = 현재 문서의 단어 수
avgdl = 전체 문서의 평균 단어 수
```

### 4.2 BM25 vs 벡터 검색 비교

```
┌─────────────────────────────┬──────────────────┬──────────────────┐
│ 상황                         │ BM25             │ 벡터 검색         │
├─────────────────────────────┼──────────────────┼──────────────────┤
│ "503 오류" vs "503 에러"      │ 부분 실패        │ 성공 (의미 유사)   │
│ 정확한 키워드 일치            │ 강함             │ 약함             │
│ 고유명사, 코드, 버전번호       │ 강함             │ 약함             │
│ 의미 유사 질문                │ 약함             │ 강함             │
│ 오타, 축약어                  │ 약함             │ 강함             │
│ 한국어 형태소 변형            │ 형태소 분석 필요  │ 임베딩이 처리     │
│ 새 문서 (임베딩 미계산)        │ 바로 사용 가능   │ 임베딩 필요       │
│ 메모리 사용                   │ 적음             │ 많음             │
└─────────────────────────────┴──────────────────┴──────────────────┘
```

**결론**: BM25와 벡터 검색은 서로 보완적이다. 둘을 결합한 하이브리드 검색이 단독 사용보다 성능이 높다.

---

## 5. BM25 + 벡터 하이브리드 검색 구현

### 5.1 LangChain BM25Retriever 사용

```python
# 파일: hybrid_retriever.py
# 목적: BM25 + 벡터 하이브리드 검색 구현
# pip install rank-bm25 langchain-community

import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever

from ollama_llm import create_embeddings
from loaders import load_directory
from chunking_strategies import strategy_fixed_size

load_dotenv()


# ─────────────────────────────────────────────
# BM25 리트리버 구축
# ─────────────────────────────────────────────

def build_bm25_retriever(chunks: List[Document], k: int = 5) -> BM25Retriever:
    """
    BM25 키워드 검색 리트리버를 구축한다.

    BM25Retriever.from_documents는 내부적으로 rank-bm25 라이브러리를 사용한다.
    별도의 서버 없이 메모리에서 동작한다.

    한계: 재시작 시 재구축 필요 (메모리 기반).
          대용량 문서는 LangChain Community의 BM25RetrieverWithPickle 고려.
    """
    def korean_tokenizer(text: str) -> List[str]:
        """
        한국어 기본 토크나이저.
        공백 기준으로 토큰화하고 불필요한 조사/어미를 일부 제거.

        한계: 형태소 분석기(Mecab, Okt) 없이는 완전한 한국어 처리 불가.
              Level 3에서 Elasticsearch + Nori를 사용하면 더 정확함.
        """
        # 소문자 변환 + 공백 분리 (영어 대비)
        tokens = text.lower().split()
        # 너무 짧은 토큰 제거 (조사, 어미 등)
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

    retriever = BM25Retriever.from_documents(
        chunks,
        k=k,
        preprocess_func=korean_tokenizer,
    )
    return retriever


# ─────────────────────────────────────────────
# 벡터 리트리버 구축
# ─────────────────────────────────────────────

def build_vector_retriever(
    chunks: List[Document],
    persist_dir: str = "./chroma_hybrid",
    k: int = 5,
):
    """Chroma 벡터 검색 리트리버"""
    embeddings = create_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="hybrid_test",
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ─────────────────────────────────────────────
# 앙상블 리트리버 (하이브리드)
# ─────────────────────────────────────────────

def build_ensemble_retriever(
    chunks: List[Document],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 5,
) -> EnsembleRetriever:
    """
    벡터 검색과 BM25를 결합한 앙상블 리트리버.

    내부적으로 RRF(Reciprocal Rank Fusion)를 사용하여
    두 검색 결과를 통합한다.

    가중치 권장값:
        일반 문서: vector=0.6, bm25=0.4
        코드/로그: vector=0.4, bm25=0.6 (키워드 중요)
        의미 검색: vector=0.7, bm25=0.3
    """
    vector_retriever = build_vector_retriever(chunks, k=k)
    bm25_retriever = build_bm25_retriever(chunks, k=k)

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )
    return ensemble


# ─────────────────────────────────────────────
# 검색 결과 비교
# ─────────────────────────────────────────────

def compare_retrievers(chunks: List[Document], query: str):
    """같은 쿼리로 단독 검색과 하이브리드 검색을 비교한다."""
    from langchain.schema import Document

    k = 5

    # BM25만
    bm25_r = build_bm25_retriever(chunks, k=k)
    bm25_results = bm25_r.invoke(query)

    # 벡터만
    vector_r = build_vector_retriever(chunks, persist_dir="./temp_chroma", k=k)
    vector_results = vector_r.invoke(query)

    # 하이브리드
    ensemble = build_ensemble_retriever(chunks, k=k)
    hybrid_results = ensemble.invoke(query)

    print(f"\n쿼리: '{query}'")
    print(f"{'─'*60}")

    print(f"\n[BM25만] ({len(bm25_results)}개)")
    for i, doc in enumerate(bm25_results[:3], 1):
        print(f"  {i}. {doc.page_content[:80]}...")

    print(f"\n[벡터만] ({len(vector_results)}개)")
    for i, doc in enumerate(vector_results[:3], 1):
        print(f"  {i}. {doc.page_content[:80]}...")

    print(f"\n[하이브리드] ({len(hybrid_results)}개)")
    for i, doc in enumerate(hybrid_results[:3], 1):
        print(f"  {i}. {doc.page_content[:80]}...")


if __name__ == "__main__":
    from chunking_strategies import SAMPLE_DOCUMENTS

    # 청킹
    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    print(f"청크 수: {len(chunks)}")

    # 검색 비교
    compare_retrievers(chunks, "503 오류 대응 방법")
    compare_retrievers(chunks, "임베딩 벡터 차원")
```

**직접 작성해보세요 — 실습 과제 2:**

```python
# 실습 2-1: 가중치 실험
# EnsembleRetriever의 vector_weight와 bm25_weight를 조정하면서
# 검색 결과가 어떻게 달라지는지 확인하라.

# 테스트할 쿼리들
queries = [
    "503 에러 발생 시 절차",           # 코드/숫자 포함 → BM25 유리
    "검색 증강 생성 기법의 장점",        # 의미 검색 → 벡터 유리
    "디스크 용량 85% 임계값 알람",       # 정확한 수치 → BM25 유리
]

weight_configs = [
    (0.8, 0.2),  # 벡터 강조
    (0.6, 0.4),  # 기본 균형
    (0.4, 0.6),  # BM25 강조
    (0.2, 0.8),  # BM25 극강조
]

# TODO: 각 설정으로 위 쿼리들을 검색하고 결과를 비교하라
# 어떤 쿼리에 어떤 가중치가 적합한지 분석하라

# 실습 2-2: 한국어 토크나이저 개선
# korean_tokenizer 함수를 개선하여 다음을 처리하라:
# 1. "서버가" → "서버" (조사 '가' 제거)
# 2. "발생했습니다" → "발생" (어미 제거)
# 힌트: 단순 규칙 기반도 괜찮다 (접미어 제거)

def improved_korean_tokenizer(text: str) -> List[str]:
    """
    개선된 한국어 토크나이저.

    단순 규칙:
    - 2자 이상 토큰만 유지
    - 공통 어미/조사 접미어 제거 (이, 가, 를, 은, 는, 이/가, 을/를 등)
    """
    common_suffixes = ["이다", "습니다", "합니다", "했다", "입니다", "이고", "이며"]
    tokens = text.lower().split()

    processed = []
    for token in tokens:
        # TODO: common_suffixes 제거 로직 구현
        processed.append(token)

    return [t for t in processed if len(t) > 1]
```

---

## 6. RRF(Reciprocal Rank Fusion) 이론 및 구현

### 6.1 RRF 이론

여러 검색 시스템의 결과를 어떻게 합칠 것인가? 단순히 점수를 더하는 것은 문제가 있다. 각 시스템의 점수 스케일이 다르기 때문이다 (BM25 점수는 0~10, 벡터 거리는 0~2).

**RRF 수식:**

```
RRF_score(d) = Σᵢ 1 / (k + rᵢ(d))

d      = 문서
rᵢ(d)  = i번째 검색 시스템에서 문서 d의 순위 (1부터 시작)
k      = 상수 (보통 60, 낮은 순위 문서의 영향 조절)
Σᵢ    = 모든 검색 시스템에 대한 합

예시:
  문서 A: BM25에서 1위, 벡터 검색에서 3위
    RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

  문서 B: BM25에서 5위, 벡터 검색에서 1위
    RRF = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

  문서 C: BM25에서 2위, 벡터 검색에서 2위
    RRF = 1/(60+2) + 1/(60+2) = 0.0161 + 0.0161 = 0.0323

  최종 순위: A ≈ C > B

핵심 인사이트:
- 한 시스템에서 1위인 것보다 양쪽에서 중간 순위가 더 높을 수 있다
- 절대적인 점수(score)가 아닌 상대적인 순위(rank)를 사용하므로
  스케일 차이 문제를 자동으로 해결한다
- k=60은 상위 60위 이하 문서들의 영향을 평탄화한다
```

### 6.2 RRF 직접 구현 (이해를 위해)

```python
# 파일: rrf_demo.py
# 목적: RRF 알고리즘을 직접 구현하여 이해

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ScoredDocument:
    """점수가 있는 문서"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


def rrf_fusion(
    result_lists: List[List[ScoredDocument]],
    k: int = 60,
) -> List[ScoredDocument]:
    """
    여러 검색 결과 리스트를 RRF로 통합한다.

    Args:
        result_lists: 각 검색 시스템의 결과 리스트
                      result_lists[0] = BM25 결과 (순서가 순위)
                      result_lists[1] = 벡터 검색 결과
        k: RRF 상수 (기본값 60)

    Returns:
        RRF 점수 기준으로 정렬된 통합 결과
    """
    # content를 키로 사용하여 RRF 점수 합산
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, ScoredDocument] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.content[:100]  # 내용 앞부분을 키로 사용
            rrf_score = 1.0 / (k + rank)

            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                doc_map[key] = doc

            rrf_scores[key] += rrf_score

    # RRF 점수 기준 정렬
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.score = rrf_scores[key]
        result.append(doc)

    return result


def demo_rrf():
    """RRF 동작 확인"""

    # 두 검색 시스템의 결과 (순서 = 순위)
    bm25_results = [
        ScoredDocument("503 오류 대응 절차: 모니터링 확인 후 재시작"),    # 1위
        ScoredDocument("서버 응답 지연 원인: 커넥션 풀 고갈"),             # 2위
        ScoredDocument("임베딩 모델은 768차원 벡터를 생성합니다"),          # 3위
        ScoredDocument("RAG는 검색 증강 생성 기법입니다"),                  # 4위
        ScoredDocument("디스크 사용률 85% 초과 시 알람 설정"),              # 5위
    ]

    vector_results = [
        ScoredDocument("RAG는 검색 증강 생성 기법입니다"),                  # 1위
        ScoredDocument("임베딩 모델은 768차원 벡터를 생성합니다"),          # 2위
        ScoredDocument("503 오류 대응 절차: 모니터링 확인 후 재시작"),      # 3위
        ScoredDocument("Chroma는 오픈소스 벡터 데이터베이스입니다"),         # 4위
        ScoredDocument("서버 응답 지연 원인: 커넥션 풀 고갈"),               # 5위
    ]

    print("=== RRF 통합 전 ===")
    print("\nBM25 결과 (순위대로):")
    for i, doc in enumerate(bm25_results, 1):
        print(f"  {i}. {doc.content[:60]}...")

    print("\n벡터 검색 결과 (순위대로):")
    for i, doc in enumerate(vector_results, 1):
        print(f"  {i}. {doc.content[:60]}...")

    print("\n=== RRF 통합 후 ===")
    fused = rrf_fusion([bm25_results, vector_results], k=60)
    for i, doc in enumerate(fused, 1):
        print(f"  {i}. (RRF={doc.score:.5f}) {doc.content[:60]}...")


if __name__ == "__main__":
    demo_rrf()
```

**직접 작성해보세요 — 실습 과제 3:**

```python
# 실습 3-1: k 파라미터 영향 분석
# RRF에서 k 값(기본 60)을 10, 30, 60, 120으로 변경하면서
# 순위 변화를 분석하라.
# k가 작을수록 상위 순위 문서의 영향이 커진다.

# 실습 3-2: 가중 RRF 구현
# 각 검색 시스템에 다른 가중치를 적용하는 Weighted RRF를 구현하라.
# 가중치를 RRF 점수에 곱하여 반영한다.

def weighted_rrf_fusion(
    result_lists: List[List[ScoredDocument]],
    weights: List[float],
    k: int = 60,
) -> List[ScoredDocument]:
    """
    가중치가 적용된 RRF.

    수식: RRF_score(d) = Σᵢ wᵢ / (k + rᵢ(d))

    wᵢ = i번째 시스템의 가중치
    """
    # TODO: rrf_fusion을 기반으로 가중치 적용 버전 구현
    pass
```

---

## 7. Reranking — 검색 후 재순위

### 7.1 왜 Reranking이 필요한가?

```
기본 벡터 검색의 문제:
  질문: "RAG에서 청킹이 왜 중요한가?"

  검색 결과 (순위대로):
  1위: "청킹은 문서를 분할하는 과정입니다." (청킹 단어 포함, 유사도 높음)
  2위: "청크 크기가 500자일 때 최적이었습니다." (실험 결과)
  3위: "청킹이 검색 품질의 60%를 결정합니다." ← 실제로 가장 관련 있음

  문제: 임베딩은 "청킹"이라는 단어 빈도에 영향받지만,
        실제 질문 의도("왜 중요한가")와의 연관성을 잘 잡지 못한다.
```

**Reranker의 해결책:**

Reranker는 (질문, 문서) 쌍을 입력으로 받아 관련성 점수를 출력한다. 단순 벡터 유사도보다 정교한 판단을 한다.

```
벡터 검색: 질문 벡터 → 문서 벡터들과 거리 계산 (O(n))
Reranker:  (질문, 문서) 쌍을 동시에 보고 교차 주의(cross-attention) 적용 (O(n×k))

Cross-encoder(Reranker) 동작:
  입력: [CLS] 질문 텍스트 [SEP] 문서 텍스트 [SEP]
  출력: 0~1 관련성 점수

  "RAG에서 청킹이 왜 중요한가?" + "청킹이 검색 품질의 60%를 결정합니다."
  → Reranker 점수: 0.92 (매우 관련 있음)

  "RAG에서 청킹이 왜 중요한가?" + "청킹은 문서를 분할하는 과정입니다."
  → Reranker 점수: 0.71 (어느 정도 관련)
```

### 7.2 BGE Reranker 사용 (오프라인 HuggingFace 모델)

```
오프라인 다운로드 필요 (외부 PC에서 실행):
  from huggingface_hub import snapshot_download
  snapshot_download("BAAI/bge-reranker-v2-m3", cache_dir="./hf_cache")

  hf_cache/ 디렉토리를 tar로 묶어 서버로 전달:
  tar czf hf_cache.tar.gz hf_cache/
```

```python
# 파일: reranker.py
# 목적: BGE Reranker를 사용한 검색 후 재순위
# pip install FlagEmbedding huggingface_hub

import os
from typing import List, Tuple
from langchain.schema import Document

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def load_reranker():
    """
    BGE Reranker 모델을 로컬 캐시에서 로드한다.

    FlagEmbedding의 FlagReranker는 cross-encoder 방식으로
    (query, document) 쌍의 관련성 점수를 계산한다.

    use_fp16=True: 속도↑ 메모리↓ (정확도 약간 감소)
    """
    from huggingface_hub import snapshot_download
    from FlagEmbedding import FlagReranker

    # 로컬 캐시에서 모델 경로 가져오기 (Hub 접속 없음)
    model_path = snapshot_download(
        RERANKER_MODEL,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )

    reranker = FlagReranker(model_path, use_fp16=True)
    return reranker


def rerank_documents(
    query: str,
    documents: List[Document],
    reranker,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    검색된 문서들을 Reranker로 재순위화한다.

    일반적인 사용 패턴:
    1. 벡터/BM25로 후보 20~50개를 검색 (recall 최대화)
    2. Reranker로 상위 5개로 압축 (precision 최대화)

    Args:
        query: 사용자 질문
        documents: 1차 검색된 후보 문서 목록
        reranker: FlagReranker 인스턴스
        top_k: 재순위 후 반환할 문서 수

    Returns:
        (문서, 재순위 점수) 튜플 목록
    """
    if not documents:
        return []

    # (query, document) 쌍 생성
    pairs = [(query, doc.page_content) for doc in documents]

    # 관련성 점수 계산
    scores = reranker.compute_score(pairs, normalize=True)  # 0~1 정규화

    # 점수와 문서를 묶어 정렬
    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    return [(doc, score) for score, doc in scored_docs[:top_k]]


# ─────────────────────────────────────────────
# Reranker 없이 LLM으로 재순위 (대안)
# ─────────────────────────────────────────────

def llm_rerank(
    query: str,
    documents: List[Document],
    llm,
    top_k: int = 3,
) -> List[Document]:
    """
    LLM을 사용한 재순위 (BGE Reranker 대안).

    BGE Reranker를 사용할 수 없는 경우 Ollama LLM으로 대체.
    비용이 더 높고 느리지만 추가 모델 없이 사용 가능.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if not documents:
        return []

    # 문서 목록을 번호와 함께 포맷팅
    doc_list = "\n\n".join([
        f"[{i+1}] {doc.page_content[:300]}"
        for i, doc in enumerate(documents)
    ])

    prompt = ChatPromptTemplate.from_template("""
다음 질문과 문서 목록이 있습니다.
질문에 가장 관련 있는 문서 {top_k}개의 번호를 쉼표로 나열하세요.

질문: {query}

문서 목록:
{doc_list}

가장 관련 있는 {top_k}개 번호 (예: 3,1,5):""")

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "query": query,
        "doc_list": doc_list,
        "top_k": top_k,
    })

    # 번호 파싱
    try:
        indices = [
            int(n.strip()) - 1
            for n in response.strip().split(",")
            if n.strip().isdigit()
        ]
        # 유효한 인덱스만
        valid_indices = [i for i in indices if 0 <= i < len(documents)]
        return [documents[i] for i in valid_indices[:top_k]]
    except Exception:
        # 파싱 실패 시 원본 순서 반환
        return documents[:top_k]


# ─────────────────────────────────────────────
# 전체 파이프라인: 하이브리드 검색 + Reranking
# ─────────────────────────────────────────────

def hybrid_retrieve_and_rerank(
    query: str,
    ensemble_retriever,
    reranker,
    initial_k: int = 20,  # 1차 검색 후보 수
    final_k: int = 5,     # Reranking 후 반환 수
) -> List[Document]:
    """
    하이브리드 검색 + Reranking 전체 파이프라인.

    Step 1: EnsembleRetriever로 20개 후보 검색 (높은 Recall)
    Step 2: Reranker로 5개로 압축 (높은 Precision)
    """
    # 1차 검색 (많은 후보)
    # EnsembleRetriever의 k는 생성 시 설정되어 있음
    candidates = ensemble_retriever.invoke(query)[:initial_k]

    if not candidates:
        return []

    # Reranking
    reranked = rerank_documents(query, candidates, reranker, top_k=final_k)

    return [doc for doc, score in reranked]


if __name__ == "__main__":
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size
    from hybrid_retriever import build_ensemble_retriever
    from ollama_llm import create_llm

    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=200)
    ensemble = build_ensemble_retriever(chunks, k=10)
    llm = create_llm()

    query = "503 오류 발생 시 커넥션 풀 조정 방법"

    print("=== LLM 재순위 (BGE Reranker 대안) ===")
    initial_candidates = ensemble.invoke(query)[:10]
    reranked_docs = llm_rerank(query, initial_candidates, llm, top_k=3)

    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n{i}위. [{doc.metadata.get('source')}]")
        print(f"   {doc.page_content[:120]}...")
```

**직접 작성해보세요 — 실습 과제 4:**

```python
# 실습 4-1: Reranking 전후 비교
# 동일 쿼리에 대해 Reranking 전후의 결과를 출력하고
# 순위 변화를 분석하라.

def compare_before_after_rerank(query: str, ensemble_retriever, llm, k: int = 5):
    """
    Reranking 전후의 결과를 비교 출력한다.
    BGE Reranker 대신 LLM 재순위를 사용.
    """
    # 1차 검색
    initial = ensemble_retriever.invoke(query)[:k*3]

    print(f"쿼리: '{query}'")
    print(f"\n[Reranking 전] 하이브리드 검색 결과 ({len(initial)}개):")
    for i, doc in enumerate(initial, 1):
        print(f"  {i}. {doc.page_content[:80]}...")

    # Reranking
    # TODO: llm_rerank 호출하여 재순위화
    reranked = None  # 여기에 구현

    print(f"\n[Reranking 후] 상위 {k}개:")
    # TODO: 재순위 결과 출력

# 실습 4-2: 점수 기반 필터링
# LLM 재순위에서 모든 후보 문서를 순위 없이 점수로 평가하는
# 방식으로 바꿔보라.
# 각 문서에 대해 "0~10점 사이의 관련성 점수를 매겨라"라고 LLM에 물어본다.

def score_documents_with_llm(
    query: str,
    documents: List[Document],
    llm,
) -> List[Tuple[Document, float]]:
    """
    LLM에게 각 문서의 관련성 점수(0~10)를 요청한다.
    """
    # TODO: 각 문서에 대해 LLM에게 점수를 요청하고
    # (문서, 점수) 튜플 리스트를 반환하라
    pass
```

---

## 8. 청킹 전략 Recall 실험

### 8.1 Recall 측정 방법

```python
# 파일: chunking_experiment.py
# 목적: 청킹 전략별 Recall@k 측정 실험
# 실행: python chunking_experiment.py

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma

from ollama_llm import create_embeddings
from chunking_strategies import (
    SAMPLE_DOCUMENTS,
    strategy_fixed_size,
    strategy_markdown_aware,
)

load_dotenv()


# ─────────────────────────────────────────────
# 평가 데이터셋 정의
# ─────────────────────────────────────────────

# 각 질문에 대해 정답 문서(ground truth)를 정의
# 실제 프로젝트에서는 도메인 전문가가 작성
EVAL_DATASET = [
    {
        "question": "503 오류 발생 시 대응 절차는?",
        "relevant_sources": ["incident_manual.md"],  # 이 파일의 내용이 검색되어야 함
        "relevant_keywords": ["503", "대응", "절차", "재시작"],
    },
    {
        "question": "디스크 용량 임계값 알람 설정",
        "relevant_sources": ["incident_manual.md"],
        "relevant_keywords": ["디스크", "85%", "알람"],
    },
    {
        "question": "임베딩 모델의 벡터 차원 수는?",
        "relevant_sources": ["rag_design.md"],
        "relevant_keywords": ["임베딩", "768", "차원"],
    },
    {
        "question": "RAG 시스템의 Recall@k 목표치",
        "relevant_sources": ["rag_design.md"],
        "relevant_keywords": ["Recall", "0.7"],
    },
]


# ─────────────────────────────────────────────
# Recall 측정 함수
# ─────────────────────────────────────────────

def measure_recall_at_k(
    vectorstore: Chroma,
    query: str,
    relevant_sources: List[str],
    k: int = 5,
) -> float:
    """
    주어진 쿼리에 대해 Recall@k를 계산한다.

    Recall@k = (검색된 k개 중 relevant source를 가진 문서 수) /
               (relevant source 총 수)

    Args:
        vectorstore: 검색 대상 벡터 DB
        query: 검색 쿼리
        relevant_sources: 정답 파일명 목록
        k: 검색할 문서 수

    Returns:
        0.0 ~ 1.0 사이의 Recall 점수
    """
    results = vectorstore.similarity_search(query, k=k)
    retrieved_sources = {doc.metadata.get("source") for doc in results}

    # relevant_sources 중 몇 개가 검색되었는가
    found = len(set(relevant_sources) & retrieved_sources)
    total = len(set(relevant_sources))

    return found / total if total > 0 else 0.0


def evaluate_chunking_strategy(
    strategy_name: str,
    chunks: List[Document],
    eval_dataset: List[Dict],
    k: int = 5,
    collection_name: str = "eval",
) -> Dict[str, float]:
    """
    청킹 전략의 평균 Recall@k를 측정한다.
    """
    embeddings = create_embeddings()

    # 임시 Chroma DB 구축
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )

    recalls = []
    for item in eval_dataset:
        recall = measure_recall_at_k(
            vectorstore=vectorstore,
            query=item["question"],
            relevant_sources=item["relevant_sources"],
            k=k,
        )
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls)

    print(f"\n[{strategy_name}] Recall@{k}:")
    for i, (item, recall) in enumerate(zip(eval_dataset, recalls)):
        print(f"  Q{i+1}: {item['question'][:40]}... → {recall:.2f}")
    print(f"  평균: {avg_recall:.4f}")

    # 임시 DB 삭제
    vectorstore.delete_collection()

    return {"strategy": strategy_name, "avg_recall": avg_recall, "recalls": recalls}


def run_all_experiments():
    """
    모든 청킹 전략에 대해 Recall@5 실험을 실행한다.
    """
    print("=== 청킹 전략 Recall@5 실험 ===")
    print(f"문서 수: {len(SAMPLE_DOCUMENTS)}")
    print(f"평가 질문 수: {len(EVAL_DATASET)}")

    results = []

    # 전략 1: 고정 크기 (100자)
    chunks_100 = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=100)
    r1 = evaluate_chunking_strategy(
        "고정크기 100자", chunks_100, EVAL_DATASET,
        collection_name="eval_100"
    )
    results.append(r1)

    # 전략 2: 고정 크기 (300자)
    chunks_300 = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    r2 = evaluate_chunking_strategy(
        "고정크기 300자", chunks_300, EVAL_DATASET,
        collection_name="eval_300"
    )
    results.append(r2)

    # 전략 3: 고정 크기 (500자)
    chunks_500 = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=500)
    r3 = evaluate_chunking_strategy(
        "고정크기 500자", chunks_500, EVAL_DATASET,
        collection_name="eval_500"
    )
    results.append(r3)

    # 전략 4: 마크다운 구조 기반
    chunks_md = strategy_markdown_aware(SAMPLE_DOCUMENTS)
    r4 = evaluate_chunking_strategy(
        "마크다운 구조", chunks_md, EVAL_DATASET,
        collection_name="eval_md"
    )
    results.append(r4)

    # 결과 요약
    print("\n=== 최종 비교 ===")
    results.sort(key=lambda x: x["avg_recall"], reverse=True)
    for i, r in enumerate(results, 1):
        print(f"  {i}위. {r['strategy']}: Recall@5 = {r['avg_recall']:.4f}")

    # JSON으로 저장
    with open("chunking_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: chunking_experiment_results.json")

    return results


if __name__ == "__main__":
    run_all_experiments()
```

**직접 작성해보세요 — 실습 과제 5:**

```python
# 실습 5-1: 하이브리드 검색 vs 벡터 단독 Recall 비교
# EnsembleRetriever(하이브리드)와 단독 벡터 검색의 Recall@5를 비교하라.

from hybrid_retriever import build_bm25_retriever, build_vector_retriever, build_ensemble_retriever

def measure_recall_retriever(retriever, query: str, relevant_sources: List[str]) -> float:
    """
    LangChain 리트리버의 Recall을 측정한다.
    (vectorstore가 아닌 retriever 인터페이스 사용)
    """
    results = retriever.invoke(query)
    retrieved_sources = {doc.metadata.get("source") for doc in results}
    found = len(set(relevant_sources) & retrieved_sources)
    total = len(set(relevant_sources))
    return found / total if total > 0 else 0.0

# TODO: 세 리트리버의 평균 Recall@5를 비교하라
# - 벡터 단독
# - BM25 단독
# - 하이브리드 (벡터 + BM25)

# 실습 5-2: 최적 청크 크기 찾기
# chunk_size를 [100, 200, 300, 400, 500, 700, 1000]으로 변경하면서
# Recall@5를 측정하고 최적값을 찾아라.
# 그래프는 없어도 되고, 텍스트 출력으로 시각화하라.

def text_bar_chart(values: List[float], labels: List[str], width: int = 40):
    """
    텍스트로 막대 그래프를 출력한다.
    예시:
      100자 |████████████       0.62
      300자 |████████████████   0.78
      500자 |████████████████   0.75
    """
    max_val = max(values) if values else 1.0
    for label, val in zip(labels, values):
        bar_len = int(val / max_val * width)
        bar = "█" * bar_len
        print(f"  {label:10} |{bar:<{width}} {val:.3f}")
```

---

## 9. 전체 파이프라인 통합

### 9.1 Level 2 완성 RAG 파이프라인

```python
# 파일: rag_enhanced.py
# 목적: Level 2 향상된 RAG 파이프라인
# 기능: 하이브리드 검색 + LLM 재순위 + 개선된 프롬프트
# 실행: python rag_enhanced.py [--rebuild] [--strategy STRATEGY]

import os
import sys
import argparse
from typing import List
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import Document

from ollama_llm import create_llm, create_embeddings
from loaders import load_directory
from chunking_strategies import (
    strategy_fixed_size,
    strategy_markdown_aware,
)
from hybrid_retriever import build_bm25_retriever, build_ensemble_retriever
from reranker import llm_rerank

load_dotenv()

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_enhanced"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
TOP_K_INITIAL = 15   # 1차 검색 후보 수
TOP_K_FINAL = 3      # 최종 LLM에 전달할 문서 수


# ─────────────────────────────────────────────
# 프롬프트
# ─────────────────────────────────────────────

ENHANCED_PROMPT = """당신은 주어진 컨텍스트 문서만 사용하는 정밀 분석가입니다.

[컨텍스트 문서]
{context}

[질문]
{question}

[규칙]
1. 컨텍스트에 있는 내용만 사용하세요.
2. 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
3. 답변에 사용한 정보의 출처(파일명)를 (출처: 파일명) 형식으로 명시하세요.
4. 답변은 3문장 이내로 간결하게 작성하세요.
5. 한국어로 답변하세요.

[답변]"""


# ─────────────────────────────────────────────
# 향상된 RAG 클래스
# ─────────────────────────────────────────────

class EnhancedRAG:
    """
    Level 2 향상된 RAG 파이프라인.

    개선 사항:
    1. 하이브리드 검색 (벡터 + BM25)
    2. LLM 재순위 (Top-K 압축)
    3. 최적화된 청킹 (마크다운 구조 또는 고정 크기)
    """

    def __init__(self, strategy: str = "fixed"):
        self.llm = create_llm()
        self.embeddings = create_embeddings()
        self.chunks: List[Document] = []
        self.vectorstore: Chroma = None
        self.ensemble_retriever = None
        self.strategy = strategy

    def build(self, data_dir: str = DATA_DIR, force_rebuild: bool = False):
        """벡터 DB 구축 또는 재로딩"""
        docs = load_directory(data_dir)
        if not docs:
            raise FileNotFoundError(f"{data_dir}에 지원 파일이 없습니다.")

        # 청킹 전략 선택
        if self.strategy == "markdown":
            self.chunks = strategy_markdown_aware(docs)
            print(f"[청킹] 마크다운 구조 기반: {len(self.chunks)}개 청크")
        else:
            self.chunks = strategy_fixed_size(docs, chunk_size=CHUNK_SIZE)
            print(f"[청킹] 고정 크기({CHUNK_SIZE}자): {len(self.chunks)}개 청크")

        # 벡터 DB
        sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
        if os.path.exists(sqlite_path) and not force_rebuild:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings,
            )
            print(f"[Chroma] 재로딩 ({self.vectorstore._collection.count()}개 청크)")
        else:
            self.vectorstore = Chroma.from_documents(
                documents=self.chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_DIR,
            )
            print(f"[Chroma] 구축 완료 ({len(self.chunks)}개 청크)")

        # 앙상블 리트리버 (하이브리드)
        # BM25는 메모리 기반이므로 self.chunks 필요
        bm25 = build_bm25_retriever(self.chunks, k=TOP_K_INITIAL)
        vector_ret = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_INITIAL}
        )
        from langchain.retrievers import EnsembleRetriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_ret, bm25],
            weights=[0.6, 0.4],
        )
        print("[하이브리드] 앙상블 리트리버 구성 완료")

    def query(self, question: str) -> dict:
        """
        질문에 답하고 결과를 반환한다.

        1. 하이브리드 검색으로 후보 15개 검색
        2. LLM 재순위로 상위 3개 선택
        3. 선택된 문서로 LLM 답변 생성
        """
        if not self.ensemble_retriever:
            raise RuntimeError("build()를 먼저 호출하세요.")

        # 1. 하이브리드 검색
        candidates = self.ensemble_retriever.invoke(question)[:TOP_K_INITIAL]

        # 2. LLM 재순위
        if len(candidates) > TOP_K_FINAL:
            final_docs = llm_rerank(question, candidates, self.llm, top_k=TOP_K_FINAL)
        else:
            final_docs = candidates[:TOP_K_FINAL]

        # 3. 답변 생성
        context = "\n\n".join([
            f"[출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in final_docs
        ])

        from langchain_core.messages import HumanMessage, SystemMessage
        system_msg = ENHANCED_PROMPT.replace("{context}", context).replace("{question}", question)

        response = self.llm.invoke([HumanMessage(content=system_msg)])

        return {
            "question": question,
            "answer": response.content,
            "source_documents": final_docs,
            "num_candidates": len(candidates),
        }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Level 2 향상된 RAG Q&A")
    parser.add_argument("--rebuild", action="store_true", help="DB 재구축")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "markdown"],
        default="fixed",
        help="청킹 전략 (기본: fixed)"
    )
    args = parser.parse_args()

    print("=== Level 2 향상된 RAG Q&A ===")
    print(f"청킹 전략: {args.strategy}")

    rag = EnhancedRAG(strategy=args.strategy)
    try:
        rag.build(force_rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    print("\n대화 시작 (종료: 'q')\n")

    while True:
        try:
            question = input("질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        result = rag.query(question)

        print(f"\n{'─'*60}")
        print(f"답변:\n{result['answer']}")
        print(f"\n근거 문서 ({len(result['source_documents'])}개, 후보 {result['num_candidates']}개에서 선택):")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:80]}...")
        print("─"*60)


if __name__ == "__main__":
    main()
```

---

## 10. 테스트 작성

```python
# 파일: test_hybrid.py
# 실행: pytest test_hybrid.py -v

import pytest
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from chunking_strategies import strategy_fixed_size, SAMPLE_DOCUMENTS
from hybrid_retriever import build_bm25_retriever
from rrf_demo import rrf_fusion, ScoredDocument


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    """샘플 문서를 고정 크기로 청킹"""
    return strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=200)


@pytest.fixture
def bm25_retriever(sample_chunks):
    """BM25 리트리버 인스턴스"""
    return build_bm25_retriever(sample_chunks, k=3)


# ─────────────────────────────────────────────
# BM25 테스트
# ─────────────────────────────────────────────

class TestBM25:
    def test_returns_documents(self, bm25_retriever):
        """BM25 검색이 Document 목록을 반환한다"""
        results = bm25_retriever.invoke("503 오류")
        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    def test_returns_k_or_fewer(self, bm25_retriever):
        """k개 이하의 결과를 반환한다"""
        results = bm25_retriever.invoke("임베딩")
        assert len(results) <= 3

    def test_keyword_match(self, bm25_retriever):
        """키워드가 포함된 문서를 우선 반환한다"""
        results = bm25_retriever.invoke("503 오류")
        # 상위 결과 중 하나에 "503"이 포함되어야 함
        all_content = " ".join([d.page_content for d in results])
        assert "503" in all_content

    def test_empty_query(self, bm25_retriever):
        """빈 쿼리는 빈 결과 또는 무작위 결과를 반환한다"""
        results = bm25_retriever.invoke("")
        # 예외가 발생하지 않아야 함
        assert isinstance(results, list)


# ─────────────────────────────────────────────
# RRF 테스트
# ─────────────────────────────────────────────

class TestRRF:
    def test_fusion_combines_results(self):
        """RRF가 두 결과 리스트를 통합한다"""
        list1 = [ScoredDocument("A"), ScoredDocument("B")]
        list2 = [ScoredDocument("C"), ScoredDocument("A")]  # A 중복

        fused = rrf_fusion([list1, list2], k=60)

        # 중복 제거 후 3개
        assert len(fused) == 3

    def test_higher_rank_gets_higher_score(self):
        """높은 순위(낮은 인덱스)의 문서가 더 높은 RRF 점수를 가진다"""
        list1 = [ScoredDocument("A"), ScoredDocument("B"), ScoredDocument("C")]

        fused = rrf_fusion([list1], k=60)

        # 1위가 가장 높은 점수
        assert fused[0].content == "A"
        assert fused[0].score > fused[1].score

    def test_documents_in_multiple_lists_rank_higher(self):
        """여러 리스트에서 나타나는 문서가 더 높은 순위를 가진다"""
        list1 = [ScoredDocument("A"), ScoredDocument("B")]
        list2 = [ScoredDocument("C"), ScoredDocument("A")]  # A가 두 번 나옴

        fused = rrf_fusion([list1, list2], k=60)
        fused_contents = [d.content for d in fused]

        # A가 B와 C보다 높은 순위여야 함
        assert fused_contents.index("A") < fused_contents.index("B")
        assert fused_contents.index("A") < fused_contents.index("C")

    def test_rrf_score_formula(self):
        """RRF 점수가 수식과 일치하는지 확인"""
        doc = ScoredDocument("A")
        list1 = [doc]

        fused = rrf_fusion([list1], k=60)

        # rank=1, k=60 이므로 점수 = 1/(60+1) = 0.016393...
        expected = 1.0 / (60 + 1)
        assert abs(fused[0].score - expected) < 1e-6


# ─────────────────────────────────────────────
# 청킹 전략 테스트
# ─────────────────────────────────────────────

class TestChunkingStrategies:
    def test_fixed_size_chunk_size(self):
        """고정 크기 청킹의 청크 크기가 지정값을 넘지 않는다"""
        chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=200)
        for chunk in chunks:
            # 약간의 여유 허용 (splitter 내부 처리)
            assert len(chunk.page_content) <= 220

    def test_markdown_chunking_preserves_structure(self):
        """마크다운 청킹이 헤더 정보를 메타데이터에 보존한다"""
        from chunking_strategies import strategy_markdown_aware
        chunks = strategy_markdown_aware(SAMPLE_DOCUMENTS)

        # 일부 청크에 헤더 메타데이터가 있어야 함
        has_header = any(
            "header1" in c.metadata or "header2" in c.metadata
            for c in chunks
        )
        assert has_header

    def test_chunking_preserves_source_metadata(self):
        """청킹 후에도 source 메타데이터가 보존된다"""
        chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
        for chunk in chunks:
            assert "source" in chunk.metadata
```

### 10.1 requirements.txt (Level 2 추가분)

```
# requirements.txt (Level 1 + Level 2 추가)
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
langchain-experimental>=0.0.60
chromadb>=0.5.0
rank-bm25>=0.2.2
FlagEmbedding>=1.2.0
huggingface_hub>=0.20.0
openpyxl>=3.1.0
pillow>=10.0.0
pytesseract>=0.3.10
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
pytest>=8.0.0
```

---

## 11. 체크리스트 및 다음 단계

### 11.1 Level 2 완성 체크리스트

**개념 이해**

- [ ] 청크 크기와 Recall/Precision 트레이드오프를 설명할 수 있다
- [ ] BM25 수식(TF 포화, 문서 길이 정규화)을 설명할 수 있다
- [ ] BM25와 벡터 검색의 차이 및 각각의 강점을 설명할 수 있다
- [ ] RRF(Reciprocal Rank Fusion) 수식을 직접 계산할 수 있다
- [ ] Reranking이 필요한 이유를 설명할 수 있다
- [ ] Parent-Child 청킹의 장점을 설명할 수 있다

**구현 능력**

- [ ] `BM25Retriever.from_documents`로 BM25 리트리버를 구성할 수 있다
- [ ] `EnsembleRetriever`로 하이브리드 검색을 구성할 수 있다
- [ ] RRF를 직접 구현할 수 있다
- [ ] LLM 기반 재순위 함수를 구현할 수 있다
- [ ] `SemanticChunker`, `ParentDocumentRetriever`를 사용할 수 있다

**실험 결과**

- [ ] 3가지 청킹 전략의 Recall@5를 비교 완료
- [ ] 벡터 단독 vs 하이브리드 Recall 차이 측정 완료
- [ ] Reranking 전후 품질 차이를 정성적으로 확인 완료

### 11.2 Level 3으로 진행하기 전 확인사항

Level 2를 완성했다면, 다음 질문들을 생각해보자:

1. "검색 결과가 관련 없을 때도 LLM이 답변을 만들어낸다" → Self-RAG
2. "짧은 질문은 임베딩이 잘 안 된다" → HyDE
3. "다단계 추론이 필요한 질문을 처리할 수 없다" → Multi-hop RAG
4. "RAG 품질을 자동으로 측정하고 싶다" → RAGAS 평가

이 모든 것이 Level 3의 주제다.
