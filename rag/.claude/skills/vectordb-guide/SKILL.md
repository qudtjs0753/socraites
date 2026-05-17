---
name: vectordb-guide
description: "Chroma DB, Elasticsearch 등 벡터/렉시컬 검색 엔진 가이드 스킬. Chroma 설정, Elasticsearch 인덱스 설계, BM25, 하이브리드 검색, RRF, 리랭킹, Nori 형태소 분석기, 임베딩 저장, 메타데이터 필터링, 청킹 전략, 벡터 인덱스 최적화 요청 시 이 스킬을 사용하라. '어떤 벡터DB 써야 해', 'Chroma vs Elasticsearch', '하이브리드 검색 설정', '한국어 ES 설정' 등 모든 검색 엔진 관련 요청에 이 스킬을 사용."
---

# VectorDB & 검색 엔진 가이드 스킬

Chroma DB, Elasticsearch를 RAG 시스템의 지식 베이스로 활용하는 DA 관점 가이드.

## 검색 엔진 선택 가이드

| 기준 | Chroma | Elasticsearch |
|------|--------|--------------|
| 설정 복잡도 | 매우 낮음 (pip install) | 중간 (Docker 또는 K8S) |
| 벡터 검색 | 기본 지원 | kNN 플러그인 |
| 렉시컬 검색 | 미지원 | BM25 기본 지원 |
| 한국어 처리 | 임베딩 의존 | Nori 형태소 분석기 |
| 프로덕션 스케일 | 제한적 | 뛰어남 (샤딩, 복제) |
| K8S 배포 | StatefulSet | ECK (공식 오퍼레이터) |

**추천:**
- 빠른 프로토타입 → Chroma
- 한국어 렉시컬 검색 필요 → Elasticsearch
- 프로덕션 하이브리드 → Elasticsearch (kNN + BM25)

## Chroma DB 심화 설정

### 컬렉션 설계

```python
import chromadb
from chromadb.config import Settings

# 영속적 Chroma 클라이언트
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# 컬렉션 생성 — 메타데이터 스키마 설계가 핵심
collection = client.get_or_create_collection(
    name="rag_documents",
    metadata={
        "hnsw:space": "cosine",    # 코사인 유사도 (기본값은 l2)
        "hnsw:construction_ef": 200,  # 인덱스 구축 품질 (높을수록 정확도↑, 속도↓)
        "hnsw:M": 16,             # 연결 수 (기본: 16)
    }
)

# 문서 추가 — 메타데이터를 풍부하게 설계
collection.add(
    documents=["RAG는 외부 지식을 활용하는 LLM 기법입니다."],
    embeddings=[[0.1, 0.2, ...]],   # 또는 embedding_function 사용
    metadatas=[{
        "source": "rag_intro.pdf",
        "page": 1,
        "category": "concept",
        "language": "ko",
        "created_at": "2024-01-01"
    }],
    ids=["doc_001"]
)
```

### 메타데이터 필터링 (where 절)

```python
# 특정 카테고리 + 최근 문서만 검색
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={
        "$and": [
            {"category": {"$eq": "concept"}},
            {"language": {"$eq": "ko"}}
        ]
    },
    include=["documents", "metadatas", "distances"]
)
```

## Elasticsearch 한국어 설정

### Nori 분석기 인덱스 생성

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# 한국어 특화 인덱스 매핑
mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "korean_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": [
                        "nori_readingform",   # 한자 → 한글 변환
                        "lowercase",
                        "nori_part_of_speech"  # 불용어 제거
                    ]
                }
            },
            "tokenizer": {
                "nori_tokenizer": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed"  # 복합어 분리
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "korean_analyzer"
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,          # 사내 임베딩 모델 차원 — EMBED_MODEL에 맞게 조정
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "type": "object",
                "dynamic": True
            }
        }
    }
}

es.indices.create(index="rag-documents", body=mapping)
```

### 하이브리드 검색 (BM25 + kNN)

```python
# Elasticsearch 하이브리드 검색 쿼리
def hybrid_search(query_text: str, query_vector: list, top_k: int = 5):
    response = es.search(
        index="rag-documents",
        body={
            "query": {
                "bool": {
                    "should": [
                        # BM25 렉시컬 검색
                        {
                            "match": {
                                "content": {
                                    "query": query_text,
                                    "boost": 0.4  # BM25 가중치
                                }
                            }
                        }
                    ]
                }
            },
            # kNN 벡터 검색
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100,
                "boost": 0.6           # 벡터 가중치
            },
            "size": top_k
        }
    )
    return response["hits"]["hits"]
```

## 청킹 전략 비교

| 전략 | 방법 | 적합한 문서 | 주의사항 |
|------|------|-----------|---------|
| 고정 크기 | 500토큰, 50 overlap | 균일한 텍스트 | 문장 끊김 가능 |
| 재귀적 | 단락→문장→단어 순서 분리 | 일반 문서 | LangChain 기본 |
| 의미 단위 | 문단/섹션 기반 | 구조화된 문서 | 크기 불균일 |
| 부모-자식 | 큰 청크 인덱스, 작은 청크 검색 | 컨텍스트 중요 | 복잡한 구현 |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 한국어 문서 최적 설정 (실험값)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       # 한국어는 영어보다 정보 밀도가 높아 작게
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", ".", " ", ""]  # 한국어 문장 구분 추가
)
```

## Reranking (검색 결과 재순위)

```python
# pip install cohere
import cohere

co = cohere.Client("COHERE_API_KEY")

def rerank_results(query: str, docs: list, top_n: int = 3):
    response = co.rerank(
        query=query,
        documents=[d.page_content for d in docs],
        top_n=top_n,
        model="rerank-multilingual-v3.0"  # 한국어 지원
    )
    return [docs[r.index] for r in response.results]
```

## 검색 품질 측정

```python
# Recall@K 측정 예시
def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_top_k & relevant) / len(relevant)

# 사용
recall = recall_at_k(retrieved_ids, ground_truth_ids, k=5)
print(f"Recall@5: {recall:.3f}")  # 목표: 0.85 이상
```

## 관련 references

- Level별 상세 내용: `../rag-study-orchestrator/references/` 참조
