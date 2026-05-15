# Level 2: 향상된 RAG (3주)

## 목차
- [학습 목표](#학습-목표)
- [3주차: 청킹 전략 최적화](#3주차-청킹-전략-최적화)
- [4주차: Elasticsearch + BM25 + 한국어 처리](#4주차-elasticsearch--bm25--한국어-처리)
- [5주차: 하이브리드 검색 + Reranking](#5주차-하이브리드-검색--reranking)
- [핵심 실습 프로젝트](#핵심-실습-프로젝트)
- [체크리스트](#체크리스트)

---

## 학습 목표

3주 후 달성 목표:
- 청킹 전략이 검색 품질에 미치는 영향 실험으로 확인
- Elasticsearch에 한국어 RAG 인덱스 구축 (Nori 분석기)
- Chroma + BM25 하이브리드 검색으로 단순 벡터 검색 대비 Recall 개선

---

## 3주차: 청킹 전략 최적화

### 청킹 전략이 왜 중요한가?

검색 품질의 60%는 청킹 전략에서 결정된다. 청크가 너무 작으면 컨텍스트 부족, 너무 크면 노이즈 포함.

### 4가지 청킹 전략 비교

**1. 고정 크기 청킹 (Fixed Size)**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 가장 단순한 방식. 대부분의 경우 시작점
fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
```

**2. 의미 단위 청킹 (Semantic Chunking)**
```python
# pip install langchain-experimental
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# 임베딩 유사도 기반으로 의미가 끊기는 지점에서 분리
semantic_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # 상위 95% 변화점에서 분리
    breakpoint_threshold_amount=95
)

# 주의: 임베딩 비용 발생 (청크당 API 호출)
semantic_chunks = semantic_splitter.split_text(long_text)
```

**3. 부모-자식 청킹 (Parent-Child Chunking)**
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 부모(큰 청크): 컨텍스트 제공용
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# 자식(작은 청크): 검색용 (더 정확한 검색)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

store = InMemoryStore()  # 부모 청크 저장소

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# 검색: 작은 청크로 찾고, 큰 청크를 반환
retriever.add_documents(docs)
results = retriever.invoke("RAG란?")  # 작은 청크 기준 검색, 부모 청크 반환
```

**4. 재귀적 청킹 (기본 권장)**
```python
# LangChain 기본 전략 — 단락 > 문장 > 단어 순서로 분리 시도
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",  # 단락
        "\n",    # 줄바꿈
        "。",    # 한국어/일본어 마침표
        ".",     # 영어 마침표
        ",",
        " ",
        ""
    ]
)
```

### 청킹 전략 실험 방법

```python
# 동일 질문으로 각 청킹 전략의 Recall 비교
def evaluate_chunking_strategy(splitter, docs, test_queries, ground_truth_docs):
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(chunks, OpenAIEmbeddings())
    
    recalls = []
    for query, gt_ids in zip(test_queries, ground_truth_docs):
        retrieved = vs.similarity_search(query, k=5)
        retrieved_ids = {d.metadata.get("source") for d in retrieved}
        recall = len(retrieved_ids & set(gt_ids)) / len(gt_ids)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls)

# 결과 비교표 출력
strategies = {
    "fixed_500": fixed_splitter,
    "fixed_200": RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
    "semantic": semantic_splitter,
}
for name, splitter in strategies.items():
    recall = evaluate_chunking_strategy(splitter, docs, queries, gt)
    print(f"{name}: Recall@5 = {recall:.3f}")
```

---

## 4주차: Elasticsearch + BM25 + 한국어 처리

### Docker로 Elasticsearch 실행

```bash
# Elasticsearch 8.x with Korean (Nori) plugin
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# Nori 플러그인 설치 (컨테이너 내에서)
docker exec -it elasticsearch \
  bin/elasticsearch-plugin install analysis-nori
docker restart elasticsearch
```

### 한국어 인덱스 설계

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# 한국어 특화 인덱스 생성
index_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "korean": {
                    "type": "nori",
                    "decompound_mode": "mixed"   # 복합어: '서울역' → '서울', '역' 분리
                },
                "korean_search": {
                    "type": "nori",
                    "decompound_mode": "mixed",
                    "stoptags": ["E", "IC", "J", "MAG", "MAJ", "MM", "SP", "SSC", "SSO", "SC", "SE", "XPN", "XSA", "XSN", "XSV", "UNA", "NA", "VSV"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "korean",         # 인덱싱 시 분석기
                "search_analyzer": "korean_search"  # 검색 시 분석기
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
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

es.indices.create(index="rag-ko", body=index_settings, ignore=400)
```

### LangChain + Elasticsearch 통합

```python
# pip install langchain-elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

# Elasticsearch 벡터스토어 (LangChain 통합)
es_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="rag-ko",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    # BM25 + 벡터 하이브리드 내장 지원
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(
        hybrid=True  # kNN + BM25 자동 혼합
    )
)

# 문서 추가
es_store.add_documents(chunks)

# 검색
results = es_store.similarity_search("RAG의 장점", k=5)
```

### BM25 직접 사용 (LangChain BM25Retriever)

```python
# pip install rank-bm25
from langchain_community.retrievers import BM25Retriever

# 텍스트 청크에서 직접 BM25 인덱스 생성 (별도 서버 불필요)
bm25_retriever = BM25Retriever.from_documents(
    chunks,
    k=5,
    preprocess_func=lambda text: text.lower()  # 전처리
)

results = bm25_retriever.invoke("검색 증강 생성")
```

**BM25 vs 벡터 검색 비교:**

| 상황 | BM25 유리 | 벡터 검색 유리 |
|------|---------|-------------|
| 키워드 정확 일치 | ✅ | ❌ (의미 분산) |
| 의미 유사 검색 | ❌ | ✅ |
| 오타/변형어 처리 | ❌ | ✅ |
| 고유명사, 코드 | ✅ | ❌ |
| 한국어 형태소 변형 | Nori 필요 | 임베딩으로 처리 |

---

## 5주차: 하이브리드 검색 + Reranking

### EnsembleRetriever (Chroma + BM25)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 두 리트리버 구성
vector_retriever = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
).as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 앙상블 — 가중치는 실험으로 결정
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 벡터 60%, BM25 40%
)

# RRF (Reciprocal Rank Fusion) 내부 적용됨
results = ensemble.invoke("한국어 RAG 구현 방법")
```

### Reranking — 검색 후 재순위

```python
# pip install cohere
import cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 1단계: 후보 검색 (많이)
base_retriever = ensemble  # 앙상블 리트리버 (20개 후보)

# 2단계: Reranker로 재순위 (상위 5개 선택)
reranker = CohereRerank(
    model="rerank-multilingual-v3.0",  # 한국어 지원
    top_n=5
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# 사용
final_results = compression_retriever.invoke("RAG 평가 방법")
```

**BGE Reranker (무료 오픈소스 대안):**
```python
# pip install FlagEmbedding
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)  # 한국어 지원

# 쌍별 점수 계산
pairs = [(query, doc.page_content) for doc in candidate_docs]
scores = reranker.compute_score(pairs)

# 점수 기준 재정렬
sorted_docs = [doc for _, doc in sorted(zip(scores, candidate_docs), reverse=True)]
top5 = sorted_docs[:5]
```

---

## 핵심 실습 프로젝트

**"한국어 하이브리드 검색 RAG" 완성**

요구사항:
1. 한국어 PDF/텍스트 문서 사용
2. Chroma(벡터) + BM25(렉시컬) 앙상블 리트리버
3. BGE Reranker로 최종 5개 선택
4. Level 1 기본 벡터 검색 대비 Recall@5 측정 비교

기대 결과:
- 기본 벡터 검색 Recall@5: ~0.70
- 하이브리드 + Reranking Recall@5: ~0.85

---

## 체크리스트

**개념 이해**
- [ ] 청킹 크기와 검색 품질의 trade-off 설명 가능
- [ ] BM25와 벡터 검색의 차이 및 각각의 강점 설명
- [ ] RRF (Reciprocal Rank Fusion) 동작 원리 이해
- [ ] Reranking이 필요한 이유 설명 가능

**구현 능력**
- [ ] Docker로 Elasticsearch + Nori 설정 가능
- [ ] LangChain EnsembleRetriever 구성 가능
- [ ] Reranker 적용 및 결과 비교 가능

**실험 결과**
- [ ] 3가지 청킹 전략 Recall@5 비교 완료
- [ ] 벡터 vs BM25 vs 하이브리드 성능 비교 완료
- [ ] Reranking 적용 전후 품질 차이 측정 완료
