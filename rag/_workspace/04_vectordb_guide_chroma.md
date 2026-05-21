# Chroma DB 벡터 검색 완전 가이드

> 대상: Python 개발자 / DA (Data Architecture 관점)
> 환경: 완전 오프라인 (Ollama 로컬 임베딩, BGE 오프라인 리랭킹)
> 구성: 이론 → Chroma 실전 → 하이브리드 검색 → Reranking → 실습 과제

---

## 목차

1. [벡터 임베딩 이론](#1-벡터-임베딩-이론)
2. [유사도 측정 방법](#2-유사도-측정-방법)
3. [HNSW 인덱싱 알고리즘](#3-hnsw-인덱싱-알고리즘)
4. [BM25 렉시컬 검색 이론](#4-bm25-렉시컬-검색-이론)
5. [RRF (Reciprocal Rank Fusion)](#5-rrf-reciprocal-rank-fusion)
6. [환경 설정](#6-환경-설정)
7. [Ollama 임베딩 클래스](#7-ollama-임베딩-클래스)
8. [Chroma 실전 사용](#8-chroma-실전-사용)
9. [컬렉션 설계 전략](#9-컬렉션-설계-전략)
10. [similarity_search vs MMR](#10-similarity_search-vs-mmr)
11. [메타데이터 필터링](#11-메타데이터-필터링)
12. [하이브리드 검색 (Chroma + BM25)](#12-하이브리드-검색-chroma--bm25)
13. [BGE Reranker 오프라인 적용](#13-bge-reranker-오프라인-적용)
14. [전체 파이프라인 통합](#14-전체-파이프라인-통합)
15. [검색 품질 평가](#15-검색-품질-평가)
16. [실습 과제](#16-실습-과제)

---

## 1. 벡터 임베딩 이론

### 1.1 임베딩이란?

텍스트를 고차원 수치 벡터로 변환하는 과정이다. 의미가 비슷한 텍스트는 벡터 공간에서 가까운 위치에 배치된다.

```
"고양이는 귀엽다" → [0.23, -0.45, 0.87, ..., 0.12]  (768차원 또는 1024차원)
"강아지는 사랑스럽다" → [0.21, -0.43, 0.85, ..., 0.09]  (비슷한 벡터)
"서울의 날씨" → [-0.78, 0.34, -0.12, ..., 0.67]  (다른 위치)
```

수식으로 표현하면:

```
f: 텍스트 T → ℝᵈ
f("고양이는 귀엽다") = v₁ ∈ ℝ⁷⁶⁸
```

여기서 d는 임베딩 차원(dimension)이다.
- nomic-embed-text: 768차원
- text-embedding-ada-002: 1536차원
- BGE-m3: 1024차원

### 1.2 임베딩 모델이 학습하는 것

Transformer 기반 임베딩 모델은 대규모 텍스트 코퍼스에서 다음을 학습한다:

**Contrastive Learning (대조 학습)**
```
목표: sim(f(q), f(d+)) > sim(f(q), f(d-)) + margin

q   = 쿼리
d+  = 관련 문서 (positive)
d-  = 무관 문서 (negative)
margin = 최소 차이값 (보통 0.3~0.5)
```

이 과정에서 모델은 의미적으로 유사한 텍스트를 벡터 공간의 가까운 위치에 매핑하도록 가중치를 업데이트한다.

### 1.3 차원의 의미

768차원 벡터의 각 차원은 특정 의미적 특징을 인코딩한다 (해석 불가능한 형태로):

```python
# 예시: 단순화된 4차원 의미 공간 (실제는 768차원)
"왕"   = [0.9, 0.1, 0.8, 0.2]  # [royalty, female, power, age]
"여왕" = [0.9, 0.9, 0.8, 0.2]  # royalty↑, female↑
"왕자" = [0.9, 0.1, 0.8, 0.8]  # royalty↑, age↑(young)

# Word2Vec의 유명한 예시:
# 왕 - 남자 + 여자 ≈ 여왕
# vector("왕") - vector("남자") + vector("여자") ≈ vector("여왕")
```

### 1.4 Dense vs Sparse 벡터

| 유형 | 설명 | 예시 | 차원 | 특징 |
|------|------|------|------|------|
| Dense | 대부분 0이 아닌 값 | Transformer 임베딩 | 768~1536 | 의미 유사도 캡처 |
| Sparse | 대부분 0, 일부만 값 | TF-IDF, BM25 | 어휘 크기(수만) | 정확 키워드 매칭 |

Chroma는 Dense 벡터를 다루고, BM25는 Sparse 벡터를 다룬다. 두 방식의 장점을 합친 것이 하이브리드 검색이다.

---

## 2. 유사도 측정 방법

### 2.1 코사인 유사도 (Cosine Similarity)

두 벡터 사이의 각도를 측정한다. 크기(magnitude)와 무관하게 방향만 비교한다.

```
cos(θ) = (A · B) / (||A|| × ||B||)

A · B = Σ(aᵢ × bᵢ)     (내적, dot product)
||A|| = √(Σaᵢ²)          (벡터 크기, L2 norm)

범위: [-1, 1]
  1.0  → 완전 동일한 방향 (매우 유사)
  0.0  → 직교 (무관)
 -1.0  → 반대 방향 (반의어)
```

Python 구현:
```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    코사인 유사도 계산
    두 벡터의 방향이 얼마나 일치하는지 측정
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

# 예시
v1 = np.array([0.23, -0.45, 0.87, 0.12])
v2 = np.array([0.21, -0.43, 0.85, 0.09])
v3 = np.array([-0.78, 0.34, -0.12, 0.67])

print(f"v1-v2 유사도: {cosine_similarity(v1, v2):.4f}")  # 높음 (유사한 의미)
print(f"v1-v3 유사도: {cosine_similarity(v1, v3):.4f}")  # 낮음 (다른 의미)
```

**Chroma 기본값이 코사인 유사도인 이유:**
- 텍스트 임베딩은 길이가 다른 문서를 비교할 때 코사인이 적합
- 짧은 문장과 긴 문단을 동등하게 비교 가능 (크기 정규화 효과)
- 임베딩 모델 학습 시 코사인 기반으로 최적화된 경우가 많음

### 2.2 유클리드 거리 (L2 Distance)

두 벡터 사이의 직선 거리를 측정한다.

```
d(A, B) = √(Σ(aᵢ - bᵢ)²)

범위: [0, ∞)
  0.0  → 완전 동일 (동일한 벡터)
  작을수록 → 유사
  클수록   → 비유사
```

```python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    유클리드 거리 계산
    실제 공간에서의 거리 개념
    """
    return np.linalg.norm(a - b)

# Chroma에서 L2 사용
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="l2_collection",
    metadata={"hnsw:space": "l2"}  # 유클리드 거리 사용
)
```

**코사인 vs 유클리드 선택 기준:**

| 상황 | 추천 | 이유 |
|------|------|------|
| 텍스트 임베딩 (일반) | 코사인 | 벡터 크기 무관하게 의미 비교 |
| 정규화된 벡터 | 동일 | L2 정규화된 경우 코사인 = 내적 |
| 이미지 임베딩 | L2 | 크기 정보가 의미를 가질 때 |
| 추천 시스템 | 내적 | 인기도(magnitude)를 고려할 때 |

### 2.3 내적 (Inner Product / Dot Product)

```
A · B = Σ(aᵢ × bᵢ)

범위: (-∞, +∞)
크기와 방향 모두 고려
```

```python
def inner_product(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)

# Chroma에서 IP 사용
collection = client.get_or_create_collection(
    name="ip_collection",
    metadata={"hnsw:space": "ip"}  # 내적 사용
)
```

**중요 관계:**
```
벡터가 L2 정규화된 경우 (||A|| = ||B|| = 1):
  코사인 유사도 = 내적
  
L2 정규화 방법:
  A_normalized = A / ||A||
```

### 2.4 Chroma 거리 설정 요약

```python
# Chroma 지원 거리 메트릭
DISTANCE_METRICS = {
    "cosine": "코사인 거리 (1 - 코사인유사도) — 텍스트 기본값",
    "l2": "유클리드 거리 — 이미지, 수치 데이터",
    "ip": "내적 (음수로 변환) — 추천 시스템",
}

# 주의: Chroma는 "유사도"가 아닌 "거리"를 반환
# 코사인 거리 = 1 - 코사인유사도
# 따라서 0에 가까울수록 더 유사
```

---

## 3. HNSW 인덱싱 알고리즘

### 3.1 왜 인덱싱이 필요한가?

1백만 개의 문서가 있을 때 쿼리 벡터와 모든 문서를 비교(brute-force)하면:
```
1,000,000 벡터 × 768차원 × 곱셈/덧셈 → 수십 초 소요
```

HNSW는 이 문제를 근사 최근접 이웃(ANN, Approximate Nearest Neighbor) 알고리즘으로 해결한다.

### 3.2 NSW (Navigable Small World) — HNSW의 기반

그래프 기반 검색 아이디어:
```
1. 각 노드(벡터)를 가장 가까운 K개의 노드와 연결
2. 검색 시: 임의의 진입점(entry point)에서 시작
3. 현재 노드의 이웃 중 쿼리에 더 가까운 노드로 이동
4. 더 가까운 이웃이 없으면 중단
```

문제점: NSW는 평탄한(flat) 그래프라 지역 최솟값(local minimum)에 빠질 수 있다.

### 3.3 HNSW (Hierarchical NSW)

계층 구조를 추가해 NSW의 문제를 해결한다.

```
레이어 구조 (위로 올라갈수록 희소):

Layer 2 (최상위):  A ─────────────── B
                   (긴 거리 연결)
                   
Layer 1 (중간):    A ──── C ──── D ── B
                   (중거리 연결)

Layer 0 (최하위):  A ─ C ─ E ─ F ─ G ─ D ─ H ─ B
                   (짧은 거리 연결, 모든 노드 포함)
```

**검색 과정:**
```
1. 최상위 레이어에서 시작 (빠른 이동)
2. 현재 레이어에서 그리디 탐색 (쿼리에 가까운 방향)
3. 더 이동 불가능하면 아래 레이어로 내려감
4. Layer 0에서 최종 후보 집합 결정
```

**삽입 과정:**
```
1. 새 노드의 최대 레이어: l = floor(-ln(uniform(0,1)) × mL)
   mL = 정규화 상수 (보통 1/ln(M))
2. 최상위 레이어부터 l까지는 그리디 탐색으로 진입점 찾기
3. l부터 Layer 0까지 각 레이어에 노드 삽입 및 M개 이웃 연결
```

### 3.4 HNSW 핵심 파라미터

```python
# Chroma에서 HNSW 파라미터 설정
collection = client.get_or_create_collection(
    name="my_collection",
    metadata={
        "hnsw:space": "cosine",
        
        # M: 각 노드의 최대 연결 수
        # 높을수록: 정확도↑, 메모리↑, 인덱싱 속도↓
        # 권장: 16~64 (기본값: 16)
        "hnsw:M": 16,
        
        # ef_construction: 인덱스 구축 시 탐색 후보 수
        # 높을수록: 정확도↑, 인덱싱 속도↓
        # 권장: 100~200 (기본값: 100)
        "hnsw:construction_ef": 100,
        
        # ef: 검색 시 탐색 후보 수
        # 높을수록: 정확도↑, 검색 속도↓
        # 권장: k의 10~100배 (기본값: 10)
        "hnsw:search_ef": 100,
    }
)
```

**파라미터 트레이드오프:**

| 파라미터 | 올리면 | 낮추면 |
|---------|-------|-------|
| M | 정확도↑, 메모리↑, 구축↓ | 정확도↓, 메모리↓, 구축↑ |
| ef_construction | 정확도↑, 구축↓ | 정확도↓, 구축↑ |
| ef (search) | 정확도↑, 검색↓ | 정확도↓, 검색↑ |

### 3.5 HNSW 성능 특성

```
시간 복잡도:
  삽입: O(log n)
  검색: O(log n)  (brute-force: O(n))
  
공간 복잡도:
  O(n × M)  (brute-force: O(n × d))

실제 성능 (1M 벡터, 768차원 기준):
  Brute-force: ~수십 초
  HNSW:        ~수 밀리초
  정확도:       ~95~99% recall (M=16, ef=100 기준)
```

---

## 4. BM25 렉시컬 검색 이론

### 4.1 TF-IDF에서 BM25로

BM25는 TF-IDF의 개선 버전이다.

**TF-IDF의 문제점:**
```
TF(term, doc) = 단어 빈도 / 문서 총 단어 수

문제: 단어가 10번 나오면 1번보다 10배 중요?
     → 실제로는 포화(saturation) 현상이 있음
```

### 4.2 BM25 수식

```
BM25(q, d) = Σᵢ IDF(qᵢ) × [TF(qᵢ, d) × (k₁ + 1)] / [TF(qᵢ, d) + k₁ × (1 - b + b × |d|/avgdl)]

변수 설명:
  q       = 쿼리 단어들
  d       = 문서
  qᵢ      = i번째 쿼리 단어
  |d|     = 문서 길이 (단어 수)
  avgdl   = 코퍼스 평균 문서 길이
  k₁      = TF 포화 파라미터 (보통 1.2~2.0, 기본 1.5)
  b       = 문서 길이 정규화 파라미터 (보통 0.75)
  
IDF(qᵢ) = log[(N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) + 1]
  N     = 총 문서 수
  n(qᵢ) = qᵢ를 포함하는 문서 수
```

### 4.3 BM25 수식 직관적 이해

**TF 부분 — 포화 현상:**
```python
import numpy as np

def tf_saturation(tf, k1=1.5):
    """
    BM25의 TF 포화 함수
    tf가 높아져도 점수는 무한정 늘어나지 않음
    """
    return (tf * (k1 + 1)) / (tf + k1)

# 확인
for tf in [0, 1, 2, 5, 10, 100]:
    score = tf_saturation(tf)
    print(f"TF={tf:3d} → 점수={score:.3f}")

# 출력:
# TF=  0 → 점수=0.000
# TF=  1 → 점수=1.000  (기준)
# TF=  2 → 점수=1.200  (2배가 아닌 1.2배)
# TF=  5 → 점수=1.400
# TF= 10 → 점수=1.455
# TF=100 → 점수=1.495  (최대값 k1+1=2.5에 수렴)
```

**문서 길이 정규화:**
```python
def bm25_tf(tf, doc_len, avg_doc_len, k1=1.5, b=0.75):
    """
    문서 길이를 고려한 BM25 TF
    긴 문서에서 단어가 많이 나오는 건 불리하게 조정
    """
    length_norm = 1 - b + b * (doc_len / avg_doc_len)
    return (tf * (k1 + 1)) / (tf + k1 * length_norm)

# b=0이면 길이 정규화 없음 (TF-IDF와 유사)
# b=1이면 완전 길이 정규화
```

### 4.4 BM25 vs 벡터 검색 비교

| 특성 | BM25 | 벡터 검색 |
|------|------|---------|
| 원리 | 키워드 빈도 통계 | 의미 유사도 |
| 동의어 처리 | 불가 ("자동차" ≠ "차량") | 가능 (유사 벡터) |
| 오타 처리 | 불가 | 부분적 가능 |
| 정확 매칭 | 강점 | 약점 (희석됨) |
| 고유명사/코드 | 강점 | 약점 |
| 다국어 | 형태소 분석 필요 | 다국어 모델로 처리 |
| 속도 | 빠름 | 느림 (ANN 사용 시 빠름) |
| 인프라 | 경량 | GPU/메모리 필요 |

---

## 5. RRF (Reciprocal Rank Fusion)

### 5.1 개념

여러 랭킹 리스트를 하나로 합치는 알고리즘이다. 각 결과의 순위(rank)만 사용하므로 점수 스케일이 달라도 문제없다.

```
RRF(d) = Σₗ 1 / (k + rankₗ(d))

d       = 문서
k       = 상수 (보통 60, 낮은 순위의 영향 완화)
rankₗ   = 리스트 l에서 문서 d의 순위 (1부터 시작)
l       = 랭킹 리스트 (벡터 검색, BM25 등)
```

### 5.2 RRF 수식 직관

```python
def rrf_score(rank: int, k: int = 60) -> float:
    """
    순위가 1인 문서: 1/(60+1) = 0.0164
    순위가 10인 문서: 1/(60+10) = 0.0143
    순위가 100인 문서: 1/(60+100) = 0.00625
    
    k=60의 의미: 상위 60위 밖은 영향이 매우 작아짐
    """
    return 1.0 / (k + rank)

# 예시: 두 리스트(벡터, BM25)에서 문서별 RRF 점수
docs_vector = ["문서A", "문서B", "문서C", "문서D"]  # 벡터 검색 결과 순위
docs_bm25   = ["문서C", "문서A", "문서E", "문서B"]  # BM25 결과 순위

k = 60
rrf_scores = {}

for rank, doc in enumerate(docs_vector, start=1):
    rrf_scores[doc] = rrf_scores.get(doc, 0) + rrf_score(rank, k)

for rank, doc in enumerate(docs_bm25, start=1):
    rrf_scores[doc] = rrf_scores.get(doc, 0) + rrf_score(rank, k)

# 결과 정렬
sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
print("RRF 최종 순위:")
for doc, score in sorted_docs:
    print(f"  {doc}: {score:.4f}")

# 출력:
# 문서A: 0.0328 (벡터 1위 + BM25 2위)
# 문서C: 0.0307 (벡터 3위 + BM25 1위)
# 문서B: 0.0286 (벡터 2위 + BM25 4위)
# 문서E: 0.0143 (BM25 3위만)
# 문서D: 0.0156 (벡터 4위만)
```

### 5.3 RRF vs 가중치 합산(Weighted Sum)

```
가중치 합산: score(d) = α × score_vector(d) + β × score_bm25(d)

문제점:
1. 벡터 점수: [0.0~1.0] 범위
2. BM25 점수: [0~수십] 범위 → 스케일 정규화 필요
3. 스케일 정규화가 잘못되면 한쪽이 지배적

RRF의 장점:
1. 점수 스케일 무관 (순위만 사용)
2. 하이퍼파라미터 k 하나만 튜닝
3. 안정적인 성능 (다양한 환경에서 검증됨)
```

LangChain의 `EnsembleRetriever`는 내부적으로 RRF를 사용한다.

---

## 6. 환경 설정

### 6.1 패키지 설치

```bash
# 기본 패키지
pip install chromadb langchain langchain-chroma langchain-community

# BM25
pip install rank-bm25

# Reranker
pip install FlagEmbedding

# 기타
pip install python-dotenv requests
```

### 6.2 .env 파일

```bash
# .env
EMBED_BASE_URL=http://localhost:11434/v1
EMBED_API_KEY=ollama
EMBED_MODEL=nomic-embed-text

# Ollama LLM
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2

# Chroma 설정
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=rag_docs

# HuggingFace 오프라인 캐시 경로
HF_CACHE_DIR=./hf_cache
```

### 6.3 Ollama 실행 확인

```bash
# Ollama 서버 상태 확인
curl http://localhost:11434/api/tags

# 임베딩 모델 확인
ollama list | grep nomic-embed-text

# 임베딩 테스트
curl http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{"model": "nomic-embed-text", "input": "안녕하세요"}'
```

---

## 7. Ollama 임베딩 클래스

```python
# embeddings.py
import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()


class OllamaEmbeddings(Embeddings):
    """
    Ollama 로컬 서버를 통한 임베딩 클래스
    OpenAI 호환 API (/v1/embeddings 엔드포인트) 사용
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        api_key: str = "ollama",
    ):
        self.base_url = base_url or os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.api_key = api_key or os.getenv("EMBED_API_KEY", "ollama")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서 임베딩 — 배치 처리"""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        """실제 임베딩 API 호출"""
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": text},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def get_dimension(self) -> int:
        """임베딩 차원 확인"""
        sample = self._embed("test")
        return len(sample)


# 싱글톤 인스턴스 (모듈 전체에서 재사용)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=os.getenv("EMBED_BASE_URL"),
        model=os.getenv("EMBED_MODEL"),
        api_key=os.getenv("EMBED_API_KEY"),
    )
```

**임베딩 차원 확인:**
```python
from embeddings import get_embeddings

embeddings = get_embeddings()
dim = embeddings.get_dimension()
print(f"nomic-embed-text 차원: {dim}")  # 768
```

---

## 8. Chroma 실전 사용

### 8.1 Chroma 클라이언트 종류

```python
import chromadb

# 1. 인메모리 (테스트용, 재시작 시 데이터 소멸)
client_memory = chromadb.Client()

# 2. 영속 DB (파일 저장, 권장)
client_persist = chromadb.PersistentClient(path="./chroma_db")

# 3. HTTP 서버 모드 (멀티 프로세스/원격 접근)
# 서버 실행: chroma run --path ./chroma_db --port 8000
client_http = chromadb.HttpClient(host="localhost", port=8000)
```

### 8.2 영속 DB 구성

```python
# chroma_client.py
import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_chroma_client() -> chromadb.PersistentClient:
    """
    영속 Chroma 클라이언트 생성
    데이터는 PERSIST_DIR에 SQLite + 파일로 저장됨
    """
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(
            anonymized_telemetry=False,  # 텔레메트리 비활성화
            allow_reset=True,             # reset() 허용 (개발 환경)
        ),
    )
```

**저장 구조 확인:**
```bash
ls ./chroma_db/
# chroma.sqlite3   ← 메타데이터, 임베딩, 컬렉션 정보
# [UUID]/          ← HNSW 인덱스 파일들
```

### 8.3 컬렉션 생성 및 기본 CRUD

```python
# collection_basics.py
import chromadb
from embeddings import get_embeddings
from langchain_chroma import Chroma

# Chroma 클라이언트 (LangChain 래퍼 없이 직접 사용)
raw_client = chromadb.PersistentClient(path="./chroma_db")

# 컬렉션 생성 (이미 있으면 가져옴)
collection = raw_client.get_or_create_collection(
    name="docs",
    metadata={
        "hnsw:space": "cosine",        # 코사인 유사도
        "hnsw:M": 16,                   # 연결 수
        "hnsw:construction_ef": 100,    # 구축 품질
        "hnsw:search_ef": 100,          # 검색 품질
    }
)

# 문서 추가 (임베딩 직접 계산)
embeddings = get_embeddings()

texts = [
    "RAG는 검색 증강 생성 기법으로 외부 지식을 LLM에 제공한다.",
    "Chroma는 Python 네이티브 벡터 데이터베이스다.",
    "BM25는 키워드 기반 렉시컬 검색 알고리즘이다.",
    "하이브리드 검색은 벡터와 키워드 검색을 결합한다.",
    "HNSW는 근사 최근접 이웃 검색 그래프 알고리즘이다.",
]

ids = [f"doc_{i}" for i in range(len(texts))]
vectors = embeddings.embed_documents(texts)

collection.add(
    ids=ids,
    embeddings=vectors,
    documents=texts,
    metadatas=[
        {"source": "rag_intro.md", "category": "rag", "level": 1},
        {"source": "chroma_guide.md", "category": "vectordb", "level": 1},
        {"source": "search_algo.md", "category": "search", "level": 2},
        {"source": "hybrid_search.md", "category": "search", "level": 2},
        {"source": "indexing.md", "category": "algorithm", "level": 3},
    ]
)

print(f"컬렉션 문서 수: {collection.count()}")
```

**조회 및 삭제:**
```python
# 특정 ID 조회
result = collection.get(ids=["doc_0"])
print(result["documents"])

# 쿼리 검색
query_vec = embeddings.embed_query("벡터 검색이란?")
results = collection.query(
    query_embeddings=[query_vec],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"거리: {dist:.4f} | {meta['source']} | {doc[:50]}")

# 문서 업데이트 (upsert)
collection.upsert(
    ids=["doc_0"],
    documents=["RAG는 검색 증강 생성 기법이다. 외부 지식베이스를 활용한다."],
    embeddings=embeddings.embed_documents(["RAG는 검색 증강 생성 기법이다. 외부 지식베이스를 활용한다."]),
    metadatas=[{"source": "rag_intro.md", "category": "rag", "level": 1, "updated": True}]
)

# 삭제
collection.delete(ids=["doc_4"])
```

### 8.4 LangChain Chroma 래퍼 사용

```python
# langchain_chroma_usage.py
import os
from langchain_chroma import Chroma
from langchain.schema import Document
from embeddings import get_embeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = get_embeddings()

# 문서 목록 준비
docs = [
    Document(
        page_content="RAG 파이프라인은 인덱싱, 검색, 생성 3단계로 구성된다.",
        metadata={"source": "rag_overview.md", "category": "rag", "page": 1}
    ),
    Document(
        page_content="벡터 임베딩은 텍스트를 고차원 수치 벡터로 변환한다.",
        metadata={"source": "embedding_guide.md", "category": "embedding", "page": 1}
    ),
    Document(
        page_content="코사인 유사도는 두 벡터 사이의 각도를 측정한다.",
        metadata={"source": "similarity.md", "category": "algorithm", "page": 1}
    ),
    Document(
        page_content="HNSW 알고리즘은 계층적 그래프 구조로 ANN 검색을 수행한다.",
        metadata={"source": "indexing.md", "category": "algorithm", "page": 2}
    ),
    Document(
        page_content="하이브리드 검색은 의미 검색과 키워드 검색을 결합해 품질을 높인다.",
        metadata={"source": "hybrid.md", "category": "search", "page": 1}
    ),
]

# 벡터스토어 생성 (from_documents: 임베딩 자동 계산)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_docs"),
    collection_metadata={"hnsw:space": "cosine"},
)

print(f"저장된 문서 수: {vectorstore._collection.count()}")

# 기존 컬렉션 로드 (재실행 시)
vectorstore_loaded = Chroma(
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
    embedding_function=embeddings,
    collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_docs"),
)
```

---

## 9. 컬렉션 설계 전략

### 9.1 단일 컬렉션 vs 멀티 컬렉션

**단일 컬렉션 전략 (권장 시작점):**
```python
# 모든 문서를 하나의 컬렉션에 저장, 메타데이터로 구분
collection = client.get_or_create_collection(name="all_docs")

# 문서 추가 시 풍부한 메타데이터 포함
collection.add(
    ids=["doc_001"],
    documents=["문서 내용..."],
    metadatas=[{
        "source": "파일명.md",
        "category": "rag",           # 카테고리
        "subcategory": "embedding",  # 서브카테고리
        "language": "ko",            # 언어
        "level": 2,                  # 난이도
        "created_at": "2026-01-15",  # 생성일
        "version": "1.0",            # 버전
        "tags": "embedding,vector",  # 태그 (리스트 불가 → 문자열)
    }]
)

# 장점: 관리 단순, 크로스 카테고리 검색 가능
# 단점: 접근 제어 불가, 대규모 시 성능 저하 가능
```

**멀티 컬렉션 전략 (격리가 필요한 경우):**
```python
# 컬렉션별 독립된 HNSW 인덱스
collections = {
    "rag_docs":       client.get_or_create_collection("rag_docs"),
    "user_manuals":   client.get_or_create_collection("user_manuals"),
    "api_references": client.get_or_create_collection("api_references"),
}

# 장점: 독립적 관리, 컬렉션별 다른 HNSW 설정 가능
# 단점: 크로스 컬렉션 검색 어려움, 관리 복잡성
```

### 9.2 메타데이터 스키마 설계

DA 관점에서 메타데이터 스키마는 사전에 설계해야 한다.

```python
# metadata_schema.py
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class DocumentMetadata:
    """
    Chroma 문서 메타데이터 스키마
    
    주의사항:
    - Chroma 메타데이터 값은 str, int, float, bool만 허용
    - 리스트, 딕셔너리 불가 → 직렬화 필요
    - 검색 필터에 사용할 필드는 인덱싱 고려
    """
    # 필수 필드
    source: str          # 원본 파일 경로 또는 URL
    chunk_index: int     # 문서 내 청크 순서
    total_chunks: int    # 해당 문서의 총 청크 수

    # 분류 필드
    category: str        # 대분류 (rag, search, embedding, ...)
    subcategory: str     # 소분류
    language: str        # 언어 코드 (ko, en, ...)

    # 품질 필드
    level: int           # 난이도 (1=기초, 2=중급, 3=고급)
    version: str         # 문서 버전

    # 시간 필드
    created_at: str      # YYYY-MM-DD 형식
    updated_at: str      # YYYY-MM-DD 형식

    # 선택 필드
    tags: str = ""       # 쉼표 구분 태그 ("rag,embedding,vector")
    author: str = ""     # 작성자
    page_number: int = 0  # PDF의 경우 페이지 번호

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_file(
        cls,
        filepath: str,
        chunk_index: int,
        total_chunks: int,
        category: str,
        **kwargs
    ) -> "DocumentMetadata":
        return cls(
            source=filepath,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            category=category,
            subcategory=kwargs.get("subcategory", ""),
            language=kwargs.get("language", "ko"),
            level=kwargs.get("level", 1),
            version=kwargs.get("version", "1.0"),
            created_at=datetime.now().strftime("%Y-%m-%d"),
            updated_at=datetime.now().strftime("%Y-%m-%d"),
            tags=kwargs.get("tags", ""),
        )


# 사용 예시
meta = DocumentMetadata.from_file(
    filepath="docs/rag_intro.md",
    chunk_index=0,
    total_chunks=5,
    category="rag",
    subcategory="overview",
    level=1,
    tags="rag,introduction,overview",
)
print(meta.to_dict())
```

### 9.3 증분 업데이트 (Upsert) 패턴

```python
# incremental_update.py
import hashlib
from langchain_chroma import Chroma
from langchain.schema import Document
from embeddings import get_embeddings


def get_doc_id(source: str, chunk_index: int) -> str:
    """
    결정적(deterministic) 문서 ID 생성
    같은 소스+청크 인덱스는 항상 같은 ID → upsert 가능
    """
    return hashlib.md5(f"{source}_{chunk_index}".encode()).hexdigest()


def get_content_hash(content: str) -> str:
    """컨텐츠 해시 — 변경 감지용"""
    return hashlib.md5(content.encode()).hexdigest()


def upsert_documents(
    vectorstore: Chroma,
    new_docs: list[Document],
    embeddings: object,
) -> dict:
    """
    문서 증분 업데이트
    - 새 문서: 추가
    - 변경된 문서: 업데이트
    - 삭제된 문서: 처리 안 함 (별도 로직 필요)
    """
    stats = {"added": 0, "updated": 0, "skipped": 0}

    for doc in new_docs:
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        doc_id = get_doc_id(source, chunk_idx)
        content_hash = get_content_hash(doc.page_content)

        # 기존 문서 확인
        existing = vectorstore._collection.get(
            ids=[doc_id],
            include=["metadatas"]
        )

        if existing["ids"]:
            existing_hash = existing["metadatas"][0].get("content_hash", "")
            if existing_hash == content_hash:
                stats["skipped"] += 1
                continue  # 내용 동일 → 스킵
            stats["updated"] += 1
        else:
            stats["added"] += 1

        # 메타데이터에 해시 추가
        doc.metadata["content_hash"] = content_hash

        # Upsert
        vector = embeddings.embed_documents([doc.page_content])[0]
        vectorstore._collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
        )

    return stats


# 사용 예시
embeddings = get_embeddings()
vs = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rag_docs",
)

docs_to_upsert = [
    Document(
        page_content="RAG 파이프라인 최신 내용",
        metadata={"source": "rag.md", "chunk_index": 0, "category": "rag"}
    )
]

result = upsert_documents(vs, docs_to_upsert, embeddings)
print(f"추가: {result['added']}, 업데이트: {result['updated']}, 스킵: {result['skipped']}")
```

---

## 10. similarity_search vs MMR

### 10.1 similarity_search (기본 유사도 검색)

```python
# 가장 유사한 k개 반환 — 중복/유사한 결과가 많을 수 있음
results = vectorstore.similarity_search(
    query="RAG 검색 방법",
    k=5,
)

# 점수와 함께 반환
results_with_scores = vectorstore.similarity_search_with_score(
    query="RAG 검색 방법",
    k=5,
)

for doc, score in results_with_scores:
    print(f"점수: {score:.4f} | {doc.page_content[:60]}")
```

**문제점:**
```
쿼리: "Python으로 RAG 구현하기"

결과:
  1. "Python으로 RAG를 구현하는 방법을 알아보자." (score: 0.95)
  2. "Python에서 RAG 파이프라인을 만드는 방법" (score: 0.94)  ← 1번과 거의 동일
  3. "Python RAG 구현 예제 코드" (score: 0.93)               ← 1번과 거의 동일
  4. "LangChain으로 RAG를 구성한다." (score: 0.88)
  5. "벡터 검색을 활용한 RAG 시스템" (score: 0.85)
```

### 10.2 MMR (Maximal Marginal Relevance)

중복을 줄이고 다양성을 높이는 검색 방식.

```
MMR(dᵢ) = arg max [λ × sim(dᵢ, q) - (1-λ) × max_{dⱼ∈S} sim(dᵢ, dⱼ)]

q    = 쿼리
dᵢ   = 후보 문서
S    = 이미 선택된 문서 집합
λ    = 관련성/다양성 가중치 (0~1)
  λ=1 → 순수 유사도 검색 (similarity_search와 동일)
  λ=0 → 완전 다양성 (쿼리 무관, 다양성만)
  λ=0.5 → 균형 (권장)
```

```python
# MMR 검색
results_mmr = vectorstore.max_marginal_relevance_search(
    query="RAG 검색 방법",
    k=5,                # 최종 반환 수
    fetch_k=20,         # 후보 풀 크기 (k보다 커야 함, 보통 4~10배)
    lambda_mult=0.5,    # 다양성/관련성 균형 (0~1)
)

for doc in results_mmr:
    print(f"- {doc.page_content[:60]}")
```

### 10.3 similarity_search vs MMR 비교

```python
# comparison.py — 두 방법 결과 직접 비교
def compare_search_methods(vectorstore, query: str, k: int = 5):
    print(f"\n쿼리: '{query}'\n")
    print("=" * 60)

    # 일반 유사도 검색
    print("[similarity_search 결과]")
    sim_results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(sim_results, 1):
        print(f"  {i}. [{score:.4f}] {doc.page_content[:50]}")

    print()

    # MMR 검색
    print("[MMR 검색 결과 (λ=0.5)]")
    mmr_results = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=k*4, lambda_mult=0.5
    )
    for i, doc in enumerate(mmr_results, 1):
        print(f"  {i}. {doc.page_content[:50]}")


# 실행
compare_search_methods(vectorstore, "RAG 구현 방법")
```

**선택 기준:**

| 상황 | 추천 |
|------|------|
| 단일 사실 질문 ("X의 정의는?") | similarity_search |
| 긴 답변 생성 (컨텍스트 다양성 필요) | MMR |
| 중복 문서가 많은 컬렉션 | MMR |
| 다각도 정보 수집 | MMR (λ=0.3~0.5) |
| 정밀 매칭 필요 | similarity_search |

---

## 11. 메타데이터 필터링

### 11.1 where 조건 사용

```python
# Chroma 메타데이터 필터 — where 절
from langchain_chroma import Chroma

# 단일 조건
results = vectorstore.similarity_search(
    query="검색 알고리즘",
    k=5,
    filter={"category": "algorithm"},  # LangChain 래퍼
)

# Chroma 네이티브 필터 (더 많은 연산자 지원)
raw_results = vectorstore._collection.query(
    query_embeddings=[embeddings.embed_query("검색 알고리즘")],
    n_results=5,
    where={"category": {"$eq": "algorithm"}},
    include=["documents", "metadatas", "distances"]
)
```

### 11.2 Chroma 필터 연산자

```python
# 지원 연산자
operators = {
    "$eq":  "같음",
    "$ne":  "같지 않음",
    "$gt":  "초과",
    "$gte": "이상",
    "$lt":  "미만",
    "$lte": "이하",
    "$in":  "포함 (리스트)",
    "$nin": "미포함 (리스트)",
    "$and": "논리 AND",
    "$or":  "논리 OR",
}

# 예시들
# 1. 레벨 2 이상의 알고리즘 문서
where_advanced = {
    "$and": [
        {"category": {"$eq": "algorithm"}},
        {"level": {"$gte": 2}}
    ]
}

# 2. 특정 카테고리 제외
where_not_basic = {
    "$and": [
        {"category": {"$ne": "basic"}},
        {"language": {"$eq": "ko"}}
    ]
}

# 3. 여러 카테고리 중 하나
where_multi_cat = {
    "category": {"$in": ["rag", "embedding", "search"]}
}

# 4. 날짜 범위 (문자열 비교 — YYYY-MM-DD 형식이면 알파벳 정렬로 동작)
where_recent = {
    "created_at": {"$gte": "2026-01-01"}
}

# 검색에 적용
results = vectorstore._collection.query(
    query_embeddings=[embeddings.embed_query("하이브리드 검색")],
    n_results=5,
    where=where_advanced,
    include=["documents", "metadatas", "distances"]
)
```

### 11.3 where_document (문서 내용 필터)

```python
# 문서 내용에 특정 문자열이 포함된 것만 검색
results = vectorstore._collection.query(
    query_embeddings=[embeddings.embed_query("검색 방법")],
    n_results=5,
    where_document={"$contains": "BM25"},  # 본문에 "BM25" 포함
    include=["documents", "metadatas"]
)

# 내용 필터 연산자
doc_operators = {
    "$contains":     "문자열 포함",
    "$not_contains": "문자열 미포함",
}
```

### 11.4 Retriever로 감싸기 (LangChain 체인 통합)

```python
from langchain_chroma import Chroma

# 필터가 적용된 Retriever 생성
filtered_retriever = vectorstore.as_retriever(
    search_type="similarity",       # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={
        "k": 5,
        "filter": {                  # 메타데이터 필터
            "category": "rag",
            "level": {"$lte": 2},
        },
    }
)

# MMR + 필터
mmr_filtered_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5,
        "filter": {"language": "ko"},
    }
)

# 점수 임계값 필터 (유사도가 일정 이상인 것만)
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # 코사인 유사도 0.7 이상만
        "k": 5,
    }
)
```

---

## 12. 하이브리드 검색 (Chroma + BM25)

### 12.1 EnsembleRetriever 개요

```
하이브리드 검색 흐름:

쿼리 ──┬──→ Chroma 벡터 검색 ──→ [결과 A 순위]
       │                                    │
       └──→ BM25 렉시컬 검색 ──→ [결과 B 순위] │
                                            ↓
                                    RRF 점수 계산
                                            ↓
                                    통합 최종 순위
```

### 12.2 완전한 하이브리드 검색 구현

```python
# hybrid_search.py
import os
from typing import List
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from embeddings import get_embeddings

load_dotenv()


def build_hybrid_retriever(
    docs: List[Document],
    persist_dir: str = None,
    collection_name: str = "hybrid_docs",
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 5,
) -> EnsembleRetriever:
    """
    Chroma + BM25 하이브리드 리트리버 구성

    Args:
        docs: 인덱싱할 문서 리스트
        persist_dir: Chroma 영속 디렉토리 (None이면 인메모리)
        collection_name: Chroma 컬렉션 이름
        vector_weight: 벡터 검색 가중치 (0~1)
        bm25_weight: BM25 가중치 (0~1), vector_weight + bm25_weight = 1.0
        k: 최종 반환 문서 수

    Returns:
        EnsembleRetriever (내부적으로 RRF 적용)
    """
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embeddings = get_embeddings()

    # 1. Chroma 벡터 리트리버
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # 2. BM25 렉시컬 리트리버
    bm25_retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=_korean_preprocess,  # 한국어 전처리
    )
    bm25_retriever.k = k

    # 3. EnsembleRetriever — RRF 내부 적용
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    return ensemble_retriever


def _korean_preprocess(text: str) -> List[str]:
    """
    한국어 BM25 전처리
    - 공백 기준 토크나이징 (기본)
    - 실제 운영 시 konlpy 형태소 분석기 사용 권장
    """
    # 소문자 변환 + 공백 분리
    tokens = text.lower().split()
    # 1글자 이하 제거 (조사, 특수문자 등)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# 사용 예시
if __name__ == "__main__":
    # 테스트 문서
    test_docs = [
        Document(
            page_content="RAG는 검색 증강 생성으로, 외부 지식베이스를 LLM에 활용하는 기법이다.",
            metadata={"source": "rag.md", "category": "rag"}
        ),
        Document(
            page_content="Chroma는 Python 네이티브 벡터 데이터베이스로 로컬 환경에 적합하다.",
            metadata={"source": "chroma.md", "category": "vectordb"}
        ),
        Document(
            page_content="BM25는 TF-IDF를 개선한 키워드 기반 검색 알고리즘이다.",
            metadata={"source": "bm25.md", "category": "search"}
        ),
        Document(
            page_content="하이브리드 검색은 벡터 검색과 키워드 검색을 RRF로 결합한다.",
            metadata={"source": "hybrid.md", "category": "search"}
        ),
        Document(
            page_content="HNSW는 계층적 그래프 구조로 ANN 검색을 수행하는 인덱싱 알고리즘이다.",
            metadata={"source": "hnsw.md", "category": "algorithm"}
        ),
        Document(
            page_content="Reranking은 초기 검색 결과를 재순위화하여 품질을 개선하는 후처리 단계다.",
            metadata={"source": "rerank.md", "category": "search"}
        ),
    ]

    retriever = build_hybrid_retriever(
        docs=test_docs,
        collection_name="test_hybrid",
        vector_weight=0.6,
        bm25_weight=0.4,
        k=3,
    )

    # 검색 실행
    query = "벡터 검색 알고리즘"
    results = retriever.invoke(query)

    print(f"\n쿼리: '{query}'")
    print(f"검색 결과 ({len(results)}개):")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.metadata.get('category')}] {doc.page_content[:60]}")
```

### 12.3 기존 Chroma DB에서 하이브리드 검색

```python
# hybrid_from_existing.py — 이미 구축된 Chroma DB 활용
import os
from typing import List
from dotenv import load_dotenv

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from embeddings import get_embeddings

load_dotenv()


def build_hybrid_from_existing_chroma(
    persist_dir: str,
    collection_name: str,
    vector_weight: float = 0.6,
    k: int = 5,
) -> tuple[EnsembleRetriever, int]:
    """
    기존 Chroma 컬렉션에서 하이브리드 리트리버 구성
    BM25 인덱스는 Chroma의 문서로 재구성
    """
    embeddings = get_embeddings()

    # 기존 Chroma 로드
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # 전체 문서 추출 (BM25 인덱스 재구성용)
    all_data = vectorstore._collection.get(include=["documents", "metadatas"])
    all_docs = []
    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        from langchain.schema import Document
        all_docs.append(Document(page_content=content, metadata=meta or {}))

    doc_count = len(all_docs)
    print(f"기존 컬렉션 문서 수: {doc_count}")

    # 벡터 리트리버
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # BM25 리트리버 (전체 문서로 구성)
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # 앙상블
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, 1 - vector_weight],
    )

    return ensemble, doc_count
```

### 12.4 RRF 수동 구현 (내부 동작 이해)

```python
# manual_rrf.py — EnsembleRetriever 내부 동작 직접 구현
from typing import List, Dict
from langchain.schema import Document


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    여러 랭킹 리스트를 RRF로 통합
    
    Args:
        result_lists: [[벡터검색 결과], [BM25 결과], ...]
        k: RRF 상수 (낮을수록 상위 순위 강조, 기본 60)
    
    Returns:
        RRF 점수로 정렬된 통합 결과
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            # 문서 고유 키 (내용 기반)
            doc_key = doc.page_content

            # RRF 점수 누적
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (k + rank)
            doc_map[doc_key] = doc

    # 점수 내림차순 정렬
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in sorted_items]


# 사용 예시
if __name__ == "__main__":
    from langchain.schema import Document

    # 벡터 검색 결과 (의미 유사도 순)
    vector_results = [
        Document(page_content="벡터 임베딩은 텍스트를 수치로 변환한다"),
        Document(page_content="코사인 유사도로 벡터 간 거리를 측정한다"),
        Document(page_content="HNSW는 ANN 검색 알고리즘이다"),
    ]

    # BM25 결과 (키워드 매칭 순)
    bm25_results = [
        Document(page_content="코사인 유사도로 벡터 간 거리를 측정한다"),  # 키워드 정확 매칭
        Document(page_content="벡터 임베딩은 텍스트를 수치로 변환한다"),
        Document(page_content="검색 알고리즘 비교"),
    ]

    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    print("RRF 통합 결과:")
    for i, doc in enumerate(fused, 1):
        print(f"  {i}. {doc.page_content}")
```

### 12.5 가중치 튜닝 전략

```python
# weight_tuning.py — 최적 가중치 탐색
def evaluate_hybrid_weight(
    docs: List[Document],
    test_queries: List[str],
    ground_truth: List[List[str]],  # 쿼리별 정답 문서 source 리스트
    vector_weights: List[float] = None,
    k: int = 5,
) -> dict:
    """
    다양한 가중치로 Recall@k 측정
    """
    if vector_weights is None:
        vector_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    results = {}

    for vw in vector_weights:
        bw = round(1.0 - vw, 2)
        retriever = build_hybrid_retriever(
            docs=docs,
            collection_name=f"eval_{int(vw*10)}",
            vector_weight=vw,
            bm25_weight=bw,
            k=k,
        )

        recalls = []
        for query, gt_sources in zip(test_queries, ground_truth):
            retrieved = retriever.invoke(query)
            retrieved_sources = {d.metadata.get("source", "") for d in retrieved}
            recall = len(retrieved_sources & set(gt_sources)) / max(len(gt_sources), 1)
            recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls)
        results[f"v{vw}_b{bw}"] = avg_recall
        print(f"벡터={vw:.1f} BM25={bw:.1f} → Recall@{k}: {avg_recall:.3f}")

    # 최적 가중치 출력
    best = max(results, key=results.get)
    print(f"\n최적 가중치: {best} (Recall@{k}: {results[best]:.3f})")
    return results
```

---

## 13. BGE Reranker 오프라인 적용

### 13.1 오프라인 다운로드 절차

> **폐쇄망 환경 필수 사전 작업**

외부 PC (인터넷 연결)에서 실행:
```python
# download_reranker.py (외부 PC에서 실행)
from huggingface_hub import snapshot_download

# BGE Reranker v2 M3 — 한국어 지원, 다국어 모델
snapshot_download(
    "BAAI/bge-reranker-v2-m3",
    cache_dir="./hf_cache",
)

print("다운로드 완료. ./hf_cache 폴더를 zip으로 압축해서 전달하세요.")
```

```bash
# 압축 및 전달
zip -r hf_cache_reranker.zip ./hf_cache/

# 폐쇄망 PC에서 압축 해제
unzip hf_cache_reranker.zip -d ./
```

### 13.2 BGE Reranker 오프라인 로드

```python
# reranker.py
import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")


def load_bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """
    오프라인 BGE Reranker 로드
    
    반드시 local_files_only=True 사용 (외부 네트워크 접근 차단)
    """
    from huggingface_hub import snapshot_download
    from FlagEmbedding import FlagReranker

    # 로컬 캐시에서 모델 경로 확인
    model_path = snapshot_download(
        model_name,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,  # 반드시 True — 외부 접근 금지
    )
    print(f"모델 경로: {model_path}")

    # FP16으로 로드 (메모리 절약, CPU에서도 동작)
    reranker = FlagReranker(
        model_path,
        use_fp16=True,  # FP16: 메모리 절감, 속도 향상
    )
    return reranker
```

### 13.3 Reranking 파이프라인

```python
# reranking_pipeline.py
import os
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./hf_cache")


class BGEReranker:
    """
    BGE Reranker 래퍼 클래스
    하이브리드 검색 결과를 재순위화
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_dir: str = None,
        use_fp16: bool = True,
    ):
        from huggingface_hub import snapshot_download
        from FlagEmbedding import FlagReranker

        cache_dir = cache_dir or HF_CACHE_DIR

        model_path = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )

        self.reranker = FlagReranker(model_path, use_fp16=use_fp16)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_n: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        쿼리-문서 쌍의 관련성 점수 계산 후 재순위화

        Args:
            query: 검색 쿼리
            docs: 재순위화할 문서 리스트
            top_n: 상위 N개 반환

        Returns:
            (문서, 점수) 튜플 리스트 (점수 내림차순)
        """
        if not docs:
            return []

        # 쿼리-문서 쌍 구성
        pairs = [(query, doc.page_content) for doc in docs]

        # 점수 계산 (교차 인코더 방식 — 쿼리+문서를 함께 인코딩)
        scores = self.reranker.compute_score(pairs, normalize=True)  # 0~1 정규화

        # 점수 기준 정렬
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 상위 N개 반환
        top_docs = [(doc, score) for score, doc in scored_docs[:top_n]]
        return top_docs

    def rerank_docs_only(
        self,
        query: str,
        docs: List[Document],
        top_n: int = 5,
    ) -> List[Document]:
        """점수 없이 문서만 반환"""
        return [doc for doc, _ in self.rerank(query, docs, top_n)]
```

### 13.4 교차 인코더(Cross-Encoder) 원리

```
Bi-Encoder (임베딩 기반):
  쿼리 → [인코더] → 쿼리 벡터
  문서 → [인코더] → 문서 벡터
  유사도 = cosine(쿼리벡터, 문서벡터)
  
  장점: 빠름 (벡터를 미리 계산 가능)
  단점: 쿼리-문서 상호작용 없음

Cross-Encoder (Reranker):
  [쿼리 + 문서] → [인코더] → 관련성 점수
  
  장점: 쿼리-문서 상호작용 캡처 → 정확도 높음
  단점: 실시간 계산 필요 → 느림 (전체 검색에 쓸 수 없음)

최적 전략:
  Bi-Encoder → 후보 50~100개 빠르게 선별
  Cross-Encoder → 후보 중 상위 5~10개 정밀 선택
```

### 13.5 Reranker 없이 구현 (score 기반 필터)

```python
# BGE Reranker가 없을 때의 대안
# 임베딩 유사도로 재순위화 (품질은 낮지만 동작)
def simple_rerank_by_embedding(
    query: str,
    docs: List[Document],
    embeddings,
    top_n: int = 5,
) -> List[Document]:
    """
    임베딩 유사도로 재순위화 (Cross-Encoder 대안)
    BGE Reranker보다 정확도 낮지만 추가 모델 불필요
    """
    import numpy as np

    query_vec = np.array(embeddings.embed_query(query))

    scored = []
    for doc in docs:
        doc_vec = np.array(embeddings.embed_documents([doc.page_content])[0])
        score = np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        )
        scored.append((score, doc))

    scored.sort(reverse=True)
    return [doc for _, doc in scored[:top_n]]
```

---

## 14. 전체 파이프라인 통합

### 14.1 완전한 RAG 파이프라인

```python
# rag_pipeline_hybrid.py
"""
완전한 하이브리드 RAG 파이프라인
구성: Ollama 임베딩 + Chroma + BM25 + BGE Reranker + Ollama LLM
"""
import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class HybridRAGPipeline:
    """
    Chroma + BM25 + BGE Reranker 기반 하이브리드 RAG
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = "rag_docs",
        vector_weight: float = 0.6,
        retrieval_k: int = 10,    # 초기 검색 수 (reranking 전)
        final_k: int = 5,         # 최종 컨텍스트 수
        use_reranker: bool = True,
    ):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = collection_name
        self.vector_weight = vector_weight
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.use_reranker = use_reranker

        # 컴포넌트 초기화
        self._init_embeddings()
        self._init_llm()
        self._init_reranker()
        self.vectorstore = None
        self.hybrid_retriever = None

    def _init_embeddings(self):
        from embeddings import get_embeddings
        self.embeddings = get_embeddings()

    def _init_llm(self):
        self.llm = Ollama(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LLM_MODEL", "llama3.2"),
            temperature=0.1,
        )

    def _init_reranker(self):
        if self.use_reranker:
            try:
                from reranker import BGEReranker
                self.reranker = BGEReranker()
                print("BGE Reranker 로드 완료")
            except Exception as e:
                print(f"Reranker 로드 실패 (비활성화): {e}")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None

    def index_documents(self, docs: List[Document]):
        """문서 인덱싱"""
        from langchain_chroma import Chroma
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        # Chroma 벡터스토어
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_k}
        )

        # BM25 리트리버
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.retrieval_k

        # 앙상블
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.vector_weight, 1 - self.vector_weight],
        )

        print(f"인덱싱 완료: {len(docs)}개 문서")

    def retrieve(self, query: str) -> List[Document]:
        """하이브리드 검색 + Reranking"""
        if self.hybrid_retriever is None:
            raise RuntimeError("먼저 index_documents()를 호출하세요.")

        # 1. 하이브리드 검색 (RRF 통합)
        candidates = self.hybrid_retriever.invoke(query)

        # 2. Reranking (BGE Cross-Encoder)
        if self.use_reranker and self.reranker:
            final_docs = self.reranker.rerank_docs_only(
                query=query,
                docs=candidates,
                top_n=self.final_k,
            )
        else:
            final_docs = candidates[:self.final_k]

        return final_docs

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """컨텍스트 기반 답변 생성"""
        context = "\n\n".join([
            f"[문서 {i+1}] {doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답하세요.
컨텍스트에 없는 내용은 "제공된 문서에서 찾을 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    def query(self, question: str, verbose: bool = False) -> str:
        """전체 RAG 파이프라인 실행"""
        # 검색
        docs = self.retrieve(question)

        if verbose:
            print(f"\n[검색 결과 {len(docs)}개]")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                print(f"  {i}. [{src}] {doc.page_content[:60]}...")

        # 생성
        answer = self.generate(question, docs)
        return answer


# 실행 예시
if __name__ == "__main__":
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import TextLoader

    # 파이프라인 초기화
    pipeline = HybridRAGPipeline(
        collection_name="demo_rag",
        vector_weight=0.6,
        retrieval_k=10,
        final_k=5,
        use_reranker=True,
    )

    # 테스트 문서 생성
    sample_docs = [
        Document(
            page_content="RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기법으로, "
                         "LLM이 외부 지식베이스에서 관련 문서를 검색하여 답변 품질을 높인다.",
            metadata={"source": "rag_intro.md", "category": "rag"}
        ),
        Document(
            page_content="Chroma는 Python 네이티브 벡터 데이터베이스다. "
                         "HNSW 인덱스를 사용하며 메타데이터 필터링을 지원한다. "
                         "SQLite 기반 영속 저장을 제공한다.",
            metadata={"source": "chroma_guide.md", "category": "vectordb"}
        ),
        Document(
            page_content="BM25 알고리즘은 TF-IDF를 개선한 키워드 검색 알고리즘이다. "
                         "단어 빈도 포화와 문서 길이 정규화를 적용한다. "
                         "k1=1.5, b=0.75가 기본 파라미터다.",
            metadata={"source": "bm25_theory.md", "category": "search"}
        ),
        Document(
            page_content="하이브리드 검색은 벡터 의미 검색과 BM25 키워드 검색을 RRF로 결합한다. "
                         "단독 벡터 검색 대비 Recall@5를 약 10~15% 개선할 수 있다.",
            metadata={"source": "hybrid_search.md", "category": "search"}
        ),
        Document(
            page_content="BGE Reranker는 교차 인코더(Cross-Encoder) 방식으로 "
                         "쿼리와 문서를 동시에 인코딩하여 더 정확한 관련성 점수를 계산한다. "
                         "BAAI에서 개발했으며 한국어를 포함한 다국어를 지원한다.",
            metadata={"source": "reranker_guide.md", "category": "reranking"}
        ),
    ]

    # 인덱싱
    pipeline.index_documents(sample_docs)

    # 질의
    questions = [
        "RAG의 주요 구성 요소는 무엇인가요?",
        "BM25의 파라미터 k1과 b는 무엇을 의미하나요?",
        "하이브리드 검색이 단순 벡터 검색보다 좋은 이유는?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"질문: {q}")
        answer = pipeline.query(q, verbose=True)
        print(f"\n답변: {answer}")
```

---

## 15. 검색 품질 평가

### 15.1 평가 지표

**Recall@K:**
```
Recall@K = |관련 문서 ∩ 검색된 상위 K개| / |전체 관련 문서|

예시:
  관련 문서: {A, B, C}
  검색 결과 상위 5: {A, D, E, B, F}
  Recall@5 = |{A, B}| / |{A, B, C}| = 2/3 ≈ 0.667
```

**Precision@K:**
```
Precision@K = |관련 문서 ∩ 검색된 상위 K개| / K

예시:
  검색 결과 상위 5: {A, D, E, B, F}  (A, B가 관련)
  Precision@5 = 2/5 = 0.4
```

**MRR (Mean Reciprocal Rank):**
```
MRR = (1/|Q|) × Σ (1/rankₖ)

rankₖ = 첫 번째 관련 문서의 순위

예시:
  쿼리 1: 첫 관련 문서가 3위 → 1/3
  쿼리 2: 첫 관련 문서가 1위 → 1/1
  MRR = (1/3 + 1/1) / 2 = 0.667
```

**NDCG@K (Normalized Discounted Cumulative Gain):**
```
DCG@K = Σᵢ₌₁ᴷ relᵢ / log₂(i+1)

relᵢ = 순위 i의 문서 관련성 점수 (0~1 또는 0/1)
NDCG@K = DCG@K / IDCG@K  (IDCG: 이상적 DCG)
```

### 15.2 평가 코드

```python
# evaluation.py
from typing import List, Dict, Callable
from dataclasses import dataclass
import math


@dataclass
class EvalResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


def evaluate_retriever(
    retriever,
    test_queries: List[str],
    ground_truth: List[List[str]],  # 쿼리별 관련 문서 ID/내용 리스트
    k: int = 5,
    get_doc_id: Callable = None,
) -> EvalResult:
    """
    리트리버 성능 평가

    Args:
        retriever: LangChain Retriever
        test_queries: 테스트 쿼리 리스트
        ground_truth: 쿼리별 정답 문서 식별자 리스트
        k: 평가할 상위 K개
        get_doc_id: 문서에서 ID 추출 함수
    """
    if get_doc_id is None:
        get_doc_id = lambda doc: doc.metadata.get("source", doc.page_content[:50])

    recalls, precisions, rrs, ndcgs = [], [], [], []

    for query, gt_ids in zip(test_queries, ground_truth):
        retrieved = retriever.invoke(query)[:k]
        retrieved_ids = [get_doc_id(doc) for doc in retrieved]

        gt_set = set(gt_ids)

        # Recall@K
        hits = sum(1 for rid in retrieved_ids if rid in gt_set)
        recall = hits / max(len(gt_set), 1)
        recalls.append(recall)

        # Precision@K
        precision = hits / k
        precisions.append(precision)

        # MRR
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in gt_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        # NDCG@K
        dcg = sum(
            (1.0 if rid in gt_set else 0.0) / math.log2(rank + 1)
            for rank, rid in enumerate(retrieved_ids, start=1)
        )
        ideal_hits = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return EvalResult(
        recall_at_k=sum(recalls) / len(recalls),
        precision_at_k=sum(precisions) / len(precisions),
        mrr=sum(rrs) / len(rrs),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        k=k,
    )


def compare_retrievers(
    retrievers: Dict[str, object],
    test_queries: List[str],
    ground_truth: List[List[str]],
    k: int = 5,
) -> Dict[str, EvalResult]:
    """
    여러 리트리버 성능 비교
    """
    results = {}
    print(f"\n{'='*70}")
    print(f"{'리트리버':<25} {'Recall@'+str(k):<15} {'Precision@'+str(k):<15} {'MRR':<10} {'NDCG@'+str(k):<10}")
    print(f"{'-'*70}")

    for name, retriever in retrievers.items():
        result = evaluate_retriever(retriever, test_queries, ground_truth, k)
        results[name] = result
        print(
            f"{name:<25} {result.recall_at_k:<15.4f} {result.precision_at_k:<15.4f} "
            f"{result.mrr:<10.4f} {result.ndcg_at_k:<10.4f}"
        )

    print(f"{'='*70}")
    return results
```

### 15.3 검색 전략별 성능 비교 실험

```python
# experiment.py — 벡터 vs BM25 vs 하이브리드 vs 하이브리드+Reranker
def run_full_experiment(docs: List[Document], test_queries, ground_truth, k=5):
    from embeddings import get_embeddings
    from langchain_chroma import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    embeddings = get_embeddings()

    # 1. 순수 벡터 검색
    vs = Chroma.from_documents(docs, embeddings, collection_name="exp_vector")
    vector_ret = vs.as_retriever(search_kwargs={"k": k})

    # 2. BM25만
    bm25_ret = BM25Retriever.from_documents(docs)
    bm25_ret.k = k

    # 3. 하이브리드
    hybrid_ret = EnsembleRetriever(
        retrievers=[vector_ret, bm25_ret],
        weights=[0.6, 0.4],
    )

    # 4. 하이브리드 + Reranker (래퍼)
    from reranker import BGEReranker

    class HybridWithReranker:
        def __init__(self, base_retriever, reranker, top_n):
            self.base = base_retriever
            self.reranker = reranker
            self.top_n = top_n

        def invoke(self, query):
            candidates = self.base.invoke(query)
            return self.reranker.rerank_docs_only(query, candidates, self.top_n)

    reranker = BGEReranker()
    hybrid_rerank_ret = HybridWithReranker(hybrid_ret, reranker, top_n=k)

    retrievers = {
        "벡터 검색 (Chroma)":         vector_ret,
        "BM25만":                      bm25_ret,
        "하이브리드 (RRF)":            hybrid_ret,
        "하이브리드 + BGE Reranker":   hybrid_rerank_ret,
    }

    return compare_retrievers(retrievers, test_queries, ground_truth, k)
```

---

## 16. 실습 과제

### 과제 1: 임베딩 탐구 (기초)

**목표:** 임베딩의 의미와 유사도 계산을 직접 확인한다.

```python
# task1_embedding_exploration.py
"""
실습 과제 1: 임베딩 탐구

목표:
1. 의미적으로 유사한 문장 쌍과 비유사한 문장 쌍 각 5개 준비
2. OllamaEmbeddings로 임베딩 계산
3. 코사인 유사도 계산 및 시각화
4. "왕 - 남자 + 여자 ≈ 여왕" 유사 실험 (한국어로)
"""
import numpy as np
from embeddings import get_embeddings


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


embeddings = get_embeddings()

# TODO 1: 유사 문장 쌍 실험
similar_pairs = [
    ("고양이가 귀엽다", "강아지가 사랑스럽다"),
    # ... 4개 더 추가
]

dissimilar_pairs = [
    ("오늘 날씨가 맑다", "양자역학 방정식"),
    # ... 4개 더 추가
]

print("[유사 문장 쌍]")
for s1, s2 in similar_pairs:
    v1 = embeddings.embed_query(s1)
    v2 = embeddings.embed_query(s2)
    sim = cosine_similarity(v1, v2)
    print(f"  {sim:.4f} | '{s1}' vs '{s2}'")

print("\n[비유사 문장 쌍]")
for s1, s2 in dissimilar_pairs:
    v1 = embeddings.embed_query(s1)
    v2 = embeddings.embed_query(s2)
    sim = cosine_similarity(v1, v2)
    print(f"  {sim:.4f} | '{s1}' vs '{s2}'")

# TODO 2: 벡터 산술 실험
# "왕" - "남성" + "여성" ≈ ?
words = ["왕", "남성", "여성", "여왕", "공주", "왕자"]
vecs = {w: np.array(embeddings.embed_query(w)) for w in words}

result = vecs["왕"] - vecs["남성"] + vecs["여성"]
print("\n[벡터 산술: 왕 - 남성 + 여성]")
for word, vec in vecs.items():
    sim = cosine_similarity(result, vec)
    print(f"  {sim:.4f} | {word}")
```

**제출 조건:**
- [ ] 유사 쌍 평균 코사인 유사도 > 0.7
- [ ] 비유사 쌍 평균 코사인 유사도 < 0.4
- [ ] 벡터 산술 결과에서 "여왕" 또는 "공주"가 상위 2위 이내

---

### 과제 2: Chroma 컬렉션 설계 (중급)

**목표:** 실제 데이터를 Chroma에 저장하고 다양한 검색 방식을 비교한다.

```python
# task2_chroma_design.py
"""
실습 과제 2: Chroma 컬렉션 설계 및 검색 비교

사용 데이터: rag/_workspace/00_input.md 또는 직접 준비한 한국어 문서 (TXT/MD)
청킹: RecursiveCharacterTextSplitter (chunk_size=300, overlap=50)

목표:
1. 메타데이터 스키마 설계 (최소 5개 필드)
2. 문서 인덱싱
3. similarity_search vs MMR 결과 비교 (3개 쿼리)
4. 메타데이터 필터 검색 구현
"""
```

**필수 구현 항목:**
- [ ] `DocumentMetadata` 데이터클래스 정의 (5필드 이상)
- [ ] Chroma 영속 DB에 문서 저장
- [ ] similarity_search와 MMR 결과 나란히 출력
- [ ] 2가지 이상의 메타데이터 필터 쿼리 실행

---

### 과제 3: 하이브리드 검색 vs 단순 벡터 검색 (중급~고급)

**목표:** 두 방식의 Recall@5를 직접 측정하고 비교한다.

```python
# task3_hybrid_evaluation.py
"""
실습 과제 3: 하이브리드 검색 성능 평가

1. 테스트 데이터셋 준비
   - 문서 20개 이상
   - 테스트 쿼리 10개
   - 각 쿼리의 정답 문서 (ground truth) 작성

2. 평가 대상 리트리버 3가지
   - 순수 벡터 검색 (Chroma)
   - BM25만
   - 하이브리드 (EnsembleRetriever)

3. Recall@5, MRR 비교표 출력

기대 결과:
  순수 벡터:  Recall@5 ≈ 0.65~0.75
  BM25만:     Recall@5 ≈ 0.55~0.65 (키워드 정확 일치가 적을 때)
  하이브리드: Recall@5 ≈ 0.75~0.85
"""
```

**제출 조건:**
- [ ] 3가지 방법의 Recall@5 비교표 출력
- [ ] 하이브리드가 단순 벡터 대비 Recall@5 개선 확인
- [ ] 가중치 (vector:BM25 = 5:5, 6:4, 7:3) 별 성능 변화 측정

---

### 과제 4: 전체 파이프라인 (고급)

**목표:** Chroma + BM25 + BGE Reranker 완전한 RAG 파이프라인을 구축하고 Recall@5를 측정한다.

```python
# task4_full_pipeline.py
"""
실습 과제 4: 전체 RAG 파이프라인 완성

필수 구현:
1. 문서 로더 (TXT, MD, CSV 중 하나 이상)
2. 청킹 (RecursiveCharacterTextSplitter)
3. Chroma 영속 DB + 메타데이터 스키마
4. 하이브리드 검색 (EnsembleRetriever)
5. BGE Reranker (오프라인)
6. Ollama LLM으로 답변 생성
7. Recall@5 평가 (최소 10개 쿼리)

평가 항목:
  - 벡터만  vs 하이브리드 vs 하이브리드+Reranker Recall@5 비교
  - 3가지 다른 쿼리로 답변 품질 체감 비교 (주관)

예상 소요 시간: 3~4시간
"""
```

**최종 제출 조건:**
- [ ] `pytest test_pipeline.py` 통과 (3개 테스트 이상)
- [ ] Recall@5: 하이브리드+Reranker ≥ 0.80
- [ ] `README.md`에 실행 방법 + 성능 비교표 포함

---

### 과제 5: 가중치 자동 튜닝 (심화)

```python
# task5_auto_tuning.py
"""
실습 과제 5: EnsembleRetriever 가중치 자동 튜닝

목표:
  Grid Search로 최적 vector:BM25 가중치 탐색
  가중치 조합: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] × 역수
  평가 지표: Recall@5

출력:
  가중치별 성능 히트맵 (print 또는 matplotlib)
  최적 가중치 및 해당 Recall@5

힌트:
  evaluate_hybrid_weight() 함수 활용
  각 가중치 조합마다 새로운 Chroma 컬렉션 생성 필요
  (같은 컬렉션에 여러 번 from_documents 호출 시 중복)
"""
```

---

## 부록 A: 트러블슈팅

### A.1 임베딩 차원 불일치

```python
# 증상: chromadb.errors.InvalidDimensionException
# 원인: 이전 컬렉션과 현재 임베딩 모델 차원이 다름

# 해결 1: 컬렉션 삭제 후 재생성
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection("my_collection")

# 해결 2: 차원 확인
embeddings = get_embeddings()
dim = len(embeddings.embed_query("test"))
print(f"현재 임베딩 차원: {dim}")

# 해결 3: 다른 컬렉션 이름 사용
collection = client.get_or_create_collection("new_collection_768d")
```

### A.2 Ollama 연결 오류

```bash
# 증상: requests.exceptions.ConnectionError
# 해결: Ollama 서버 시작
ollama serve

# 모델 확인
ollama list

# nomic-embed-text 없을 때
ollama pull nomic-embed-text
```

### A.3 BGE Reranker 로드 실패

```python
# 증상: OSError: We couldn't connect to 'https://huggingface.co'
# 해결: local_files_only=True 확인

from huggingface_hub import snapshot_download
model_path = snapshot_download(
    "BAAI/bge-reranker-v2-m3",
    cache_dir="./hf_cache",
    local_files_only=True,  # ← 이 옵션이 없으면 네트워크 시도
)

# 캐시 경로 확인
import os
cache_dir = "./hf_cache"
models = os.listdir(cache_dir)
print(f"캐시된 모델: {models}")
```

### A.4 BM25 한국어 처리 품질 개선

```python
# 기본 공백 분리 → 형태소 분석기 적용
# konlpy 설치: pip install konlpy
# Java 필요: sudo apt-get install default-jdk

from konlpy.tag import Okt

okt = Okt()

def korean_tokenize(text: str):
    """
    형태소 분석 기반 토크나이징
    "검색 증강 생성" → ["검색", "증강", "생성"]
    조사/어미 제거로 BM25 품질 향상
    """
    # 명사, 동사, 형용사만 추출
    morphs = okt.pos(text, norm=True, stem=True)
    tokens = [
        word for word, pos in morphs
        if pos in ["Noun", "Verb", "Adjective"] and len(word) > 1
    ]
    return tokens

# BM25Retriever에 적용
bm25_retriever = BM25Retriever.from_documents(
    docs,
    preprocess_func=korean_tokenize,
)
```

### A.5 Chroma 성능 최적화

```python
# 대용량 문서 배치 삽입
def batch_add_documents(
    collection,
    docs: List[Document],
    embeddings,
    batch_size: int = 100,
):
    """
    대용량 문서 배치 삽입
    한 번에 너무 많이 삽입하면 메모리 부족 가능
    """
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i:i+batch_size]
        texts = [d.page_content for d in batch]
        metas = [d.metadata for d in batch]
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        vectors = embeddings.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metas,
        )
        print(f"진행: {min(i+batch_size, total)}/{total}")
```

---

## 부록 B: 성능 벤치마크 참고값

| 구성 | Recall@5 (참고) | 비고 |
|------|----------------|------|
| 순수 벡터 (Chroma) | 0.65~0.75 | 데이터/쿼리에 따라 다름 |
| BM25만 | 0.55~0.70 | 키워드 매칭 강도에 따라 |
| 하이브리드 (RRF 6:4) | 0.75~0.85 | 일반적으로 단독 대비 +10% |
| 하이브리드 + Reranker | 0.80~0.90 | Reranker 추가 시 +5~10% |

> 위 수치는 참고값이며, 실제 데이터와 도메인에 따라 크게 달라진다.
> 직접 실험하여 자신의 데이터에 맞는 수치를 확인해야 한다.

---

*작성일: 2026-05-21*
*환경: Python 3.10+, Chroma 0.5+, LangChain 0.2+, Ollama (nomic-embed-text)*
