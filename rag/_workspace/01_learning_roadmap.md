# RAG 학습 로드맵 — Ollama + Chroma 중심 (4레벨, 12주)

> 작성일: 2026-05-21  
> 환경: 로컬 전용 (Ollama LLM + HuggingFace 임베딩 + Chroma)  
> 외부 API 사용 불가 — 모든 모델은 로컬에서 실행

---

## 전체 구조

```
Level 1 (2주)          Level 2 (3주)          Level 3 (3주)          Level 4 (4주)
기초 RAG              향상된 RAG             고급 RAG              프로덕션 K8S
────────────          ──────────────         ────────────          ────────────
• RAG 개념             • 청킹 전략             • Self-RAG            • Kind 클러스터
• LangChain 기초       • BM25 검색            • CRAG                • Chroma K8S
• Chroma 기본          • 하이브리드 검색        • HyDE                • ES ECK
• Ollama 설정          • Reranking            • Multi-hop RAG       • 모니터링
• 다중 파일 Q&A 봇      • Recall@5 측정         • RAGAS 평가           • HPA 스케일링
```

---

## Ollama 환경 설정 (Level 1 진입 전 필수)

### Ollama 설치

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 서버 실행 확인
ollama serve          # 백그라운드 실행 중이면 생략
curl http://localhost:11434/api/tags   # 설치 확인
```

### 권장 모델 선택 가이드

| 모델 | 크기 | 용도 | 설치 명령 |
|------|------|------|----------|
| `llama3.2:3b` | 2GB | 개발/테스트 (빠름) | `ollama pull llama3.2:3b` |
| `llama3.1:8b` | 4.7GB | 일반 학습 (균형) | `ollama pull llama3.1:8b` |
| `llama3.1:70b` | 40GB | 고품질 (GPU 권장) | `ollama pull llama3.1:70b` |
| `mistral:7b` | 4.1GB | 영어 문서 특화 | `ollama pull mistral:7b` |
| `qwen2.5:7b` | 4.4GB | 한국어 성능 우수 | `ollama pull qwen2.5:7b` |
| `nomic-embed-text` | 274MB | 임베딩 전용 | `ollama pull nomic-embed-text` |

> **권장 시작 조합**: LLM은 `qwen2.5:7b` (한국어 문서), 임베딩은 `nomic-embed-text`

### LangChain에서 Ollama 연결

```python
# pip install langchain-ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="qwen2.5:7b", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
```

### HuggingFace 로컬 임베딩 (오프라인 대안)

```python
# pip install sentence-transformers langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    cache_folder="./hf_cache"   # 오프라인 캐시 경로
)
```

> **오프라인 환경**: 외부 PC에서 `snapshot_download("BAAI/bge-m3", cache_dir="./hf_cache")` 실행 후 `hf_cache/` 폴더를 통째로 복사

---

## Level 1 — 기초 RAG (2주)

### 이론 요약

**RAG(Retrieval Augmented Generation)란?**

LLM이 학습하지 않은 문서에 대해 답변할 수 있도록, 질문과 관련된 문서를 먼저 검색한 뒤 해당 내용을 컨텍스트로 붙여 LLM에게 전달하는 방식이다.

```
사용자 질문
    │
    ▼
임베딩(벡터화) → 벡터DB 검색 → 관련 문서 k개 추출
                                        │
                                        ▼
                              LLM에 [문서 + 질문] 전달 → 답변 생성
```

**핵심 개념**

| 개념 | 설명 |
|------|------|
| 임베딩 | 텍스트를 숫자 벡터로 변환. 의미가 비슷한 문장은 벡터 공간에서 가깝게 위치 |
| 코사인 유사도 | 두 벡터 사이의 각도로 유사도를 측정. 1에 가까울수록 유사 |
| 청킹(Chunking) | 긴 문서를 LLM 컨텍스트 창에 맞게 작은 조각으로 분할하는 과정 |
| 벡터DB | 임베딩 벡터를 저장하고 유사도 검색(ANN)을 제공하는 데이터베이스 |
| Chroma | 로컬 파일 기반 벡터DB. pip 설치만으로 사용 가능 |
| DocumentLoader | LangChain에서 파일(CSV, TXT, MD 등)을 Document 객체로 변환하는 컴포넌트 |
| TextSplitter | Document를 청크로 분할하는 LangChain 컴포넌트 |
| RetrievalChain | 검색 → LLM 답변 생성을 하나의 체인으로 연결 |

**Chroma 내부 구조**

Chroma는 각 컬렉션에 벡터, 원본 텍스트, 메타데이터를 함께 저장한다. 기본 인덱스는 HNSW(Hierarchical Navigable Small World)로 근사 최근접 이웃 탐색을 수행한다. `persist_directory`를 지정하면 SQLite + 파일 형태로 디스크에 저장된다.

### 핵심 목표

- Ollama로 로컬 LLM을 실행하고 LangChain과 연결한다
- CSV/TXT/MD/LOG/이미지 파일을 로드하여 Chroma에 인덱싱한다
- 기본 Q&A RAG 파이프라인을 구현한다

### 실습 목표

1. Ollama 설치 및 `qwen2.5:7b`, `nomic-embed-text` 모델 실행
2. LangChain DocumentLoader로 다중 파일 형식 로드
3. `RecursiveCharacterTextSplitter`로 청킹
4. Chroma 영속 컬렉션 생성 및 문서 임베딩 저장
5. `RetrievalQA` 체인으로 Q&A 봇 완성

### 체크리스트

- [ ] `ollama pull qwen2.5:7b && ollama pull nomic-embed-text` 성공
- [ ] `ChatOllama`로 간단한 질문 응답 확인
- [ ] CSV 파일 → Document 변환 → Chroma 저장
- [ ] TXT/MD/LOG 파일 로더 각각 구현
- [ ] 이미지 파일 설명 추출 (Ollama vision 모델 또는 텍스트 대체 처리)
- [ ] `persist_directory`로 Chroma 영속화 확인 (재시작 후 데이터 유지)
- [ ] Q&A 봇에서 관련 문서가 컨텍스트로 포함되는지 확인 (`return_source_documents=True`)

### 산출물

- `rag_pipeline.py` — 다중 파일 Q&A 봇
- `chroma_db/` — 영속적 Chroma 컬렉션

---

## Level 2 — 향상된 RAG (3주)

### 이론 요약

**왜 기본 RAG가 부족한가?**

기본 RAG는 임베딩 유사도만으로 검색하므로 정확한 키워드가 포함된 문서를 놓칠 수 있다(어휘 불일치 문제). 또한 청크 크기가 고정되어 있으면 문맥이 잘리거나 노이즈가 포함된다.

**청킹 전략**

| 전략 | 특징 | 적합한 경우 |
|------|------|-----------|
| Fixed-size | 고정 크기(예: 512 토큰)로 분할 | 단순 구현, 기본값 |
| Recursive | 단락 → 문장 → 단어 순서로 계층적 분할 | 일반 텍스트 (권장) |
| Semantic | 임베딩 유사도 기반 의미 단위 분할 | 주제 전환이 명확한 문서 |
| Parent-Child | 큰 청크(부모)를 검색하고 작은 청크(자식)를 반환 | 컨텍스트 보존 + 정밀도 |

**BM25 (Best Match 25)**

TF-IDF를 개선한 렉시컬(키워드) 검색 알고리즘이다. 문서 길이 정규화를 포함하여 단어 빈도와 역문서 빈도를 결합해 점수를 산출한다. 임베딩이 포착하지 못하는 고유명사, 코드, 전문용어 검색에 강하다.

**하이브리드 검색 (Hybrid Search)**

벡터 검색(의미 유사도)과 BM25(키워드 매칭)의 결과를 결합한다. LangChain의 `EnsembleRetriever`는 두 검색 결과를 RRF(Reciprocal Rank Fusion)로 합산한다.

```
RRF 점수 = Σ 1 / (k + rank_i)   (k=60 기본값)
```

**Reranking**

1차 검색(빠른 ANN/BM25)으로 후보 20~50개를 추린 뒤, Cross-Encoder 모델로 정밀하게 재순위를 매기는 2단계 검색이다. Cross-Encoder는 질문과 문서를 함께 입력받아 관련성 점수를 직접 계산한다(Bi-Encoder보다 느리지만 정확).

**Recall@k 측정**

검색 품질 평가 지표. 정답 문서가 상위 k개 안에 포함되는 비율이다.

```
Recall@5 = (상위 5개 중 정답 문서 수) / (전체 정답 문서 수)
```

### 핵심 목표

- 청킹 전략을 실험하고 최적 파라미터를 찾는다
- BM25 + 벡터 하이브리드 검색을 구현한다
- Cross-Encoder Reranking으로 검색 품질을 높인다
- Recall@5를 측정하여 전략 간 성능을 비교한다

### 실습 목표

1. `RecursiveCharacterTextSplitter` 파라미터(chunk_size, overlap) 실험
2. `BM25Retriever`와 `Chroma` 검색기를 `EnsembleRetriever`로 결합
3. `BAAI/bge-reranker-v2-m3`로 Cross-Encoder Reranking 구현
4. (선택 심화) Elasticsearch 설치 → Nori 형태소 분석기 설정 → 한국어 BM25
5. Recall@5 측정 스크립트 작성 및 전략별 비교표 생성

### 체크리스트

- [ ] chunk_size 256/512/1024 실험, 청크 수 및 검색 품질 비교
- [ ] `EnsembleRetriever(retrievers=[bm25, chroma], weights=[0.4, 0.6])` 동작 확인
- [ ] Reranker 모델 로컬 로드 (`./hf_cache` 경로 사용)
- [ ] Recall@5 측정 결과 비교표 작성 (Chroma only vs Hybrid vs Hybrid+Rerank)
- [ ] (선택) ES Docker 이미지 오프라인 로드 후 Nori 분석기 테스트

### 산출물

- `hybrid_search.py` — Chroma + BM25 하이브리드 + Reranking
- `recall_benchmark.py` — Recall@5 측정 스크립트
- Recall@5 비교표 (전략별 성능)
- (선택) `es_korean_rag.py` — Elasticsearch Nori + RAG

---

## Level 3 — 고급 RAG (3주)

### 이론 요약

**Self-RAG**

LLM이 스스로 "지금 검색이 필요한가?", "검색 결과가 관련 있는가?", "답변이 사실에 근거하는가?"를 판단하여 검색 여부와 답변 방식을 제어하는 패턴이다. 특수 토큰(`[Retrieve]`, `[Relevant]`, `[Supported]`)을 통해 생성 과정에 검색을 통합한다.

```
질문 입력
    │
    ▼
검색 필요 여부 판단 (LLM)
    ├─ 불필요 → 직접 답변 생성
    └─ 필요 → 벡터DB 검색
              │
              ▼
         관련성 판단 (LLM)
              ├─ 무관 → 재검색 또는 직접 답변
              └─ 관련 → 답변 생성 → 사실 근거 판단 → 최종 답변
```

**CRAG (Corrective RAG)**

검색된 문서의 품질을 평가하고 신뢰도가 낮으면 웹 검색(로컬 환경에서는 다른 문서 컬렉션)으로 보완하는 패턴이다. 문서 품질 평가에 LLM 또는 Cross-Encoder를 사용한다.

```
검색 결과 품질 평가
    ├─ High (신뢰): 그대로 사용
    ├─ Low (불신): 대안 소스에서 재검색
    └─ Ambiguous: 원본 + 대안 혼합 사용
```

**HyDE (Hypothetical Document Embeddings)**

질문을 직접 임베딩하는 대신, LLM이 "질문에 대한 가상의 답변 문서"를 생성하고 그 문서를 임베딩하여 검색하는 방법이다. 질문과 답변 문서의 임베딩 공간 불일치 문제를 해결한다.

```
원래 방식: 질문 임베딩 → 검색
HyDE 방식: 질문 → LLM → 가상 답변 문서 → 임베딩 → 검색 (더 정확)
```

**Multi-hop RAG**

단일 검색으로 답하기 어려운 복잡한 질문을 여러 단계의 검색으로 분해하여 답하는 방식이다. "A의 CEO는 누구이고 그 사람의 출신 대학은?"처럼 연쇄적 추론이 필요한 질문에 적합하다.

**RAGAS (RAG Assessment)**

RAG 파이프라인의 품질을 자동으로 평가하는 프레임워크다. LLM을 Judge로 사용한다.

| 지표 | 설명 | 범위 |
|------|------|------|
| Faithfulness | 답변이 검색 문서에 근거하는 정도 | 0~1 |
| Answer Relevancy | 답변이 질문에 얼마나 관련 있는가 | 0~1 |
| Context Precision | 검색된 문서 중 실제 유용한 비율 | 0~1 |
| Context Recall | 정답에 필요한 정보가 검색됐는가 | 0~1 |

**LangGraph**

LangChain 위에서 복잡한 에이전트 워크플로우(분기, 루프, 조건)를 그래프로 정의하는 프레임워크다. Self-RAG, CRAG 같은 조건부 흐름 구현에 적합하다.

### 핵심 목표

- LangGraph로 Self-RAG, CRAG 패턴을 구현한다
- HyDE로 검색 품질을 향상시킨다
- RAGAS로 파이프라인을 자동 평가하고 지표 개선 사이클을 돌린다

### 실습 목표

1. LangGraph StateGraph로 Self-RAG 흐름 구현 (검색 필요 여부 판단 노드 포함)
2. CRAG: 문서 품질 평가 → 신뢰도 낮으면 대안 컬렉션 검색
3. HyDE: 가상 답변 생성 → 임베딩 검색 → 기존 방식과 Recall 비교
4. Multi-hop: 질문 분해 → 순차 검색 → 답변 통합
5. RAGAS 평가 파이프라인: 테스트셋 생성 → 4가지 지표 자동 측정

### 체크리스트

- [ ] LangGraph 설치 및 기본 StateGraph 동작 확인
- [ ] Self-RAG: "검색 필요 여부" 판단 프롬프트 작성 및 분기 동작 확인
- [ ] CRAG: 문서 관련성 점수 임계값 설정 및 대안 검색 경로 구현
- [ ] HyDE: 가상 답변 생성 품질 확인, 기존 방식 대비 Recall@5 비교
- [ ] RAGAS: `from ragas import evaluate` 로 Faithfulness / Answer Relevancy 측정
- [ ] RAGAS Judge LLM을 Ollama로 설정 (`llm=ChatOllama(model="qwen2.5:7b")`)
- [ ] RAGAS 지표 기준선(baseline) 측정 → Level 2 파이프라인과 비교 보고서 작성

### 산출물

- `self_rag.py` — LangGraph 기반 Self-RAG
- `crag.py` — Corrective RAG 구현
- `hyde_search.py` — HyDE 검색
- `ragas_evaluation.py` — 자동화 평가 파이프라인
- RAGAS 지표 개선 보고서 (baseline → Level 3 적용 후 비교)

---

## Level 4 — 프로덕션 K8S (4주)

### 이론 요약

**왜 K8S에 배포하는가?**

로컬 Python 스크립트 수준을 넘어 여러 사용자가 동시에 사용할 수 있는 서비스로 만들기 위해 컨테이너 오케스트레이션이 필요하다. Kind(Kubernetes in Docker)는 로컬 머신에서 실제 K8S 클러스터를 시뮬레이션한다.

**RAG 서비스 아키텍처**

```
클라이언트
    │ HTTP
    ▼
FastAPI (RAG API 서버) ──── Ollama (LLM 서버)
    │                              │
    ▼                              │
Chroma (StatefulSet)      BGE Embedding 서버
    │
    ▼ (선택 심화)
Elasticsearch (ECK)
```

**StatefulSet vs Deployment**

Chroma, Elasticsearch처럼 데이터를 저장하는 컴포넌트는 StatefulSet으로 배포한다. StatefulSet은 안정적인 네트워크 ID, 순서 있는 배포/종료, PVC 자동 연결을 보장한다.

**HPA (Horizontal Pod Autoscaler)**

CPU 사용률이나 커스텀 메트릭(RAG 쿼리 처리량)을 기준으로 RAG API 서버의 Pod 수를 자동으로 늘리거나 줄인다.

**Prometheus + Grafana**

RAG API에 `/metrics` 엔드포인트를 노출(prometheus-client 라이브러리)하면 Prometheus가 수집하고 Grafana에서 시각화할 수 있다. 주요 지표: 요청 레이턴시, 처리량, 검색 소요 시간, LLM 응답 시간.

### 핵심 목표

- Level 3 RAG 파이프라인을 FastAPI 서비스로 패키징한다
- Kind 클러스터에 Chroma StatefulSet을 배포한다
- HPA와 Prometheus 모니터링을 설정한다

### 실습 목표

1. Level 3 RAG를 FastAPI 엔드포인트로 래핑 + Dockerfile 작성
2. Kind 클러스터 생성 (`kind create cluster --config kind-rag-cluster.yaml`)
3. Chroma StatefulSet + PVC 배포
4. Ollama를 K8S DaemonSet 또는 Deployment로 배포
5. HPA 설정: CPU 70% 기준 RAG API 1~5개 Pod 자동 스케일링
6. Prometheus + Grafana Helm 설치 + RAG 레이턴시 대시보드 구성

### 체크리스트

- [ ] `Dockerfile` 빌드 및 Kind 클러스터에 이미지 로드 (`kind load docker-image`)
- [ ] `kubectl get pods -n rag` 로 모든 Pod Running 확인
- [ ] Chroma PVC 마운트 확인 (Pod 재시작 후 데이터 유지)
- [ ] Ollama 모델 볼륨 마운트 설정 (모델 파일 영속화)
- [ ] `kubectl autoscale deployment rag-api --cpu-percent=70 --min=1 --max=5`
- [ ] Grafana에서 RAG 레이턴시 p50/p95/p99 확인
- [ ] 부하 테스트(`locust` 등)로 HPA 스케일아웃 동작 확인

### 산출물

- `Dockerfile` — RAG API 컨테이너 이미지
- `kind-rag-cluster.yaml` — Kind 클러스터 설정
- `k8s/` — 전체 K8S 매니페스트 (Deployment, StatefulSet, Service, HPA, PVC)
- Grafana 대시보드 JSON (RAG 레이턴시, 처리량)

---

## 전제 지식 체크리스트

### Python 기본 (Level 1 진입 전 필수)

- [ ] `pip install`, 가상환경 (`.venv`) 생성 및 활성화
- [ ] `.env` 파일로 환경변수 관리 (`python-dotenv`)
- [ ] `async/await` 비동기 패턴 기본 이해
- [ ] `requests` 또는 `httpx`로 REST API 호출

### DB 기본 (Level 2 진입 전)

- [ ] Elasticsearch REST API 기본 (`_search`, `_index`, `_mapping`)
- [ ] JSON 문서 구조와 인덱스 매핑 개념
- [ ] 인덱스 vs 컬렉션 차이 이해

### K8S 기본 (Level 4 진입 전)

- [ ] `kubectl get/describe/logs` 명령어
- [ ] Deployment, Service, ConfigMap, Secret, PVC 개념
- [ ] Kind 설치 및 클러스터 생성 경험 (`kind create cluster`)
- [ ] Helm 기본 사용법 (`helm install`, `helm upgrade`)

---

## 오프라인 설치 사전 준비 목록

외부 PC(인터넷 가능)에서 미리 다운로드 후 전달:

| 항목 | 종류 | 외부 PC 다운로드 명령 |
|------|------|---------------------|
| `BAAI/bge-m3` | HuggingFace 임베딩 모델 | `from huggingface_hub import snapshot_download; snapshot_download("BAAI/bge-m3", cache_dir="./hf_cache")` |
| `BAAI/bge-reranker-v2-m3` | HuggingFace Reranker 모델 | `snapshot_download("BAAI/bge-reranker-v2-m3", cache_dir="./hf_cache")` |
| Elasticsearch 8.x | Docker 이미지 | `docker pull elasticsearch:8.15.0 && docker save -o es.tar elasticsearch:8.15.0` |
| Prometheus | Docker 이미지 | `docker pull prom/prometheus && docker save -o prometheus.tar prom/prometheus` |
| Grafana | Docker 이미지 | `docker pull grafana/grafana && docker save -o grafana.tar grafana/grafana` |
| Kind node image | Docker 이미지 | `docker pull kindest/node:v1.31.0 && docker save -o kindest.tar kindest/node:v1.31.0` |

> Ollama 모델(`qwen2.5:7b`, `nomic-embed-text`)은 외부 PC에서 `ollama pull` 후 `~/.ollama/models/` 디렉토리를 압축하여 전달

---

## 레벨별 핵심 산출물 요약

| 레벨 | 파일 | 설명 |
|------|------|------|
| L1 | `rag_pipeline.py` | 다중 파일 형식 Q&A 봇 |
| L1 | `chroma_db/` | 영속적 Chroma 컬렉션 |
| L2 | `hybrid_search.py` | Chroma + BM25 + Reranking |
| L2 | `recall_benchmark.py` | Recall@5 측정 스크립트 |
| L3 | `self_rag.py` | LangGraph 기반 Self-RAG |
| L3 | `crag.py` | Corrective RAG |
| L3 | `hyde_search.py` | HyDE 검색 |
| L3 | `ragas_evaluation.py` | RAGAS 자동 평가 파이프라인 |
| L4 | `Dockerfile` | RAG API 컨테이너 |
| L4 | `kind-rag-cluster.yaml` | Kind 클러스터 설정 |
| L4 | `k8s/` | 전체 K8S 매니페스트 |
