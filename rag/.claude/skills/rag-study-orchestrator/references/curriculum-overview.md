# RAG 학습 커리큘럼 전체 개요

## 목차
- [학습 목표](#학습-목표)
- [커리큘럼 구조](#커리큘럼-구조)
- [역할별 학습 포인트](#역할별-학습-포인트)
- [기술 스택 전체 맵](#기술-스택-전체-맵)
- [레벨별 핵심 산출물](#레벨별-핵심-산출물)
- [전제 지식 체크리스트](#전제-지식-체크리스트)

---

## 학습 목표

이 커리큘럼은 다음 세 가지 배경을 가진 학습자가 RAG 시스템을 **설계 → 구현 → 배포 → 운영**할 수 있도록 한다:

1. **Python 개발자**: LangChain/LlamaIndex로 RAG 파이프라인 구현
2. **AIOps (K8S+Kind)**: 프로덕션 수준 RAG 서비스 K8S 배포 및 운영
3. **DA (Chroma, Elasticsearch)**: 벡터/렉시컬 검색 최적화, 하이브리드 검색 설계

---

## 커리큘럼 구조

```
Level 1 (2주)          Level 2 (3주)          Level 3 (3주)          Level 4 (4주)
기초 RAG              향상된 RAG             고급 RAG              프로덕션 K8S
────────────          ──────────────         ────────────          ────────────
• RAG 개념            • 청킹 전략             • Self-RAG            • Kind 클러스터
• LangChain 기초      • Elasticsearch        • CRAG                • Chroma K8S
• Chroma 기본         • BM25 검색            • HyDE                • ES ECK
• 간단한 Q&A 봇       • 하이브리드 검색       • Multi-hop RAG       • 모니터링
                      • Reranking            • RAGAS 평가          • HPA 스케일링
```

**총 12주 과정**

---

## 역할별 학습 포인트

### Python 개발자 트랙

| 레벨 | 핵심 기술 | 배우는 이유 |
|------|---------|-----------|
| L1 | LangChain Chain, DocumentLoader, TextSplitter | RAG 파이프라인의 기본 빌딩 블록 |
| L2 | EnsembleRetriever, BM25Retriever | 검색 품질 개선 |
| L3 | LangGraph, Self-RAG 패턴 | 복잡한 에이전트 흐름 구현 |
| L4 | FastAPI + RAG 서비스화, 환경변수 관리 | 프로덕션 API 서빙 |

### AIOps (K8S+Kind) 트랙

| 레벨 | 핵심 기술 | 배우는 이유 |
|------|---------|-----------|
| L1 | Docker로 RAG 앱 실행 | 컨테이너 기본 |
| L2 | Docker Compose로 ES + RAG 구성 | 멀티 서비스 관리 |
| L3 | - (주로 개발자/DA 심화) | - |
| L4 | Kind, StatefulSet, ECK, HPA, Prometheus | 실제 K8S 운영 |

### DA (Chroma, Elasticsearch) 트랙

| 레벨 | 핵심 기술 | 배우는 이유 |
|------|---------|-----------|
| L1 | Chroma 컬렉션 설계, 메타데이터 스키마 | 임베딩 저장소 기본 |
| L2 | ES 인덱스 매핑, Nori 분석기, BM25 | 한국어 렉시컬 검색 |
| L3 | 청킹 최적화, 부모-자식 청크 | 검색 품질 극대화 |
| L4 | Chroma StatefulSet, ES 샤딩/복제 | 데이터 영속성 및 스케일 |

---

## 기술 스택 전체 맵

> **폐쇄망 환경**: `pip install`은 가능. HuggingFace 모델 가중치·Docker 이미지는 외부 PC에서 다운로드 후 메일 전달 필요. 커리큘럼 작성 시 해당 항목에 오프라인 설치 절차를 반드시 포함하라.

```
RAG 애플리케이션 레이어
├── Framework:     LangChain, LlamaIndex
├── LLM:          사내 LLM (REST API, requests 직접 호출) ✓ | HuggingFace 로컬 [오프라인 다운로드]
├── Embedding:    사내 임베딩 (REST API, requests 직접 호출) ✓ | BGE [오프라인 다운로드]
└── Evaluation:   RAGAS (pip) ✓ | LangSmith (API) ✓

검색/저장 레이어
├── 벡터 검색:    Chroma DB (pip) ✓ | FAISS (pip) ✓
├── 렉시컬 검색:  Elasticsearch [Docker 이미지 오프라인 필요]
└── 하이브리드:   EnsembleRetriever (LangChain) ✓ | ES kNN+BM25

인프라 레이어 (Level 4)                    ← 모두 Docker 이미지 오프라인 필요
├── 컨테이너:     Docker [이미지 docker save → 메일 전달]
├── 오케스트레이션: Kubernetes / Kind [이미지 포함]
├── 패키지 관리:  Helm [오프라인 저장소 설정 필요]
├── 모니터링:    Prometheus + Grafana [이미지 오프라인]
└── MLOps:       MLflow (선택) [이미지 오프라인]
```

### 오프라인 전달 대상 목록

| 항목 | 종류 | 외부 다운로드 방법 |
|------|------|----------------|
| `BAAI/bge-reranker-v2-m3` | HuggingFace 모델 | `huggingface-cli download BAAI/bge-reranker-v2-m3` |
| `BAAI/bge-m3` | HuggingFace 모델 | `huggingface-cli download BAAI/bge-m3` |
| Elasticsearch 8.x | Docker 이미지 | `docker pull elasticsearch:8.15.0 && docker save -o es.tar elasticsearch:8.15.0` |
| Kibana 8.x | Docker 이미지 | 동일 방식 |
| Prometheus | Docker 이미지 | `docker pull prom/prometheus && docker save -o prometheus.tar prom/prometheus` |
| Grafana | Docker 이미지 | 동일 방식 |
| Kind node image | Docker 이미지 | `docker pull kindest/node:v1.31.0 && docker save ...` |

---

## 레벨별 핵심 산출물

### Level 1 산출물
- [ ] `01_basic_rag.py` — PDF 기반 Q&A 봇 (30줄)
- [ ] `chroma_db/` — 영속적 Chroma 컬렉션

### Level 2 산출물
- [ ] `02_hybrid_search.py` — Chroma + BM25 하이브리드
- [ ] `03_es_korean_rag.py` — Elasticsearch Nori + RAG
- [ ] Recall@5 측정 결과 비교표

### Level 3 산출물
- [ ] `04_self_rag.py` — LangGraph 기반 Self-RAG
- [ ] `05_ragas_evaluation.py` — 자동화 평가 파이프라인
- [ ] RAGAS 지표 개선 보고서

### Level 4 산출물
- [ ] `kind-rag-cluster.yaml` — Kind 클러스터 설정
- [ ] `k8s/` — 전체 K8S 매니페스트 디렉토리
- [ ] Grafana 대시보드 (RAG 레이턴시, 처리량)

---

## 전제 지식 체크리스트

RAG 학습 시작 전 확인:

**Python 기본 (필수)**
- [ ] async/await 비동기 패턴 이해
- [ ] pip 가상환경 (.venv) 설정
- [ ] `.env` 파일로 환경변수 관리

**K8S 기본 (Level 4 진입 전)**
- [ ] `kubectl get/describe/logs` 명령어
- [ ] Deployment, Service, ConfigMap, Secret 개념
- [ ] Kind 설치 및 클러스터 생성 경험

**DB 기본 (Level 2 진입 전)**
- [ ] Elasticsearch REST API 기본 (`_search`, `_index`)
- [ ] JSON 문서 구조와 매핑 개념
- [ ] 인덱스 vs 컬렉션 차이 이해
