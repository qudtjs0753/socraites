---
name: rag-curriculum
description: "RAG 학습 커리큘럼을 설계하고 학습 로드맵을 생성하는 스킬. Python 개발자 / AIOps / DA 역할별 맞춤 커리큘럼 요청, 학습 계획 수립, 레벨 평가, 다음 단계 추천, 주차별 학습 목표 설정 시 이 스킬을 사용하라. '내 수준 평가해줘', '어디서 시작해야 해', '12주 커리큘럼', 'RAG 공부 계획' 요청 시에도 반드시 이 스킬을 사용."
---

# RAG 커리큘럼 스킬

Python 개발자 / AIOps / DA 배경을 가진 학습자를 위한 RAG 학습 커리큘럼을 설계하는 스킬.

## 커리큘럼 설계 원칙

사용자 배경에서 RAG로의 연결 지점:
- **Python 개발자** → LangChain/LlamaIndex API 사용 → RAG 파이프라인 구현
- **K8S AIOps** → Docker → RAG 서비스 컨테이너화 → K8S 배포
- **DA(Chroma, ES)** → 임베딩 이해 → 벡터 인덱스 설계 → 하이브리드 검색

## 4단계 커리큘럼 개요

### Level 1: 기초 RAG (2주)

**목표:** 동작하는 최소 RAG 파이프라인 구축

| 주차 | 주제 | 역할별 포인트 |
|------|------|-------------|
| 1주 | RAG 개념 + LangChain 기초 | 개발자: LangChain chain 패턴; DA: 청킹이란? |
| 2주 | Chroma DB + 기본 RAG 완성 | 개발자: RAG 코드 완성; DA: Chroma 컬렉션 설계 |

**핵심 실습:** PDF 기반 Q&A 봇 (30줄 Python)

### Level 2: 향상된 RAG (3주)

**목표:** 검색 품질 개선, Elasticsearch 통합, 하이브리드 검색

| 주차 | 주제 | 역할별 포인트 |
|------|------|-------------|
| 3주 | 청킹 전략 최적화 | 개발자: TextSplitter 비교; DA: 청크 크기 실험 |
| 4주 | Elasticsearch + BM25 | DA: ES 인덱스 설계, Nori 형태소 분석기 |
| 5주 | 하이브리드 검색 + Reranking | 개발자: EnsembleRetriever; DA: RRF 가중치 |

**핵심 실습:** 한국어 문서 검색 시스템 (Chroma + ES 하이브리드)

### Level 3: 고급 RAG (3주)

**목표:** Self-RAG, 평가 프레임워크, 복잡한 아키텍처

| 주차 | 주제 | 역할별 포인트 |
|------|------|-------------|
| 6주 | Self-RAG / CRAG | 개발자: LangGraph 활용 |
| 7주 | HyDE + Multi-hop RAG | 개발자: 복합 쿼리 처리 |
| 8주 | RAGAS 평가 | 개발자: 자동화 평가 파이프라인 |

**핵심 실습:** RAGAS로 측정한 RAG 품질 개선 실험

### Level 4: 프로덕션 K8S (4주)

**목표:** Kind에서 프로덕션 수준 RAG 서비스 배포

| 주차 | 주제 | 역할별 포인트 |
|------|------|-------------|
| 9주 | Kind 클러스터 + Docker RAG | AIOps: Kind 설정, Dockerfile |
| 10주 | Chroma K8S 배포 | AIOps: StatefulSet; DA: 영속성 설정 |
| 11주 | Elasticsearch(ECK) 배포 | AIOps: Helm, ECK 오퍼레이터 |
| 12주 | 모니터링 + 스케일링 | AIOps: Prometheus, HPA |

**핵심 실습:** Kind 클러스터에서 완전한 RAG 서비스 운영

## 수준 평가 질문

커리큘럼 시작 전 사용자 수준 확인:

1. LangChain/LlamaIndex 사용 경험이 있나요?
2. 임베딩(Embedding)이 무엇인지 설명할 수 있나요?
3. Chroma에 데이터를 저장해본 경험이 있나요?
4. 주당 학습에 투자할 수 있는 시간은?

**시작 레벨 결정 기준:**
- 1번 No → Level 1부터
- 1번 Yes + 2번 Yes + 3번 No → Level 2부터
- 모두 Yes → Level 3부터

## 산출물 형식

커리큘럼 문서는 다음 구조로 작성:
```markdown
# RAG 학습 커리큘럼 — {사용자_레벨}부터 시작

## 전체 로드맵
[레벨별 목표와 기간 표]

## {현재_레벨} 상세 커리큘럼
### {주차} 학습 목표
- 개발자 포인트: ...
- AIOps 포인트: ...
- DA 포인트: ...

### 실습 프로젝트
[구체적인 만들어볼 것]

### 체크리스트
- [ ] 완료 기준 1
- [ ] 완료 기준 2
```

## 관련 references

- 레벨별 상세 내용: `../rag-study-orchestrator/references/` 참조
