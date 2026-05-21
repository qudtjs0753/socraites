# 학습 목표 입력

## 사용자 요청 (2026-05-21)

RAG 구축 학습 하네스 구성 요청.

### 핵심 요구사항
- 수준을 고려한 단계별 RAG 학습 계획 (기초 → 고급)
- 이론 구체적으로 포함
- 코드 작성 가이드 (직접 작성할 수 있도록)
- **모든 LLM 로컬 실행** — Ollama 사용 (OpenAI 호환 REST API)
- 기술 스택: Python + Chroma (벡터DB 중심)
- Level: 기초 RAG → 고급 RAG (Self-RAG, CRAG, HyDE, RAGAS)

### 환경 제약
- 로컬 LLM: Ollama (http://localhost:11434/v1)
- 임베딩: Ollama 임베딩 모델 또는 HuggingFace 로컬 (BGE 등)
- 외부 API 사용 불가
- pip install 가능

### 기술 스택 선택
- Vector DB: Chroma (메인)
- Framework: LangChain
- LLM: Ollama (로컬)
- Evaluation: RAGAS
