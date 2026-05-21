# Level 1: 기초 RAG 파이프라인 — Python 구현 가이드

> **환경**: Python 3.10+, LangChain, Chroma, Ollama 로컬 LLM  
> **기간**: 2주  
> **목표**: CSV/Excel/TXT/MD/LOG/이미지 → 청킹 → Chroma → 검색 → 생성 파이프라인 직접 구현

---

## 목차

1. [RAG가 왜 필요한가? — 이론](#1-rag가-왜-필요한가--이론)
2. [임베딩(Embedding) 심층 이해](#2-임베딩embedding-심층-이해)
3. [환경 설정](#3-환경-설정)
4. [Ollama LLM / Embedding 클래스 구현](#4-ollama-llm--embedding-클래스-구현)
5. [Document Loader — 파일 읽기](#5-document-loader--파일-읽기)
6. [Text Splitter — 청킹 전략](#6-text-splitter--청킹-전략)
7. [Chroma 벡터 DB 구축](#7-chroma-벡터-db-구축)
8. [RAG 파이프라인 완성](#8-rag-파이프라인-완성)
9. [프롬프트 엔지니어링](#9-프롬프트-엔지니어링)
10. [전체 실습 프로젝트: 다중 파일 Q&A 봇](#10-전체-실습-프로젝트-다중-파일-qa-봇)
11. [테스트 작성](#11-테스트-작성)
12. [체크리스트 및 다음 단계](#12-체크리스트-및-다음-단계)

---

## 1. RAG가 왜 필요한가? — 이론

### 1.1 LLM의 근본적 한계

LLM(Large Language Model)은 학습 데이터의 스냅샷에 불과하다. GPT-4, LLaMA, Qwen 등 어떤 모델이든 학습 컷오프(cutoff) 이후의 정보는 알지 못한다. 더 심각한 문제는 **사내 정보**다. 사내 운영 로그, 장애 보고서, 사용자 매뉴얼은 어떤 공개 LLM도 학습한 적이 없다.

해결책을 나열하면:

| 방법 | 비용 | 속도 | 최신성 | 사내 정보 |
|------|------|------|--------|----------|
| Fine-tuning | 매우 높음 | 매우 느림 | 업데이트 필요 | 가능 |
| In-context Learning | 낮음 | 빠름 | 실시간 | 컨텍스트 길이 제한 |
| **RAG** | **낮음** | **빠름** | **실시간** | **무제한** |

RAG(Retrieval-Augmented Generation)는 LLM에게 "참고 문서"를 실시간으로 제공하는 방법이다. 모델을 재학습할 필요 없이, 질문에 관련된 문서 청크를 검색해서 프롬프트에 포함시킨다.

### 1.2 RAG 동작 원리 — 2단계 파이프라인

```
╔══════════════════════════════════════════════════════════════╗
║  오프라인 단계 — Indexing (문서 전처리)                        ║
║                                                              ║
║  원본 문서                                                    ║
║  (CSV, Excel, TXT, MD, LOG, 이미지)                          ║
║       │                                                      ║
║       ▼                                                      ║
║  청킹(Chunking)  ─── 문서를 500자 단위로 분할               ║
║       │                                                      ║
║       ▼                                                      ║
║  임베딩(Embedding)  ─── 각 청크를 숫자 벡터로 변환           ║
║  "RAG는 검색 기반" → [0.12, -0.34, 0.56, ...]               ║
║       │                                                      ║
║       ▼                                                      ║
║  벡터 DB(Chroma)  ─── 벡터를 저장 + 인덱싱                  ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  온라인 단계 — Retrieval + Generation (사용자 질문 처리)       ║
║                                                              ║
║  사용자 질문: "RAG의 핵심 구성요소는?"                         ║
║       │                                                      ║
║       ▼                                                      ║
║  질문 임베딩  ─── 질문도 벡터로 변환                         ║
║  "RAG의 핵심..." → [0.11, -0.32, 0.54, ...]                 ║
║       │                                                      ║
║       ▼                                                      ║
║  벡터 유사도 검색  ─── 코사인 유사도로 k개 청크 검색           ║
║       │                                                      ║
║       ▼                                                      ║
║  LLM 생성  ─── "질문 + 검색된 청크"를 프롬프트에 포함         ║
║       │                                                      ║
║       ▼                                                      ║
║  최종 답변 반환                                               ║
╚══════════════════════════════════════════════════════════════╝
```

### 1.3 RAG를 구성하는 핵심 요소

1. **Document Loader**: 파일을 읽어 `Document` 객체로 변환
2. **Text Splitter**: 긴 문서를 검색 가능한 크기로 분할 (청킹)
3. **Embedding Model**: 텍스트를 벡터로 변환 (Ollama `nomic-embed-text`)
4. **Vector Store**: 벡터를 저장하고 유사도 검색 (Chroma)
5. **LLM**: 검색된 문서를 바탕으로 답변 생성 (Ollama `llama3.2:3b`)
6. **Chain**: 위 컴포넌트를 연결하는 파이프라인 (LangChain)

---

## 2. 임베딩(Embedding) 심층 이해

### 2.1 임베딩이란?

임베딩은 텍스트를 고차원 숫자 벡터로 변환하는 기술이다. 핵심은 **의미가 비슷한 텍스트는 벡터 공간에서 가깝다**는 것이다.

```
"RAG는 검색 증강 생성입니다" → [0.12, -0.34, 0.56, 0.08, ...]  (768차원)
"검색 기반 LLM 기법"         → [0.11, -0.33, 0.55, 0.07, ...]  (비슷한 벡터)
"오늘 점심은 비빔밥"          → [-0.45, 0.67, -0.23, 0.89, ...] (먼 벡터)
```

### 2.2 코사인 유사도(Cosine Similarity)

벡터 간 유사도를 측정할 때 코사인 유사도를 사용한다. 벡터의 방향이 같을수록 1에 가깝고, 반대 방향이면 -1에 가깝다.

```
                  A · B
cos(θ) = ─────────────────────
           ||A|| × ||B||

A · B = Σ(Aᵢ × Bᵢ)   (내적, dot product)
||A|| = √(Σ Aᵢ²)      (L2 노름, 벡터 길이)
```

- `cos(θ) = 1.0` → 완전히 같은 의미
- `cos(θ) = 0.0` → 관계없음
- `cos(θ) = -1.0` → 반대 의미

**왜 유클리드 거리가 아닌 코사인인가?**  
벡터의 크기(길이)가 다르더라도 방향이 같으면 같은 의미다. 긴 문장과 짧은 문장을 비교할 때 코사인이 더 안정적이다.

### 2.3 임베딩 실습 — 직접 확인해보기

```python
# 파일: embedding_playground.py
# 목적: 임베딩 벡터와 코사인 유사도를 직접 확인
# 실행: python embedding_playground.py

import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY  = os.getenv("EMBED_API_KEY", "ollama")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list[float]:
    """Ollama 임베딩 API 호출"""
    resp = requests.post(
        f"{EMBED_BASE_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": text},
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # 세 문장: 두 개는 비슷한 의미, 하나는 관계없음
    texts = [
        "RAG는 외부 문서를 활용하는 기법입니다",          # 기준 문장
        "검색 증강 생성은 LLM에 지식을 추가합니다",        # 비슷한 의미
        "오늘 날씨가 맑고 기온이 25도입니다",              # 관계없음
    ]

    print("임베딩 생성 중...")
    embeddings = [np.array(get_embedding(t)) for t in texts]

    print(f"\n벡터 차원: {len(embeddings[0])}")
    print(f"벡터 앞 5개 값: {embeddings[0][:5]}")

    print("\n=== 코사인 유사도 비교 ===")
    print(f"문장 1-2 유사도 (비슷한 의미): {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"문장 1-3 유사도 (다른 의미):   {cosine_similarity(embeddings[0], embeddings[2]):.4f}")

    # 기대 출력:
    # 벡터 차원: 768
    # 문장 1-2 유사도 (비슷한 의미): 0.8523
    # 문장 1-3 유사도 (다른 의미):   0.1842
```

**직접 작성해보세요 — 실습 과제 1:**

위 코드를 참고하여 아래 코드를 직접 작성해보세요.

```python
# 실습 1-1: 임베딩 차원 확인
# - "안녕하세요"의 임베딩 벡터를 출력하고 차원을 확인하라
# - 모델에 따라 차원이 다르다 (nomic-embed-text: 768차원)

# TODO: 여기에 코드를 작성하세요
def print_embedding_info(text: str):
    emb = get_embedding(text)
    # 1. 벡터 차원 출력
    # 2. 벡터의 앞 10개 값 출력
    # 3. 벡터의 최솟값, 최댓값, 평균값 출력
    pass

# 실습 1-2: 유사도 탐색
# - 다음 세 쌍의 유사도를 계산하고 결과를 비교하라
pairs = [
    ("파이썬은 프로그래밍 언어다", "Python is a programming language"),  # 한영 번역 쌍
    ("서버가 다운됐다", "서버 장애가 발생했다"),                          # 동의어 표현
    ("배가 고프다", "오늘 코드 리뷰가 있다"),                            # 무관한 문장
]
# TODO: 각 쌍의 코사인 유사도를 출력하라
```

---

## 3. 환경 설정

### 3.1 필수 패키지 설치

```bash
pip install langchain langchain-core langchain-community langchain-chroma chromadb \
            openpyxl pillow pytesseract python-dotenv requests numpy pytest
```

> **이미지 OCR 패키지 (선택, 이미지 파일 처리 시 필요):**
> ```bash
> # macOS
> brew install tesseract tesseract-lang
> # Ubuntu/Debian
> sudo apt-get install tesseract-ocr tesseract-ocr-kor
> ```

### 3.2 .env 파일 설정

```bash
# .env (프로젝트 루트에 생성)
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2:3b

EMBED_BASE_URL=http://localhost:11434/v1
EMBED_API_KEY=ollama
EMBED_MODEL=nomic-embed-text
```

### 3.3 Ollama 설치 및 모델 준비

Ollama는 로컬에서 LLM을 실행하는 런타임이다. 외부 인터넷이 차단된 환경에서도 모델 파일만 있으면 동작한다.

```bash
# Ollama 설치 (외부 PC에서 다운로드 후 설치)
# https://ollama.com/download

# 모델 다운로드 (외부 PC에서)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 모델 파일 위치: ~/.ollama/models/
# 이 디렉토리를 tar로 묶어 서버로 전달

# 서버에서 Ollama 실행
ollama serve  # 기본 포트: 11434

# 동작 확인
curl http://localhost:11434/v1/models
```

### 3.4 디렉토리 구조

```
level1/
├── .env                  # 환경변수
├── .env.example          # 환경변수 템플릿
├── requirements.txt      # 패키지 목록
├── ollama_llm.py         # Ollama LLM/Embedding 클래스
├── loaders.py            # Document Loader
├── rag_pipeline.py       # 메인 RAG 파이프라인
├── test_loaders.py       # pytest 테스트
├── data/                 # 문서 파일 (CSV, Excel, TXT, MD, LOG, 이미지)
└── chroma_db/            # Chroma 영속성 디렉토리 (자동 생성)
```

---

## 4. Ollama LLM / Embedding 클래스 구현

### 4.1 왜 커스텀 클래스가 필요한가?

LangChain은 OpenAI, Anthropic 등 주요 LLM 제공자의 클래스를 내장 지원한다. Ollama도 `langchain-ollama` 패키지가 있지만, OpenAI 호환 REST API를 직접 사용하는 방식이 더 유연하고 폐쇄망 환경에서 안정적이다.

Ollama는 OpenAI 호환 REST API를 제공하므로, `requests`로 직접 호출할 수 있다.

### 4.2 OllamaLLM 클래스 — LangChain BaseChatModel 상속

```python
# 파일: ollama_llm.py
# 목적: Ollama OpenAI 호환 API를 LangChain에서 사용하기 위한 래퍼 클래스
# 실행: python ollama_llm.py (단독 테스트)

import os
import requests
from typing import List, Optional, Any
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, BaseMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.embeddings import Embeddings


class OllamaLLM(BaseChatModel):
    """
    Ollama OpenAI 호환 REST API를 LangChain BaseChatModel로 래핑.

    BaseChatModel을 상속하면 LangChain의 모든 체인, 파이프라인,
    LCEL(LangChain Expression Language) 연산자(|)와 호환된다.
    """
    base_url: str = ""
    api_key: str = "ollama"
    model_name: str = ""
    temperature: float = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> ChatResult:
        """
        메시지 목록을 받아 LLM 응답을 반환한다.

        BaseChatModel의 __call__, invoke, stream 등은
        모두 내부적으로 _generate를 호출한다.
        """
        # LangChain 메시지 타입 → OpenAI role 문자열 변환
        role_map = {
            HumanMessage: "user",
            SystemMessage: "system",
            AIMessage: "assistant",
        }

        openai_messages = [
            {
                "role": role_map.get(type(m), "user"),
                "content": m.content,
            }
            for m in messages
        ]

        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120,
        )
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        """LangChain 내부 식별자"""
        return "ollama_llm"


class OllamaEmbeddings(Embeddings):
    """
    Ollama 임베딩 API를 LangChain Embeddings 인터페이스로 래핑.

    Chroma, FAISS 등 LangChain 벡터스토어에 바로 사용 가능.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "ollama"):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서를 한 번에 임베딩 (인덱싱 시 사용)"""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """쿼리 하나를 임베딩 (검색 시 사용)"""
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        """실제 API 호출"""
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": text},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def create_llm() -> OllamaLLM:
    """환경변수에서 LLM 인스턴스를 생성하는 팩토리 함수"""
    load_dotenv()
    return OllamaLLM(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        model_name=os.getenv("LLM_MODEL", "llama3.2:3b"),
        temperature=0,
    )


def create_embeddings() -> OllamaEmbeddings:
    """환경변수에서 임베딩 인스턴스를 생성하는 팩토리 함수"""
    load_dotenv()
    return OllamaEmbeddings(
        base_url=os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("EMBED_API_KEY", "ollama"),
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    )


# 단독 실행 테스트
if __name__ == "__main__":
    llm = create_llm()
    embeddings = create_embeddings()

    # LLM 테스트
    print("=== LLM 테스트 ===")
    response = llm.invoke("한 문장으로 RAG를 설명해줘")
    print(f"응답: {response.content}")

    # 임베딩 테스트
    print("\n=== 임베딩 테스트 ===")
    vec = embeddings.embed_query("RAG란 무엇인가")
    print(f"임베딩 차원: {len(vec)}")
    print(f"첫 5개 값: {vec[:5]}")
```

**직접 작성해보세요 — 실습 과제 2:**

```python
# 실습 2-1: OllamaLLM 확장
# SystemMessage와 HumanMessage를 함께 사용하는 대화 함수를 작성하라.
# LangChain의 ChatPromptTemplate을 사용하지 말고, 직접 메시지 목록을 구성하라.

from langchain_core.messages import SystemMessage, HumanMessage

def chat_with_system(llm: OllamaLLM, system_prompt: str, user_message: str) -> str:
    """
    시스템 프롬프트와 사용자 메시지를 결합하여 LLM에 전달하고 응답을 반환한다.
    """
    # TODO: 메시지 목록을 구성하고 llm.invoke()를 호출하라
    # 힌트: messages = [SystemMessage(content=...), HumanMessage(content=...)]
    pass

# 테스트
llm = create_llm()
result = chat_with_system(
    llm,
    system_prompt="당신은 RAG 전문가입니다. 모든 답변은 한국어로 3문장 이내로 답하세요.",
    user_message="임베딩이란 무엇인가요?"
)
print(result)

# 실습 2-2: 배치 임베딩 성능 측정
# embed_documents는 texts를 순차적으로 처리한다.
# 10개 텍스트를 임베딩할 때 걸리는 시간을 측정하라.
import time

texts = [f"RAG 관련 문서 청크 번호 {i}" for i in range(10)]
embeddings = create_embeddings()

start = time.time()
# TODO: embed_documents 호출
elapsed = time.time() - start
print(f"10개 임베딩 소요 시간: {elapsed:.2f}초")
print(f"청크당 평균 시간: {elapsed/10:.2f}초")
```

---

## 5. Document Loader — 파일 읽기

### 5.1 LangChain Document 객체

LangChain에서 모든 문서는 `Document` 객체로 표현된다:

```python
from langchain.schema import Document

doc = Document(
    page_content="여기에 문서의 텍스트 내용이 들어간다",
    metadata={
        "source": "report.csv",  # 출처 파일명
        "row": 5,                # CSV의 경우 행 번호
        "sheet": "Sheet1",       # Excel의 경우 시트명
    }
)

print(doc.page_content)  # 텍스트 내용 접근
print(doc.metadata)      # 메타데이터 접근 (딕셔너리)
```

`metadata`는 검색 후 출처를 추적하는 데 사용된다. "어떤 파일의 몇 번째 행에서 왔는가"를 기록해두면 답변의 근거를 사용자에게 보여줄 수 있다.

### 5.2 지원 파일 형식별 로더 구현

```python
# 파일: loaders.py
# 목적: CSV, Excel, TXT, MD, LOG, 이미지를 Document 목록으로 변환
# 지원 형식: .csv, .xlsx, .xls, .txt, .md, .log, .jpg, .jpeg, .png
# 비지원 형식: PDF (지원하지 않음)

import csv
import glob
import os
from pathlib import Path
from typing import List
from langchain.schema import Document


# ─────────────────────────────────────────────
# CSV 로더
# ─────────────────────────────────────────────

def load_csv(path: str) -> List[Document]:
    """
    CSV 파일을 행(row)별로 Document 목록으로 변환한다.

    각 행의 컬럼명: 값 형식으로 page_content를 구성하여
    LLM이 컬럼 정보를 이해할 수 있게 한다.

    예시 입력:
        이름,부서,역할
        홍길동,개발팀,백엔드

    예시 출력 Document.page_content:
        이름: 홍길동
        부서: 개발팀
        역할: 백엔드
    """
    docs = []
    with open(path, encoding="utf-8-sig") as f:  # BOM 처리를 위해 utf-8-sig 사용
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # 빈 값 제외, "컬럼: 값" 형식으로 content 구성
            content = "\n".join(
                f"{k}: {v}"
                for k, v in row.items()
                if v and str(v).strip()
            )
            if not content.strip():
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(path),
                    "file_type": "csv",
                    "row": i,
                }
            ))
    return docs


# ─────────────────────────────────────────────
# Excel 로더
# ─────────────────────────────────────────────

def load_excel(path: str) -> List[Document]:
    """
    Excel 파일(.xlsx, .xls)의 모든 시트를 Document 목록으로 변환한다.

    - 첫 행을 헤더로 인식
    - 빈 셀(None)은 제외
    - 시트명을 metadata에 기록
    """
    import openpyxl

    docs = []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        if not rows:
            continue

        # 첫 행 = 헤더, None인 경우 "col{인덱스}"로 대체
        headers = [
            str(h).strip() if h is not None else f"col{i}"
            for i, h in enumerate(rows[0])
        ]

        for row_idx, row in enumerate(rows[1:], start=1):
            content = "\n".join(
                f"{headers[j]}: {v}"
                for j, v in enumerate(row)
                if v is not None and str(v).strip()
            )
            if not content.strip():
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(path),
                    "file_type": "excel",
                    "sheet": sheet_name,
                    "row": row_idx,
                }
            ))

    wb.close()
    return docs


# ─────────────────────────────────────────────
# 텍스트 계열 로더 (TXT, MD, LOG)
# ─────────────────────────────────────────────

def load_text(path: str) -> List[Document]:
    """
    TXT, MD, LOG 파일 전체를 하나의 Document로 변환한다.

    TextSplitter로 나중에 청킹되므로, 여기서는 파일 전체를
    하나의 Document로 반환한다.

    인코딩 처리: UTF-8 우선, 실패 시 CP949(한국어 윈도우) 시도
    """
    ext = Path(path).suffix.lower()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = Path(path).read_text(encoding="cp949", errors="replace")

    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={
            "source": os.path.basename(path),
            "file_type": ext.lstrip("."),
        }
    )]


# ─────────────────────────────────────────────
# 이미지 로더 (OCR)
# ─────────────────────────────────────────────

def load_image(path: str) -> List[Document]:
    """
    이미지 파일(JPG, JPEG, PNG)에서 OCR로 텍스트를 추출한다.

    tesseract와 pytesseract가 설치되어 있어야 한다.
    한국어+영어 혼합 인식 (kor+eng).

    설치:
        macOS: brew install tesseract tesseract-lang
        Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-kor
        pip: pip install pytesseract pillow
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        print(f"[경고] pytesseract 미설치 — {path} 건너뜀")
        return []

    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang="kor+eng")
    except Exception as e:
        print(f"[경고] OCR 실패 — {path}: {e}")
        return []

    if not text.strip():
        return []

    return [Document(
        page_content=text,
        metadata={
            "source": os.path.basename(path),
            "file_type": "image",
        }
    )]


# ─────────────────────────────────────────────
# 통합 로더
# ─────────────────────────────────────────────

# 확장자 → 로더 함수 매핑
LOADERS = {
    ".csv":  load_csv,
    ".xlsx": load_excel,
    ".xls":  load_excel,
    ".txt":  load_text,
    ".md":   load_text,
    ".log":  load_text,
    ".jpg":  load_image,
    ".jpeg": load_image,
    ".png":  load_image,
}


def load_directory(data_dir: str = "./data") -> List[Document]:
    """
    디렉토리의 모든 지원 파일을 로드하여 Document 목록으로 반환한다.

    Args:
        data_dir: 문서 파일이 있는 디렉토리 경로

    Returns:
        모든 파일의 Document 목록 (합산)
    """
    all_docs: List[Document] = []

    for ext, loader in LOADERS.items():
        pattern = os.path.join(data_dir, f"*{ext}")
        for file_path in glob.glob(pattern):
            try:
                docs = loader(file_path)
                all_docs.extend(docs)
                print(f"  로드: {os.path.basename(file_path)} → {len(docs)}개 Document")
            except Exception as e:
                print(f"  [오류] {file_path}: {e}")

    return all_docs


# 단독 실행 테스트
if __name__ == "__main__":
    print("=== Document Loader 테스트 ===")

    # data/ 디렉토리가 없으면 생성하고 샘플 파일 만들기
    os.makedirs("./data", exist_ok=True)

    # 샘플 CSV 생성
    sample_csv = "./data/sample.csv"
    with open(sample_csv, "w", encoding="utf-8") as f:
        f.write("제목,내용,날짜\n")
        f.write("RAG 소개,RAG는 검색 증강 생성 기법입니다,2024-01-01\n")
        f.write("임베딩 설명,임베딩은 텍스트를 벡터로 변환합니다,2024-01-02\n")

    # 샘플 TXT 생성
    sample_txt = "./data/sample.txt"
    with open(sample_txt, "w", encoding="utf-8") as f:
        f.write("LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.\n")
        f.write("Chroma는 오픈소스 벡터 데이터베이스입니다.\n")

    docs = load_directory("./data")
    print(f"\n총 Document 수: {len(docs)}")
    for doc in docs:
        print(f"\n[{doc.metadata}]")
        print(doc.page_content[:100])
```

**직접 작성해보세요 — 실습 과제 3:**

```python
# 실습 3-1: load_markdown 함수 작성
# MD 파일에서 헤더(# 으로 시작하는 줄)만 따로 추출하여
# 각 섹션을 별도 Document로 만드는 함수를 작성하라.
# 이 방식은 마크다운 문서의 구조를 유지하는 데 유용하다.

def load_markdown_by_section(path: str) -> List[Document]:
    """
    마크다운 파일을 헤더(#) 기준으로 섹션별 Document로 분리한다.

    예시:
        # 제목
        내용1
        ## 섹션2
        내용2

    결과:
        Document(page_content="# 제목\n내용1", metadata={"section": "제목"})
        Document(page_content="## 섹션2\n내용2", metadata={"section": "섹션2"})
    """
    # TODO: 구현
    pass

# 실습 3-2: 로더 통합 테스트
# load_directory 함수를 사용하여 data/ 디렉토리의 파일을 모두 로드하고
# 파일 형식별 Document 수를 출력하는 코드를 작성하라.

docs = load_directory("./data")
# TODO: file_type별로 그룹화하여 카운트 출력
# 예상 출력:
# csv: 2개
# txt: 1개
# md: 0개
```

---

## 6. Text Splitter — 청킹 전략

### 6.1 청킹이 필요한 이유

LLM에는 한 번에 처리할 수 있는 토큰 수 제한(컨텍스트 윈도우)이 있다. `llama3.2:3b`는 약 8,000 토큰이다. 문서 전체가 이 한계를 넘으면 잘라낼 수밖에 없다. 또한, 너무 긴 컨텍스트는 검색 정확도를 떨어뜨린다 — "잃어버린 가운데(Lost in the Middle)" 현상으로 LLM은 컨텍스트의 가운데 부분을 잘 활용하지 못한다.

**청크 크기와 품질의 트레이드오프:**

```
청크가 너무 작을 때 (예: 100자)
  장점: 검색 정밀도 높음 (관련 부분만 가져옴)
  단점: 컨텍스트 부족 → LLM이 답변하기 어려움
       예: "RAG는..." → "RAG는 무엇이 줄임말인지" 누락

청크가 너무 클 때 (예: 5000자)
  장점: 풍부한 컨텍스트
  단점: 노이즈 포함 → 검색 정확도 하락
       컨텍스트 윈도우 낭비

권장 시작값: chunk_size=500, chunk_overlap=50
```

### 6.2 RecursiveCharacterTextSplitter

LangChain의 기본 텍스트 분할기. 분리자(separator)를 우선순위 순서로 시도한다.

```python
# 파일: chunking_demo.py
# 목적: 청킹 전략과 overlap의 효과를 직접 확인

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 샘플 문서 (실제 긴 문서를 시뮬레이션)
sample_text = """
RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기법입니다.

LLM의 한계를 극복하기 위해 개발되었습니다. LLM은 학습 데이터 이후의 정보를
알지 못하며, 사내 문서에 대한 지식도 없습니다.

RAG는 두 단계로 동작합니다. 첫 번째는 인덱싱 단계입니다. 문서를 청크로 분할하고
임베딩 벡터로 변환하여 벡터 DB에 저장합니다.

두 번째는 검색 및 생성 단계입니다. 사용자의 질문을 임베딩하고 벡터 DB에서
유사한 청크를 검색합니다. 검색된 청크와 질문을 함께 LLM에 전달하여 답변을 생성합니다.

임베딩 모델은 텍스트를 고차원 벡터로 변환합니다. nomic-embed-text 모델은
768차원 벡터를 생성하며, 의미가 비슷한 텍스트는 벡터 공간에서 가깝게 위치합니다.
"""

# ─────────────────────────────────────────────
# 기본 청킹
# ─────────────────────────────────────────────

def demo_basic_chunking():
    """기본 RecursiveCharacterTextSplitter 동작 확인"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,         # 청크 최대 크기 (문자 수)
        chunk_overlap=20,       # 청크 간 겹치는 문자 수
        separators=["\n\n", "\n", ".", " ", ""]  # 분리 우선순위
    )

    doc = Document(page_content=sample_text, metadata={"source": "sample.txt"})
    chunks = splitter.split_documents([doc])

    print(f"원본 길이: {len(sample_text)}자")
    print(f"청크 수: {len(chunks)}")
    print(f"청크 크기 분포:")
    sizes = [len(c.page_content) for c in chunks]
    print(f"  최소: {min(sizes)}자, 최대: {max(sizes)}자, 평균: {sum(sizes)/len(sizes):.0f}자")

    print("\n=== 청크 목록 ===")
    for i, chunk in enumerate(chunks):
        print(f"\n[청크 {i+1}] ({len(chunk.page_content)}자)")
        print(chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content)


# ─────────────────────────────────────────────
# Overlap의 효과 시각화
# ─────────────────────────────────────────────

def demo_overlap_effect():
    """
    Overlap이 있을 때와 없을 때의 차이를 비교한다.

    핵심: 청크 경계에서 중요한 정보가 잘릴 수 있다.
    Overlap이 있으면 해당 정보가 두 청크에 모두 포함된다.
    """

    text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"

    no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=0, separators=[" "]
    )
    with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=3, separators=[" "]
    )

    chunks_no = no_overlap.split_text(text)
    chunks_with = with_overlap.split_text(text)

    print("=== Overlap 없음 ===")
    for i, c in enumerate(chunks_no):
        print(f"  청크{i+1}: '{c}'")

    print("\n=== Overlap=3 ===")
    for i, c in enumerate(chunks_with):
        print(f"  청크{i+1}: '{c}'")


# ─────────────────────────────────────────────
# 청크 크기별 비교
# ─────────────────────────────────────────────

def compare_chunk_sizes():
    """청크 크기가 100, 300, 500일 때 결과 비교"""

    doc = Document(page_content=sample_text, metadata={"source": "test"})

    for size in [100, 300, 500]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=size // 10,
        )
        chunks = splitter.split_documents([doc])
        print(f"chunk_size={size}: {len(chunks)}개 청크")


if __name__ == "__main__":
    print("=== 기본 청킹 데모 ===")
    demo_basic_chunking()

    print("\n=== Overlap 효과 ===")
    demo_overlap_effect()

    print("\n=== 청크 크기 비교 ===")
    compare_chunk_sizes()
```

**직접 작성해보세요 — 실습 과제 4:**

```python
# 실습 4-1: 한국어 청킹 실험
# 한국어 텍스트는 마침표 대신 줄바꿈 기준으로 분리하는 것이 적합할 수 있다.
# 아래 한국어 문서를 두 가지 방식으로 청킹하고 결과를 비교하라.

korean_doc = """
장애 보고서 2024-01-15

오전 9:23에 API 서버 응답 지연이 탐지되었습니다.
응답 시간이 평균 500ms에서 3000ms로 증가했습니다.
원인은 데이터베이스 커넥션 풀 고갈로 확인되었습니다.

오전 9:35에 긴급 패치를 배포했습니다.
커넥션 풀 크기를 50에서 200으로 증가했습니다.
모니터링을 통해 정상화를 확인했습니다.

재발 방지 대책으로 커넥션 풀 알람 임계값을 80%로 설정했습니다.
"""

# 방법 1: 기본 설정 (영어 마침표 기준)
splitter_a = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    # TODO: separators 설정
)

# 방법 2: 한국어 최적화 (단락 기준)
splitter_b = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    # TODO: 한국어에 맞는 separators 설정
    # 힌트: 빈 줄(\n\n)을 최우선으로 하고, 줄바꿈(\n)을 다음 우선순위로 하라
)

# TODO: 두 방식의 청크 수와 내용을 비교하라
```

---

## 7. Chroma 벡터 DB 구축

### 7.1 Chroma란?

Chroma는 오픈소스 벡터 데이터베이스다. 파이썬으로 작성되어 있고, `pip install chromadb` 하나로 설치된다. 별도의 서버 없이 로컬 디렉토리에 데이터를 저장할 수 있다(영속 모드).

**핵심 개념:**

```
Chroma DB
├── Collection (테이블 개념)
│   ├── id: "doc_0001"
│   ├── embedding: [0.12, -0.34, ...]  ← 벡터 인덱스
│   ├── document: "RAG는 검색 증강..."   ← 텍스트 원본
│   └── metadata: {"source": "a.csv"}   ← 메타데이터
```

### 7.2 Chroma DB 구축 및 재사용

```python
# 파일: chroma_demo.py
# 목적: Chroma DB의 기본 동작과 영속성 확인

import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ollama_llm import create_embeddings  # 위에서 작성한 클래스

load_dotenv()


def build_vectorstore(
    docs: list[Document],
    persist_dir: str = "./chroma_db",
) -> Chroma:
    """
    Document 목록으로 Chroma 벡터 DB를 구축한다.

    첫 실행: 임베딩 계산 후 persist_dir에 저장
    이후 실행: load_vectorstore()로 재로딩 (임베딩 재계산 없음)
    """
    embeddings = create_embeddings()

    # 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"청크 수: {len(chunks)}")

    # 벡터 DB 구축 (임베딩 계산 + 저장)
    print("임베딩 계산 중... (시간이 걸릴 수 있습니다)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Chroma DB 저장 완료: {persist_dir}")

    return vectorstore


def load_vectorstore(persist_dir: str = "./chroma_db") -> Chroma:
    """
    이미 구축된 Chroma DB를 재로딩한다.

    임베딩을 재계산하지 않으므로 빠르다.
    """
    embeddings = create_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # 저장된 문서 수 확인
    count = vectorstore._collection.count()
    print(f"Chroma DB 로드 완료: {count}개 청크")

    return vectorstore


def demo_search_types(vectorstore: Chroma, query: str):
    """
    3가지 검색 방식을 비교한다.

    1. similarity_search: 기본 유사도 검색
    2. similarity_search_with_score: 점수 포함 검색
    3. max_marginal_relevance_search: 다양성 고려 검색 (MMR)
    """
    print(f"\n=== 검색 쿼리: '{query}' ===")

    # 1. 기본 유사도 검색
    print("\n[1] 기본 유사도 검색 (k=3)")
    sim_results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(sim_results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:80]}...")

    # 2. 점수 포함 검색 (L2 거리, 낮을수록 유사)
    print("\n[2] 점수 포함 검색 (k=3)")
    scored_results = vectorstore.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(scored_results, 1):
        print(f"  {i}. 거리={score:.4f} [{doc.metadata.get('source')}] {doc.page_content[:60]}...")

    # 3. MMR — 관련성과 다양성을 동시에 고려
    # fetch_k개를 먼저 검색한 후, 그 중 가장 다양한 k개를 반환
    print("\n[3] MMR 검색 (k=3, fetch_k=10)")
    mmr_results = vectorstore.max_marginal_relevance_search(
        query, k=3, fetch_k=10
    )
    for i, doc in enumerate(mmr_results, 1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:80]}...")


def demo_metadata_filter(vectorstore: Chroma, query: str):
    """
    메타데이터 필터링으로 특정 파일 출처만 검색한다.
    """
    print(f"\n=== 메타데이터 필터 검색 ===")

    # 'source'가 'sample.csv'인 Document만 검색
    filtered_results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"source": "sample.csv"},  # 메타데이터 필터
    )
    print(f"'sample.csv' 출처 결과: {len(filtered_results)}개")
    for doc in filtered_results:
        print(f"  {doc.metadata} | {doc.page_content[:60]}...")


if __name__ == "__main__":
    # 샘플 문서로 테스트
    sample_docs = [
        Document(
            page_content="RAG는 검색 증강 생성 기법으로, LLM에 외부 지식을 제공한다.",
            metadata={"source": "sample.csv", "row": 0}
        ),
        Document(
            page_content="임베딩은 텍스트를 고차원 벡터로 변환하는 기술이다.",
            metadata={"source": "sample.csv", "row": 1}
        ),
        Document(
            page_content="Chroma는 오픈소스 벡터 데이터베이스로 로컬에서 실행된다.",
            metadata={"source": "readme.md"}
        ),
    ]

    vectorstore = build_vectorstore(sample_docs, persist_dir="./test_chroma")
    demo_search_types(vectorstore, "벡터 검색이란?")
    demo_metadata_filter(vectorstore, "임베딩")
```

**직접 작성해보세요 — 실습 과제 5:**

```python
# 실습 5-1: Chroma DB 문서 추가 및 삭제
# 이미 만들어진 vectorstore에 새 문서를 추가하고,
# 특정 id를 가진 문서를 삭제하는 코드를 작성하라.

vectorstore = load_vectorstore("./test_chroma")

# TODO: 새 문서 추가
new_docs = [
    Document(
        page_content="LangChain은 LLM 애플리케이션 개발 프레임워크다.",
        metadata={"source": "new_doc.txt"}
    )
]
# 힌트: vectorstore.add_documents(new_docs)

# TODO: 추가 후 총 문서 수 확인

# 실습 5-2: 점수 기반 필터링
# similarity_search_with_score를 사용하여
# 거리가 1.0 이하인 결과만 반환하는 함수를 작성하라.

def search_with_threshold(
    vectorstore: Chroma,
    query: str,
    k: int = 10,
    max_distance: float = 1.0
) -> list:
    """거리 임계값 이하의 결과만 반환"""
    # TODO: 구현
    pass
```

---

## 8. RAG 파이프라인 완성

### 8.1 RetrievalQA 체인 구성 원리

LangChain의 `RetrievalQA`는 검색(Retrieval)과 생성(Generation)을 연결하는 체인이다. 내부 동작:

```
사용자 질문 입력
      │
      ▼
retriever.invoke(질문)    ← 벡터 DB에서 k개 청크 검색
      │
      ▼
프롬프트 구성
  "컨텍스트: {청크1}\n{청크2}\n...\n\n질문: {질문}"
      │
      ▼
llm.invoke(프롬프트)       ← LLM 답변 생성
      │
      ▼
결과 반환 {"result": "답변", "source_documents": [청크1, ...]}
```

### 8.2 기본 RAG 파이프라인

```python
# 파일: rag_basic.py
# 목적: 가장 단순한 RAG Q&A 파이프라인 구현
# 실행: python rag_basic.py

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from loaders import load_directory
from ollama_llm import create_llm, create_embeddings

load_dotenv()


def build_and_query():
    """데이터 로드 → 인덱싱 → Q&A 순서로 실행"""

    # 1. 문서 로드
    print("1. 문서 로드 중...")
    docs = load_directory("./data")
    if not docs:
        print("  [경고] data/ 디렉토리에 파일이 없습니다.")
        return

    # 2. 청킹
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  총 청크 수: {len(chunks)}")

    # 3. 벡터 DB 구축
    print("2. 벡터 DB 구축 중...")
    embeddings = create_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )

    # 4. RAG 체인 구성
    print("3. RAG 체인 구성...")
    llm = create_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                                  # 모든 청크를 하나의 프롬프트에
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}                           # 상위 3개 청크 검색
        ),
        return_source_documents=True,                        # 근거 문서 반환
    )

    # 5. 질문 루프
    print("\n=== RAG Q&A 봇 시작 (종료: 'q' 입력) ===\n")
    while True:
        question = input("질문: ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue

        result = qa_chain.invoke({"query": question})

        print(f"\n[답변]\n{result['result']}")
        print("\n[근거 문서]")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:100]}...")
        print()


if __name__ == "__main__":
    build_and_query()
```

### 8.3 기존 Chroma DB 재사용 (효율적 버전)

매번 임베딩을 재계산하는 것은 비효율적이다. `persist_directory`가 있으면 재로딩한다.

```python
# 파일: rag_with_cache.py
# 목적: 기존 Chroma DB를 재사용하는 RAG 파이프라인

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from loaders import load_directory
from ollama_llm import create_llm, create_embeddings

load_dotenv()

CHROMA_DIR = "./chroma_db"


def get_or_build_vectorstore() -> Chroma:
    """
    Chroma DB가 이미 존재하면 재로딩, 없으면 새로 구축한다.

    Returns:
        Chroma 인스턴스
    """
    embeddings = create_embeddings()

    # chroma_db 디렉토리와 chroma.sqlite3 파일이 있으면 재로딩
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")

    if os.path.exists(sqlite_path):
        print(f"기존 Chroma DB 로드: {CHROMA_DIR}")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        print(f"  저장된 청크 수: {count}")
        return vectorstore

    # 새로 구축
    print("새 Chroma DB 구축 중...")
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = load_directory("./data")
    if not docs:
        raise ValueError("data/ 디렉토리에 파일이 없습니다.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  구축 완료: {len(chunks)}개 청크 저장")
    return vectorstore


def main():
    vectorstore = get_or_build_vectorstore()
    llm = create_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    # 인터랙티브 Q&A
    print("\n=== RAG Q&A (종료: 'q') ===")
    while True:
        q = input("\n질문: ").strip()
        if q.lower() == "q":
            break
        result = qa_chain.invoke({"query": q})
        print(f"\n답변: {result['result']}")
        print("근거:")
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            row = doc.metadata.get("row", "")
            loc = f"{src}:{row}" if row != "" else src
            print(f"  - [{loc}] {doc.page_content[:80]}...")


if __name__ == "__main__":
    main()
```

---

## 9. 프롬프트 엔지니어링

### 9.1 기본 RAG 프롬프트의 문제

기본 `RetrievalQA`가 사용하는 내장 프롬프트:

```
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know.

{context}

Question: {question}
```

이 프롬프트의 문제:
1. 영어로 되어 있어 한국어 LLM과 잘 맞지 않는다
2. "just say you don't know"가 실제로 잘 지켜지지 않는다
3. 근거 인용(citation)을 요청하지 않는다

### 9.2 개선된 한국어 RAG 프롬프트

```python
# 파일: rag_with_prompt.py
# 목적: 커스텀 프롬프트 템플릿으로 RAG 품질 개선

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from ollama_llm import create_llm, create_embeddings
from rag_with_cache import get_or_build_vectorstore  # 위에서 작성한 함수

load_dotenv()


# ─────────────────────────────────────────────
# 개선된 프롬프트 템플릿
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """당신은 주어진 컨텍스트 문서만을 사용하여 질문에 답하는 전문가입니다.

[컨텍스트 문서]
{context}

[질문]
{question}

[답변 규칙]
1. 반드시 위 컨텍스트 문서에 있는 정보만 사용하여 답변하세요.
2. 컨텍스트에 관련 정보가 없으면 정확히 다음과 같이 답하세요:
   "제공된 문서에서 해당 정보를 찾을 수 없습니다."
3. 답변에 근거가 된 문장을 인용하세요. 예: (출처: sample.csv)
4. 추측이나 일반 지식을 추가하지 마세요.
5. 한국어로 답변하세요.

[답변]"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """커스텀 프롬프트를 적용한 RAG 체인 생성"""
    llm = create_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},   # 커스텀 프롬프트 적용
        return_source_documents=True,
    )


def query_with_sources(qa_chain: RetrievalQA, question: str) -> dict:
    """
    질문에 답하고 결과를 구조화된 형태로 반환한다.

    Returns:
        {
            "question": 질문,
            "answer": 답변,
            "sources": [{"file": 파일명, "row": 행번호, "content": 내용}]
        }
    """
    result = qa_chain.invoke({"query": question})

    sources = []
    for doc in result["source_documents"]:
        sources.append({
            "file": doc.metadata.get("source", "unknown"),
            "row": doc.metadata.get("row", ""),
            "sheet": doc.metadata.get("sheet", ""),
            "content": doc.page_content[:200],
        })

    return {
        "question": question,
        "answer": result["result"],
        "sources": sources,
    }


def print_result(result: dict):
    """결과를 보기 좋게 출력"""
    print(f"\n{'='*60}")
    print(f"질문: {result['question']}")
    print(f"\n답변:\n{result['answer']}")
    print(f"\n근거 문서 ({len(result['sources'])}개):")
    for i, src in enumerate(result['sources'], 1):
        loc = src['file']
        if src['row'] != '':
            loc += f" (행: {src['row']})"
        if src['sheet']:
            loc += f" [시트: {src['sheet']}]"
        print(f"  {i}. [{loc}]")
        print(f"     {src['content'][:100]}...")
    print('='*60)


if __name__ == "__main__":
    vectorstore = get_or_build_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    # 테스트 질문들
    test_questions = [
        "RAG의 핵심 구성 요소는 무엇인가요?",
        "임베딩 모델의 차원은 몇 개인가요?",
        "오늘 날씨는 어떻습니까?",  # 문서에 없는 질문 → "모른다" 응답 테스트
    ]

    for question in test_questions:
        result = query_with_sources(qa_chain, question)
        print_result(result)
```

**직접 작성해보세요 — 실습 과제 6:**

```python
# 실습 6-1: 프롬프트 개선
# 현재 프롬프트에서 다음 기능을 추가하라:
# 1. 답변 길이를 "3문장 이내"로 제한하라
# 2. 컨텍스트에 여러 출처가 있을 때, 각 문장에 출처를 (출처: 파일명) 형식으로 표시하라
# 3. "확실하지 않다"고 판단되면 신뢰도를 "낮음/보통/높음"으로 표시하라

IMPROVED_PROMPT_TEMPLATE = """당신은 주어진 컨텍스트만 사용하는 전문 분석가입니다.

[컨텍스트]
{context}

[질문]
{question}

[규칙]
# TODO: 위 세 가지 기능을 포함한 규칙을 작성하라

[답변]"""

# 실습 6-2: 멀티턴 대화 구현
# 이전 질문과 답변을 컨텍스트에 포함하여 대화 맥락을 유지하는
# 대화형 RAG를 구현하라.
# LangChain의 ConversationalRetrievalChain을 참고하라.
# pip install langchain

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def build_conversational_qa(vectorstore):
    """
    대화 이력을 유지하는 RAG 체인을 생성한다.
    이전 질문-답변 쌍을 자동으로 컨텍스트에 추가한다.
    """
    # TODO: ConversationalRetrievalChain 구성
    # 힌트: memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pass
```

---

## 10. 전체 실습 프로젝트: 다중 파일 Q&A 봇

### 10.1 요구사항

1. CSV / Excel / TXT / MD / LOG / 이미지 혼합 지원
2. Chroma DB 영속성 (재실행 시 임베딩 재계산 없음)
3. 근거 파일명 및 행 번호 표시
4. "모르면 모른다고" 하는 프롬프트
5. 대화형 CLI 인터페이스

### 10.2 완성 코드

```python
# 파일: rag_pipeline.py
# 목적: Level 1 완성 RAG Q&A 봇
# 실행: python rag_pipeline.py
# 실행 (재인덱싱): python rag_pipeline.py --rebuild

import os
import sys
import argparse
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 위에서 작성한 모듈들
from ollama_llm import create_llm, create_embeddings
from loaders import load_directory

load_dotenv()

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

SYSTEM_PROMPT = """당신은 주어진 컨텍스트 문서만을 사용하여 질문에 답하는 전문가입니다.

[컨텍스트 문서]
{context}

[질문]
{question}

[답변 규칙]
1. 반드시 위 컨텍스트 문서에 있는 정보만 사용하세요.
2. 컨텍스트에 관련 정보가 없으면 정확히 이렇게 답하세요:
   "제공된 문서에서 해당 정보를 찾을 수 없습니다."
3. 답변에 사용한 출처를 (출처: 파일명) 형식으로 명시하세요.
4. 추측이나 일반 상식을 추가하지 마세요.
5. 한국어로 답변하세요.

[답변]"""


# ─────────────────────────────────────────────
# 벡터 DB 관리
# ─────────────────────────────────────────────

def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    벡터 DB를 구축하거나 재로딩한다.

    Args:
        force_rebuild: True면 기존 DB를 무시하고 새로 구축
    """
    embeddings = create_embeddings()
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")

    # 재사용 가능하고 강제 재구축이 아니면 로딩
    if os.path.exists(sqlite_path) and not force_rebuild:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        print(f"[Chroma] 기존 DB 로드 ({count}개 청크)")
        return vectorstore

    # 새로 구축
    print("[Chroma] 새 DB 구축 중...")

    docs = load_directory(DATA_DIR)
    if not docs:
        raise FileNotFoundError(
            f"'{DATA_DIR}' 디렉토리에 지원 파일이 없습니다.\n"
            "지원 형식: CSV, Excel, TXT, MD, LOG, JPG, JPEG, PNG"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[청킹] {len(docs)}개 문서 → {len(chunks)}개 청크")

    # 기존 DB 삭제 (rebuild 시)
    if force_rebuild and os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[Chroma] 구축 완료 → {CHROMA_DIR}")
    return vectorstore


# ─────────────────────────────────────────────
# RAG 체인 구성
# ─────────────────────────────────────────────

def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """커스텀 프롬프트를 적용한 RAG 체인"""
    llm = create_llm()
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


# ─────────────────────────────────────────────
# 출력 포맷
# ─────────────────────────────────────────────

def format_result(question: str, result: dict) -> str:
    """
    RAG 결과를 사람이 읽기 좋은 형식으로 변환한다.
    """
    lines = [
        f"\n{'─'*60}",
        f"질문: {question}",
        f"\n답변:\n{result['result']}",
    ]

    if result.get("source_documents"):
        lines.append(f"\n근거 문서 ({len(result['source_documents'])}개):")
        for i, doc in enumerate(result["source_documents"], 1):
            meta = doc.metadata
            loc_parts = [meta.get("source", "unknown")]
            if "row" in meta:
                loc_parts.append(f"행{meta['row']}")
            if "sheet" in meta:
                loc_parts.append(f"시트:{meta['sheet']}")
            loc = ", ".join(loc_parts)
            content_preview = doc.page_content[:100].replace("\n", " ")
            lines.append(f"  {i}. [{loc}]")
            lines.append(f"     {content_preview}...")

    lines.append("─"*60)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="다중 파일 Q&A 봇")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Chroma DB를 강제로 재구축합니다",
    )
    args = parser.parse_args()

    print("=== 다중 파일 Q&A 봇 (Level 1) ===")
    print(f"LLM: {os.getenv('LLM_MODEL', 'llama3.2:3b')}")
    print(f"임베딩: {os.getenv('EMBED_MODEL', 'nomic-embed-text')}")
    print(f"데이터: {DATA_DIR}")

    try:
        vectorstore = build_vectorstore(force_rebuild=args.rebuild)
        qa_chain = build_qa_chain(vectorstore)
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        sys.exit(1)

    print("\n대화 시작 (종료: 'q' 또는 Ctrl+C)\n")

    while True:
        try:
            question = input("질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit", "종료"):
            print("종료합니다.")
            break

        result = qa_chain.invoke({"query": question})
        print(format_result(question, result))


if __name__ == "__main__":
    main()
```

---

## 11. 테스트 작성

### 11.1 pytest 테스트

```python
# 파일: test_loaders.py
# 실행: pytest test_loaders.py -v

import os
import csv
import tempfile
import pytest
from pathlib import Path
from langchain.schema import Document

# 테스트 대상 모듈
from loaders import load_csv, load_excel, load_text, load_image, load_directory


# ─────────────────────────────────────────────
# Fixture: 임시 파일 생성
# ─────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """임시 디렉토리를 생성하고 테스트 후 삭제"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_csv(temp_dir):
    """샘플 CSV 파일 생성"""
    path = os.path.join(temp_dir, "test.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["이름", "부서", "역할"])
        writer.writeheader()
        writer.writerow({"이름": "홍길동", "부서": "개발팀", "역할": "백엔드"})
        writer.writerow({"이름": "김철수", "부서": "운영팀", "역할": "DevOps"})
        writer.writerow({"이름": "", "부서": "", "역할": ""})  # 빈 행
    return path


@pytest.fixture
def sample_txt(temp_dir):
    """샘플 TXT 파일 생성"""
    path = os.path.join(temp_dir, "test.txt")
    Path(path).write_text("RAG는 검색 증강 생성입니다.\nLangChain을 사용합니다.", encoding="utf-8")
    return path


@pytest.fixture
def sample_md(temp_dir):
    """샘플 MD 파일 생성"""
    path = os.path.join(temp_dir, "test.md")
    Path(path).write_text("# 제목\n## 섹션1\n내용입니다.", encoding="utf-8")
    return path


@pytest.fixture
def empty_txt(temp_dir):
    """빈 TXT 파일"""
    path = os.path.join(temp_dir, "empty.txt")
    Path(path).write_text("   \n\n   ", encoding="utf-8")
    return path


# ─────────────────────────────────────────────
# CSV 로더 테스트
# ─────────────────────────────────────────────

class TestLoadCsv:
    def test_returns_document_list(self, sample_csv):
        docs = load_csv(sample_csv)
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_non_empty_rows_only(self, sample_csv):
        """빈 행은 포함하지 않는다"""
        docs = load_csv(sample_csv)
        assert len(docs) == 2  # 빈 행 제외

    def test_content_format(self, sample_csv):
        """page_content가 '키: 값' 형식이다"""
        docs = load_csv(sample_csv)
        assert "이름: 홍길동" in docs[0].page_content
        assert "부서: 개발팀" in docs[0].page_content

    def test_metadata_source(self, sample_csv):
        """metadata에 source (파일명)가 있다"""
        docs = load_csv(sample_csv)
        assert docs[0].metadata["source"] == "test.csv"

    def test_metadata_row(self, sample_csv):
        """metadata에 row 번호가 있다"""
        docs = load_csv(sample_csv)
        assert "row" in docs[0].metadata

    def test_file_not_found(self):
        """존재하지 않는 파일은 예외 발생"""
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path.csv")


# ─────────────────────────────────────────────
# 텍스트 로더 테스트
# ─────────────────────────────────────────────

class TestLoadText:
    def test_returns_document(self, sample_txt):
        docs = load_text(sample_txt)
        assert len(docs) == 1

    def test_content_preserved(self, sample_txt):
        docs = load_text(sample_txt)
        assert "RAG는 검색 증강 생성입니다" in docs[0].page_content

    def test_empty_file_returns_empty_list(self, empty_txt):
        docs = load_text(empty_txt)
        assert docs == []

    def test_md_extension(self, sample_md):
        """MD 파일도 load_text로 로드된다"""
        docs = load_text(sample_md)
        assert len(docs) == 1
        assert "# 제목" in docs[0].page_content

    def test_metadata_file_type(self, sample_txt):
        docs = load_text(sample_txt)
        assert docs[0].metadata["file_type"] == "txt"


# ─────────────────────────────────────────────
# load_directory 테스트
# ─────────────────────────────────────────────

class TestLoadDirectory:
    def test_loads_all_supported_files(self, temp_dir, sample_csv, sample_txt, sample_md):
        docs = load_directory(temp_dir)
        # CSV 2행 + TXT 1개 + MD 1개
        assert len(docs) == 4

    def test_empty_directory(self, temp_dir):
        docs = load_directory(temp_dir)
        assert docs == []

    def test_unsupported_files_ignored(self, temp_dir):
        """지원하지 않는 형식(PDF 등)은 무시된다"""
        # PDF 파일 생성 (실제 PDF 아닌 텍스트 파일)
        pdf_path = os.path.join(temp_dir, "test.pdf")
        Path(pdf_path).write_text("PDF content", encoding="utf-8")

        docs = load_directory(temp_dir)
        assert docs == []

    def test_nonexistent_directory(self):
        """존재하지 않는 디렉토리는 빈 리스트 반환"""
        docs = load_directory("/nonexistent/directory")
        assert docs == []


# ─────────────────────────────────────────────
# 청킹 테스트
# ─────────────────────────────────────────────

class TestChunking:
    def test_chunk_size_respected(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
        )
        long_text = "가나다라마바사아자차카타파하" * 50  # 약 700자

        doc = Document(page_content=long_text, metadata={})
        chunks = splitter.split_documents([doc])

        # 모든 청크가 chunk_size + overlap 이하
        for chunk in chunks:
            assert len(chunk.page_content) <= 110  # 약간의 여유 허용

    def test_overlap_creates_shared_content(self):
        """Overlap이 있으면 인접 청크 간 공유 내용이 있다"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text = " ".join([f"word{i}" for i in range(100)])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            separators=[" "],
        )
        chunks = splitter.split_text(text)

        if len(chunks) >= 2:
            # 인접 청크 간 공유 단어가 있어야 함
            words_c0 = set(chunks[0].split())
            words_c1 = set(chunks[1].split())
            overlap = words_c0 & words_c1
            assert len(overlap) > 0
```

### 11.2 requirements.txt

```
# requirements.txt
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
chromadb>=0.5.0
openpyxl>=3.1.0
pillow>=10.0.0
pytesseract>=0.3.10
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
pytest>=8.0.0
```

### 11.3 pytest 실행 방법

```bash
# 전체 테스트 실행
pytest test_loaders.py -v

# 특정 클래스만 실행
pytest test_loaders.py::TestLoadCsv -v

# 특정 테스트만 실행
pytest test_loaders.py::TestLoadCsv::test_content_format -v

# 실패 시 즉시 중단
pytest test_loaders.py -x

# 커버리지 측정 (pip install pytest-cov)
pytest test_loaders.py --cov=loaders --cov-report=term-missing
```

---

## 12. 체크리스트 및 다음 단계

### 12.1 Level 1 완성 체크리스트

**개념 이해**

- [ ] RAG = Indexing + Retrieval + Generation 구조를 그림으로 설명할 수 있다
- [ ] 임베딩이 숫자 벡터임을 이해하고, 코사인 유사도를 직접 계산할 수 있다
- [ ] 청크 크기와 overlap의 trade-off를 설명할 수 있다
- [ ] `Document` 객체의 `page_content`와 `metadata`의 역할을 설명할 수 있다

**구현 능력**

- [ ] `OllamaLLM`, `OllamaEmbeddings` 클래스를 처음부터 직접 작성할 수 있다
- [ ] CSV, Excel, TXT, MD, LOG, 이미지 로더를 직접 구현할 수 있다
- [ ] Chroma DB를 구축하고 재로딩할 수 있다
- [ ] `RetrievalQA` 체인에 커스텀 프롬프트를 적용할 수 있다
- [ ] 근거 문서(파일명, 행 번호)와 함께 답변을 반환할 수 있다

**테스트 및 품질**

- [ ] pytest 테스트를 작성하고 실행할 수 있다
- [ ] "모르는 질문"에 "모른다"고 응답하는 것을 확인했다
- [ ] 기존 Chroma DB를 재사용하여 임베딩 재계산을 피할 수 있다

### 12.2 Level 2로 진행하기 전 확인사항

Level 1 RAG를 완성했다면, 다음 한계를 인식했을 것이다:

1. **검색 품질**: 벡터 유사도만으로는 키워드 정확 일치를 잘 처리하지 못한다
2. **청킹**: 고정 크기 청킹은 의미 단위를 무시한다
3. **한국어**: 한국어 형태소 분석 없이는 변형어 처리가 약하다

이 문제들을 Level 2에서 해결한다:
- **하이브리드 검색**: BM25(키워드) + 벡터(의미) 결합
- **의미 단위 청킹**: SemanticChunker, Parent-Child Chunking
- **Reranking**: 검색 후 재순위로 품질 향상

### 12.3 참고 자료

- LangChain 공식 문서: https://python.langchain.com
- Chroma 공식 문서: https://docs.trychroma.com
- Ollama API 레퍼런스: https://github.com/ollama/ollama/blob/main/docs/api.md
- nomic-embed-text 모델: https://ollama.com/library/nomic-embed-text
