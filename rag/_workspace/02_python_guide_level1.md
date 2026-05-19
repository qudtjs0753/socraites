# Level 1 실습 가이드 — Python으로 첫 RAG 만들기

> 목표: 환경 세팅부터 동작하는 PDF Q&A 봇까지, **복사하면 바로 실행되는** 코드로 RAG의 핵심 개념을 손에 익힙니다.
>
> 학습자 배경 가정: Python 경험 있음 / Chroma·Elasticsearch DA 경험 있음 / K8S·Kind AIOps 경험 있음. 그래서 **파이썬 문법보다 RAG 특유의 설계 결정**에 집중합니다.

---

## 0. 학습 흐름 한눈에 보기

```
[실습 1] 임베딩 + 코사인 유사도
   "텍스트가 어떻게 벡터가 되고, 왜 의미 검색이 되는가"
            ↓
[실습 2] PDF → Chroma → Q&A 봇 (30줄)
   "RAG 파이프라인 전체를 한 번에 체험"
            ↓
[다음 단계 힌트]
   "이 기본 RAG의 한계는 무엇이고 어디부터 깎아낼 것인가"
```

---

## 1. 환경 설정

### 1-1. 디렉토리 + 가상환경

```bash
# 작업 디렉토리 생성
mkdir -p ~/workspace/project/rag/_workspace/level1
cd ~/workspace/project/rag/_workspace/level1

# Python 3.11+ 권장. 가상환경 생성 후 활성화
python3 -m venv .venv
source .venv/bin/activate

# pip 업그레이드
pip install --upgrade pip
```

### 1-2. 의존성 설치 (`requirements.txt`)

버전을 고정해두면 "어제 되던 게 오늘 안 됨" 사고가 줄어듭니다.

```bash
cat > requirements.txt <<'EOF'
langchain==0.3.7
langchain-core==0.3.15
langchain-community==0.3.7
langchain-huggingface==0.1.2
langchain-chroma==0.1.4
sentence-transformers==3.3.1
chromadb==0.5.20
pypdf==5.1.0
python-dotenv==1.0.1
numpy==1.26.4
requests==2.32.3
EOF

pip install -r requirements.txt
```

> `openai` 패키지와 `langchain-openai`는 사용하지 않습니다. 사내 LLM 서버에 `requests`로 직접 호출합니다.

> K8S 비유로 보면 `requirements.txt`는 컨테이너 이미지 태그 핀(pin)과 같은 역할입니다. **재현 가능한 환경 = 디버깅 가능한 환경**입니다.

### 1-3. `.env` 파일 설정

LLM/임베딩 설정을 코드에 직접 박지 않고 환경 변수로 분리합니다.

```bash
cat > .env <<'EOF'
LLM_BASE_URL=http://사내-llm-서버/v1
LLM_API_KEY=사내_발급_키
LLM_MODEL=모델명
EMBED_MODEL=BAAI/bge-m3
EMBED_CACHE_DIR=./hf_cache
EOF

# git 사용 시 절대 커밋되지 않도록
cat > .gitignore <<'EOF'
.venv/
.env
chroma_db/
models/
__pycache__/
*.pyc
EOF
```

설정이 잘 로딩되는지 확인:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('LLM_BASE_URL') else 'FAIL: LLM_BASE_URL 미설정')"
# 출력: OK
```

### 1-4. 로컬 임베딩 모델 사용 (BAAI/bge-m3, 폐쇄망)

> **오프라인 다운로드 필요** — 외부망 PC에서 아래 명령어로 모델을 다운로드한 뒤, 압축해서 사내로 반입하세요.
>
> ```bash
> # 외부망 PC에서 실행
> pip install huggingface-hub
> huggingface-cli download BAAI/bge-m3 --local-dir ./bge-m3
> # bge-m3/ 디렉토리를 zip으로 묶어 메일로 전달
> ```

사내 임베딩 REST API 대신 로컬에 내려받은 BAAI/bge-m3를 직접 쓰려면 추가 패키지가 필요합니다.

```bash
pip install sentence-transformers==3.3.1
```

**핵심: 재다운로드를 막는 두 가지 설정**

`HuggingFaceEmbeddings`에 모델 이름만 넘기면 실행 때마다 HuggingFace Hub에 접속해 최신 버전을 확인합니다. 폐쇄망에서는 이 연결이 실패하거나 타임아웃이 반복됩니다. 아래 두 가지를 **반드시** 같이 써야 합니다.

| 설정 | 역할 |
|------|------|
| `model_kwargs={"local_files_only": True}` | Hub 접속 없이 캐시만 사용 |
| `cache_folder` 또는 절대 경로 | 모델이 있는 위치를 명시 |

**방법 A — `cache_folder` + `local_files_only` (권장)**

모델이 `~/models/bge-m3/` 에 있다면:

```python
# internal_llm.py 하단에 추가하거나 별도 파일로 분리 가능
from langchain_community.embeddings import HuggingFaceEmbeddings

LOCAL_MODEL_DIR = os.path.expanduser("~/models")   # bge-m3/ 폴더가 있는 상위 디렉토리

local_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",         # 폴더명 기준으로 탐색
    cache_folder=LOCAL_MODEL_DIR,      # 실제 파일 위치
    model_kwargs={
        "device": "cpu",               # GPU 없으면 cpu
        "local_files_only": True,      # Hub 접속 차단 — 핵심 설정
    },
    encode_kwargs={"normalize_embeddings": True},
)
```

**방법 B — 절대 경로로 바로 지정**

캐시 구조를 신경 쓰기 싫을 때 가장 단순합니다.

```python
local_embeddings = HuggingFaceEmbeddings(
    model_name="/home/user/models/bge-m3",  # 모델 파일이 있는 디렉토리 경로
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

**방법 C — 환경 변수로 전역 차단 (스크립트 실행 전)**

```bash
export HF_HOME=~/models          # 모델 캐시 루트
export HF_HUB_OFFLINE=1          # Hub 접속 전역 차단
export TRANSFORMERS_OFFLINE=1    # transformers 라이브러리도 차단

python 01_embedding_playground.py
```

> 방법 C는 프로세스 전체에 적용되므로 다른 패키지까지 오프라인으로 강제됩니다. 개발 편의보다 운영 환경에 더 적합합니다.

**로컬 모델 동작 확인**

```python
# 모델이 제대로 로컬에서 로딩되는지 확인
print(local_embeddings.embed_query("테스트")[:3])
# Hub 접속 없이 즉시 숫자 3개가 출력되면 성공
```

---

## 2. 실습 1 — 임베딩 직접 체험

### 2-1. 목표

> "텍스트 → 벡터 → 코사인 유사도"를 **직접 손으로 만져보고**, 의미 검색이 왜 가능한지 체득.

### 2-2. 완전한 실행 코드 (`01_embedding_playground.py`)

```python
# 목적: 임베딩이 무엇이고, 왜 "의미 검색"이 가능한지 직접 확인
# 핵심 개념: 텍스트 → 1024차원 벡터(BGE-M3) → 코사인 유사도
# 실행: python 01_embedding_playground.py

import os
import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    cache_folder=os.getenv("EMBED_CACHE_DIR", "./hf_cache"),
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

# (1) 비교할 문장들 — 의미가 비슷한 그룹과 다른 그룹을 섞어 봅니다
texts = [
    "RAG는 외부 문서를 활용해 LLM의 답변을 보강하는 기법입니다",
    "검색 증강 생성은 LLM에 최신 지식을 주입하는 방법입니다",   # 1번과 유사 (의미)
    "리트리벌은 벡터 DB에서 관련 청크를 가져오는 단계입니다",   # 1번과 약한 관련
    "오늘 서울 날씨가 맑고 기온은 22도입니다",                  # 1번과 무관
    "쿠버네티스 Pod이 CrashLoopBackOff 상태입니다",            # 1번과 무관
]

# (2) 임베딩 생성 — 배치로 한 번에 처리
vectors = [np.array(v) for v in embeddings.embed_documents(texts)]

print(f"임베딩 차원: {len(vectors[0])}")  # BGE-M3: 1024
print(f"벡터 첫 5개 값(샘플): {vectors[0][:5]}\n")

# (3) 코사인 유사도 — 벡터의 '방향'이 얼마나 같은가
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# (4) 기준 문장(0번)과 나머지 비교
print(f"[기준 문장] {texts[0]}\n")
for i in range(1, len(texts)):
    sim = cosine_similarity(vectors[0], vectors[i])
    bar = "#" * int(sim * 40)  # 시각화
    print(f"  유사도 {sim:.3f} | {bar}")
    print(f"           {texts[i]}\n")
```

### 2-3. 실행 + 예상 출력

```bash
python 01_embedding_playground.py
```

```
임베딩 차원: 1024
벡터 첫 5개 값(샘플): [ 0.0123 -0.0456  0.0789  ... ]

[기준 문장] RAG는 외부 문서를 활용해 LLM의 답변을 보강하는 기법입니다

  유사도 0.812 | ################################
           검색 증강 생성은 LLM에 최신 지식을 주입하는 방법입니다

  유사도 0.654 | ##########################
           리트리벌은 벡터 DB에서 관련 청크를 가져오는 단계입니다

  유사도 0.187 | #######
           오늘 서울 날씨가 맑고 기온은 22도입니다

  유사도 0.142 | #####
           쿠버네티스 Pod이 CrashLoopBackOff 상태입니다
```

> 실제 숫자는 약간 다를 수 있습니다(모델 버전·내부 결정성 차이). 중요한 건 **순서**: 의미가 가까울수록 점수가 높습니다.

### 2-4. 여기서 꼭 잡고 가야 할 포인트

- **임베딩은 결정론적 함수**: 같은 텍스트 + 같은 모델 = 같은 벡터. → Chroma에 저장하면 재계산 불필요(비용 절감).
- **차원 수는 모델이 정함**: 컬렉션을 만들 때 한 번 정해지면 같은 모델만 사용해야 함 (DA 경험과 동일: 스키마 호환성). 사내 모델 차원은 임베딩 담당자에게 확인.
- **유사도는 "방향" 비교**: 길이(노름)는 무시. 그래서 코사인.
- **0.6~0.8 사이가 흔히 보이는 "관련 있음" 영역**: 절대값보다 *상대 순위*가 중요.

---

## 3. 실습 2 — 첫 RAG 파이프라인 (PDF → Chroma → Q&A 봇)

### 3-1. 사용할 PDF 준비

테스트용 PDF가 없으면 아무 한글/영문 PDF를 `level1/sample.pdf`로 저장하세요. 추천:

```bash
# 예시: Python 공식 문서 PDF (없으면 본인 PDF 1개 아무거나)
curl -L -o sample.pdf "https://docs.python.org/3/archives/python-3.12.0-docs-pdf-a4.zip" || true
# 위가 안 되면 그냥 본인 가지고 있는 PDF 한 개를 sample.pdf 로 복사
```

> 빠르게 돌려보려면 **10~30페이지짜리** PDF를 추천합니다(임베딩 비용·시간 절약).

### 3-0. 공통 모듈 먼저 만들기 (`internal_llm.py`)

모든 실습에서 재사용하는 사내 LLM/임베딩 래퍼입니다. **한 번만 만들면** 나머지 실습에서는 `from internal_llm import ...`으로 가져다 씁니다.

```python
# internal_llm.py — 사내 LLM/임베딩 래퍼 (공통 모듈)
import requests
from typing import List, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.embeddings import Embeddings

class InternalChatLLM(BaseChatModel):
    """사내 LLM REST API 래퍼"""
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        role_map = {HumanMessage: "user", SystemMessage: "system", AIMessage: "assistant"}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": [{"role": role_map.get(type(m), "user"), "content": m.content} for m in messages],
                "temperature": self.temperature,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resp.json()["choices"][0]["message"]["content"]))])

    @property
    def _llm_type(self) -> str:
        return "internal_llm"

class InternalEmbeddings(Embeddings):
    """사내 임베딩 REST API 래퍼"""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url, api_key, model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": text},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
```

### 3-2. 완전한 실행 코드 (`02_pdf_rag.py`, 핵심 ~30줄)

```python
# 목적: PDF 1개로 동작하는 Q&A 봇을 30줄로 완성
# 핵심 개념: Load → Split → Embed → Store → Retrieve → Generate
# 실행: python 02_pdf_rag.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from internal_llm import InternalChatLLM  # 공통 모듈

load_dotenv()

PDF_PATH = "sample.pdf"
PERSIST_DIR = "./chroma_db"

# 사내 LLM 초기화 (requests 직접 호출 — openai 패키지 불필요)
llm = InternalChatLLM(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model_name=os.getenv("LLM_MODEL"),
    temperature=0,
)
# 임베딩: 로컬 HuggingFace 모델 (local_files_only=True → Hub 접속 차단)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    cache_folder=os.getenv("EMBED_CACHE_DIR", "./hf_cache"),
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

# (1) 로딩 + 청킹
pages = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
).split_documents(pages)
print(f"[Index] 페이지 {len(pages)}개 → 청크 {len(chunks)}개")

# (2) 벡터 DB: 있으면 재사용, 없으면 새로 임베딩
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print("[Index] 기존 Chroma DB 재사용 (임베딩 재계산 안 함)")
else:
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    print("[Index] 새 Chroma DB 생성 + 임베딩 저장")

# (4) RAG 체인: 사내 LLM 서버 + 검색 k=3개 + 근거 반환
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

# (5) 질문 루프
while True:
    q = input("\n질문 (exit 입력 시 종료): ").strip()
    if q.lower() in {"exit", "quit", ""}:
        break
    out = qa.invoke({"query": q})
    print(f"\n[답변]\n{out['result']}")
    print("\n[근거]")
    for d in out["source_documents"]:
        page = d.metadata.get("page", "?")
        snippet = d.page_content[:120].replace("\n", " ")
        print(f"  - p.{page}: {snippet}...")
```

### 3-3. 실행 방법

```bash
python 02_pdf_rag.py
```

### 3-4. 예상 출력 (첫 실행)

```
[Index] 페이지 27개 → 청크 184개
[Index] 새 Chroma DB 생성 + 임베딩 저장

질문 (exit 입력 시 종료): 이 문서에서 가장 중요한 개념은 무엇인가요?

[답변]
이 문서는 ... (PDF 내용에 따라 다름) ...

[근거]
  - p.3: Python is an easy to learn, powerful programming language ...
  - p.5: ... interpreter and the extensive standard library ...
  - p.12: ... built-in high level data types ...
```

### 3-5. 예상 출력 (두 번째 실행 — 임베딩 재사용)

```
[Index] 페이지 27개 → 청크 184개
[Index] 기존 Chroma DB 재사용 (임베딩 재계산 안 함)

질문 (exit 입력 시 종료): ...
```

> 같은 PDF에 대해 두 번째 실행부터는 임베딩 API 호출이 일어나지 않습니다. **비용·시간이 0이 되는 지점**입니다.

---

## 4. 코드 단계별 설명 (Python 개발자 / DA·K8S 경험 연결)

### 4-1. `PyPDFLoader().load()` — Load

- 각 페이지를 `Document(page_content=..., metadata={"source":..., "page":...})`로 반환합니다.
- **메타데이터가 핵심**: 나중에 "근거 페이지 번호" 표시는 이 `metadata["page"]` 덕분입니다.
- *DA 비유*: Elasticsearch에 `_source`로 원본을 같이 저장하는 패턴과 동일합니다. 벡터만으로는 사람이 못 읽으니까요.

### 4-2. `RecursiveCharacterTextSplitter` — Split (청킹)

- LLM 컨텍스트 윈도우는 유한합니다. PDF 전체를 그대로 넣을 수 없습니다.
- `chunk_size=500` (문자 단위, 토큰 아님!), `chunk_overlap=50`.
- `separators=["\n\n", "\n", ".", " "]`: 의미 단위가 큰 구분자부터 시도합니다. 문단 > 줄 > 문장 > 단어.
- **왜 overlap?** 청크 경계에서 잘리는 정보 손실을 방지. 보통 `chunk_size`의 10% 정도.
- *튜닝 포인트*: 기술 문서는 500~800, 대화록·블로그는 300~500이 흔합니다. Level 2에서 실험.

### 4-3. `HuggingFaceEmbeddings(cache_folder=..., local_files_only=True)` — Embed

- 청크 텍스트 → 1024차원 벡터 (BGE-M3 기준).
- `cache_folder`에 `snapshot_download(cache_dir=)`로 받아둔 모델을 로컬에서 로드합니다 — 외부 API 호출 없음, 비용 0.
- `local_files_only=True`가 핵심: 이 옵션 없이 실행하면 매번 HuggingFace Hub 접속을 시도해 타임아웃이 반복됩니다.
- LangChain이 내부적으로 배치(batch) 처리합니다.

### 4-4. `Chroma.from_documents(..., persist_directory=...)` — Store

- **DA 경험 연결**:
  - Elasticsearch ≈ "역색인(inverted index) + BM25" 기반 텍스트 검색
  - Chroma ≈ "임베딩 벡터 + ANN(근사 최근접 이웃)" 기반 의미 검색
  - 둘은 적이 아니라 **상호 보완재**입니다. Level 3에서 Hybrid Search로 합칩니다.
- `persist_directory="./chroma_db"`: 디스크에 영속화. 재실행 시 임베딩 비용 0.
- *K8S 비유*: PVC(PersistentVolumeClaim)와 같은 개념. 컨테이너(프로세스)가 죽어도 데이터는 남습니다.

### 4-5. `vectorstore.as_retriever(search_kwargs={"k": 3})` — Retrieve

- 질문도 같은 임베딩 모델로 벡터화 → Chroma에서 가장 가까운 `k=3` 청크 반환.
- `k`는 RAG에서 가장 자주 조정하는 파라미터입니다.
  - 작으면 → 노이즈는 적지만 정보 부족
  - 크면 → 정보는 많지만 LLM이 헷갈리고 비용·지연 증가
- *모니터링 비유*: AIOps에서 알람 임계값 튜닝과 동일한 trade-off입니다.

### 4-6. `RetrievalQA.from_chain_type(chain_type="stuff")` — Generate

- `stuff`: 검색된 k개 청크를 **하나의 프롬프트에 다 욱여넣고** LLM 호출.
  - 가장 단순·빠름. k가 작을 때 적합.
- 다른 옵션: `map_reduce`(청크별 요약 → 합치기), `refine`(반복 개선). Level 2에서 비교.
- `temperature=0`: 답변의 결정성 확보(같은 입력 → 같은 출력에 가까움). 평가·디버깅에 필수.

### 4-7. `return_source_documents=True` — 근거 추적

- RAG에서 **가장 자주 잊는데 가장 중요한** 옵션.
- 사용자 신뢰 + 디버깅(왜 이 답이 나왔는지)에 필수.
- 운영 관점: 모든 답변 로그에 근거 청크 ID를 같이 남기면 사후 분석이 가능합니다 (AIOps에서 trace_id 박는 것과 동일한 사고방식).

---

## 5. 자주 부딪히는 함정 (디버깅 체크리스트)

| 증상 | 원인 | 해결 |
|---|---|---|
| `requests.exceptions.ConnectionError` | `.env` 미로딩 또는 `LLM_BASE_URL` 오타 | `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('LLM_BASE_URL'))"` 확인 |
| `requests.exceptions.HTTPError: 401` | `LLM_API_KEY` 오류 | 사내 LLM 담당자에게 키 재발급 요청 |
| `OSError: model not found` | `EMBED_CACHE_DIR` 경로에 모델 없음 | `ls $EMBED_CACHE_DIR` 로 `models--BAAI--bge-m3/` 폴더 존재 확인 |
| `ImportError: langchain_chroma` | 패키지 미설치 | `pip install langchain-chroma` |
| 답변이 항상 "모르겠습니다" | 청크가 너무 작거나 k=1 | `chunk_size` ↑, `k=3~5`로 조정 |
| 답변에 PDF에 없는 내용 등장(환각) | 프롬프트가 LLM 자체 지식 허용 | Level 2의 커스텀 프롬프트로 "컨텍스트만 사용" 강제 |
| 두 번째 실행이 더 느림 | `persist_directory` 미지정 → 매번 재임베딩 | 코드의 분기(`if os.path.exists`) 확인 |
| 한글 PDF 인코딩 깨짐 | `PyPDFLoader`가 일부 한글 PDF 약함 | `PyPDFLoader` → `PyMuPDFLoader`(별도 설치: `pip install pymupdf`)로 교체 |
| 로컬 모델인데 실행마다 재다운로드 시도 | `local_files_only=True` 누락 또는 `cache_folder` 경로 불일치 | 1-4절 참고: `model_kwargs={"local_files_only": True}` 추가, `cache_folder`가 실제 모델 상위 디렉토리인지 확인 |
| `OSError: ... does not appear to have a file named config.json` | 절대 경로 오류 — 스냅샷 해시 하위까지 지정해야 할 수 있음 | 방법 B에서 경로를 `ls ~/models/bge-m3/` 로 확인 후 재지정 |

---

## 6. 학습 체크포인트

이번 실습이 끝나면 다음 질문에 즉답 가능해야 합니다.

- [ ] "임베딩이 뭐냐"고 누가 물어보면 → "텍스트를 N차원 벡터로 변환하는 함수. 의미가 가까우면 코사인 유사도가 높다."
- [ ] PDF 1개로 동작하는 Q&A 봇을 **빈 디렉토리에서 30분 안에 다시** 만들 수 있다.
- [ ] `persist_directory`가 왜 중요한지 (비용 측면) 설명 가능.
- [ ] 답변과 함께 **근거 페이지**가 같이 나온다.
- [ ] `chunk_size`, `chunk_overlap`, `k` 세 파라미터의 trade-off를 1분 안에 설명 가능.

---

## 7. 다음 단계 힌트 (Level 2 예고)

이 기본 RAG는 작동은 하지만, 운영하려고 보면 금방 한계가 드러납니다. **다음에 깎아낼 곳들**:

### 7-1. 검색 품질 문제

- 사용자가 "RAG 한계"라고만 쳐도 잘 찾아야 함 → **쿼리 리라이팅 / HyDE**
- 동의어·오타 강건성 → **BM25 + 벡터 하이브리드 검색** (DA 경험 직접 활용 지점!)
- 너무 유사한 청크 k개만 나옴 → **MMR(다양성 고려)**

### 7-2. 답변 품질 문제

- "모르면 모른다" 강제 → **커스텀 프롬프트 템플릿**
- 환각 검증 → **답변-근거 일치도 평가 (RAGAS의 faithfulness)**
- 멀티홉 질문(2단계 추론 필요) → **Re-ranking, Self-RAG**

### 7-3. 운영 문제

- 임베딩·LLM 호출 추적 → **LangSmith Tracing** (AIOps 경험과 직결: trace/span)
- 청크 업데이트 시 중복 방지 → **문서 ID 기반 upsert**
- 평가 자동화 → **Golden Q&A 셋 + RAGAS 점수 추적**

### 7-4. 인프라 측면 (K8S 경험 활용 지점)

- 로컬 Chroma → 원격 Chroma 서버 (`chromadb run`) → K8S StatefulSet 배포
- 임베딩 모델을 OpenAI에서 self-hosted(sentence-transformers, BGE-M3)로 전환 → GPU 노드 풀
- API 서버 래핑 (FastAPI) + Helm chart 배포

> **Level 1의 핵심 깨달음**: "RAG는 마법이 아니다. Load–Split–Embed–Store–Retrieve–Generate **6단계의 평범한 파이프라인**이고, 각 단계마다 조절 가능한 노브가 있다." 그 노브를 하나씩 만져보는 게 Level 2 이후의 여정입니다.

---

## 부록 A. 디렉토리 최종 모습

```
~/workspace/project/rag/_workspace/level1/
├── .venv/                       # 가상환경
├── .env                         # LLM_BASE_URL, LLM_API_KEY, EMBED_CACHE_DIR 등 (커밋 금지)
├── .gitignore
├── requirements.txt
├── internal_llm.py              # 사내 LLM/임베딩 공통 모듈
├── sample.pdf                   # 실습용 PDF
├── models/                      # 수동 설치한 HuggingFace 모델 (커밋 금지)
│   └── BAAI/
│       └── bge-m3/
├── 01_embedding_playground.py
├── 02_pdf_rag.py
└── chroma_db/                   # Chroma 영속 데이터 (자동 생성)
    ├── chroma.sqlite3
    └── ...
```

## 부록 B. 한 줄 요약 카드 (출퇴근용)

> **RAG = Indexing(Load→Split→Embed→Store) + Retrieval(질문 임베딩→k개 검색) + Generation(LLM에 컨텍스트+질문 같이 던지기). Chroma는 "임베딩을 저장하는 DB"이고, LangChain은 이 6단계를 묶어주는 접착제다.**
