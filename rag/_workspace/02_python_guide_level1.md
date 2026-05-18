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
langchain-openai==0.2.8
langchain-huggingface==0.1.2
langchain-community==0.3.7
langchain-chroma==0.1.4
sentence-transformers==3.3.1
chromadb==0.5.20
pypdf==5.1.0
python-dotenv==1.0.1
numpy==1.26.4
EOF

pip install -r requirements.txt
```

> K8S 비유로 보면 `requirements.txt`는 컨테이너 이미지 태그 핀(pin)과 같은 역할입니다. **재현 가능한 환경 = 디버깅 가능한 환경**입니다.

### 1-3. `.env` 파일 설정

LLM/임베딩 설정을 코드에 직접 박지 않고 환경 변수로 분리합니다.

```bash
cat > .env <<'EOF'
LLM_BASE_URL=http://사내-llm-서버/v1
LLM_API_KEY=사내키
LLM_MODEL=모델명
EMBED_MODEL=BAAI/bge-m3
EMBED_CACHE_DIR=./models   # python 실행 디렉토리 기준 상대경로
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
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('LLM_BASE_URL') else 'FAIL')"
# 출력: OK
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

# EMBED_CACHE_DIR: 수동 설치한 모델 경로 (python 실행 디렉토리 기준 상대경로)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    cache_folder=os.getenv("EMBED_CACHE_DIR", "./models"),
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
- **차원 수(1536)는 모델이 정함**: 컬렉션을 만들 때 한 번 정해지면 같은 모델만 사용해야 함 (DA 경험과 동일: 스키마 호환성).
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
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

PDF_PATH = "sample.pdf"
PERSIST_DIR = "./chroma_db"

# (1) 로딩 + 청킹
pages = PyPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
).split_documents(pages)
print(f"[Index] 페이지 {len(pages)}개 → 청크 {len(chunks)}개")

# (2) 임베딩: EMBED_CACHE_DIR에 수동 설치한 모델 로드 (python 실행 디렉토리 기준)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    cache_folder=os.getenv("EMBED_CACHE_DIR", "./models"),
)

# (3) 벡터 DB: 있으면 재사용, 없으면 새로 임베딩
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    print("[Index] 기존 Chroma DB 재사용 (임베딩 재계산 안 함)")
else:
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    print("[Index] 새 Chroma DB 생성 + 임베딩 저장")

# (4) RAG 체인: 사내 LLM 서버 + 검색 k=3개 + 근거 반환
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        temperature=0,
    ),
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

### 4-3. `HuggingFaceEmbeddings(cache_folder=...)` — Embed

- 청크 텍스트 → 1024차원 벡터 (BGE-M3 기준).
- `cache_folder`에 수동 설치된 모델을 로컬에서 로드합니다 — 외부 API 호출 없음, 비용 0.
- LangChain이 내부적으로 배치(batch) 처리합니다.

> **오프라인 모델 설치 필요** — 외부 PC에서 다운로드 후 폐쇄망으로 전달:
> ```bash
> # 외부 PC (인터넷 가능 환경)
> pip install huggingface_hub
> python -c "
> from huggingface_hub import snapshot_download
> snapshot_download('BAAI/bge-m3', local_dir='./models/BAAI/bge-m3')
> "
> # ./models/ 를 압축하여 전달
> ```

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
| `ConnectionError` / `401` | `.env` 미로딩 또는 `LLM_BASE_URL`/`LLM_API_KEY` 오타 | `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('LLM_BASE_URL'), os.getenv('LLM_API_KEY'))"` 확인 |
| `OSError: model not found` | `EMBED_CACHE_DIR` 경로에 모델 없음 | `ls $EMBED_CACHE_DIR` 로 모델 폴더 존재 확인 |
| `ImportError: langchain_chroma` | 패키지 미설치 | `pip install langchain-chroma` |
| 답변이 항상 "모르겠습니다" | 청크가 너무 작거나 k=1 | `chunk_size` ↑, `k=3~5`로 조정 |
| 답변에 PDF에 없는 내용 등장(환각) | 프롬프트가 LLM 자체 지식 허용 | Level 2의 커스텀 프롬프트로 "컨텍스트만 사용" 강제 |
| 두 번째 실행이 더 느림 | `persist_directory` 미지정 → 매번 재임베딩 | 코드의 분기(`if os.path.exists`) 확인 |
| 한글 PDF 인코딩 깨짐 | `PyPDFLoader`가 일부 한글 PDF 약함 | `PyPDFLoader` → `PyMuPDFLoader`(별도 설치: `pip install pymupdf`)로 교체 |

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
