# Level 1 실습 — 기초 RAG

## 파일 구성

| 파일 | 설명 |
|------|------|
| `embedding_playground.py` | 임베딩 + 코사인 유사도 체험 |
| `rag_pipeline.py` | 다중 파일 형식 → Chroma → Q&A 봇 |
| `test_loaders.py` | 파일 로더 단위 테스트 |
| `requirements.txt` | 패키지 목록 |

### 지원 파일 형식

| 형식 | 확장자 |
|------|--------|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| 텍스트 | `.txt`, `.md`, `.log` |
| 이미지(OCR) | `.jpg`, `.jpeg`, `.png` |

> PDF는 지원하지 않습니다. 필요 시 별도 추가.

---

## 빠른 시작

```bash
# 1. 가상환경 생성 + 활성화
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 열어서 사내 LLM 서버 정보 + 모델 경로 입력
```

---

## 1. 셋업

### 1-1. BGE-M3 모델 설치 (외부 PC → 폐쇄망 전달)

**외부 PC (인터넷 가능 환경)에서 실행:**

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', cache_dir='./hf_cache')
"

tar -czf bge-m3.tar.gz ./hf_cache
```

> `bge-m3.tar.gz` 파일을 메일 또는 파일 서버를 통해 폐쇄망으로 전달합니다.

**폐쇄망 PC (이 디렉토리에서)에서 실행:**

```bash
tar -xzf bge-m3.tar.gz
```

해제 후 디렉토리 구조:

```
level1/
└── hf_cache/
    └── models--BAAI--bge-m3/
        └── snapshots/{hash}/
            ├── config.json
            └── ...
```

모델 로드 검증:

```bash
python -c "
import os
os.environ['EMBED_MODEL'] = 'BAAI/bge-m3'
os.environ['EMBED_CACHE_DIR'] = './hf_cache'
from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', cache_folder='./hf_cache', model_kwargs={'local_files_only': True})
print('차원:', len(emb.embed_query('테스트')))  # 1024 출력되면 정상
"
```

### 1-2. 가상환경 + 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1-3. 환경변수 설정

`.env` 파일 생성 후 사내 값으로 수정:

```
LLM_BASE_URL=http://사내-llm-서버/v1
LLM_API_KEY=사내키
LLM_MODEL=모델명
EMBED_MODEL=BAAI/bge-m3
EMBED_CACHE_DIR=./hf_cache
```

---

## 2. 테스트

파일 로더 단위 테스트는 LLM/임베딩 모델 없이 실행 가능합니다.

```bash
# 전체 테스트
pytest test_loaders.py -v

# 특정 클래스만
pytest test_loaders.py::TestLoadCsv -v
pytest test_loaders.py::TestLoadExcel -v
pytest test_loaders.py::TestLoadText -v
pytest test_loaders.py::TestLoadImage -v
pytest test_loaders.py::TestLoadDataDir -v
```

### 테스트 항목

| 클래스 | 테스트 수 | 내용 |
|--------|----------|------|
| `TestLoadCsv` | 9 | 기본 변환, 메타데이터, 빈 값 제거, 인코딩(utf-8/euc-kr) |
| `TestLoadExcel` | 5 | 단일/다중 시트, None 셀, 빈 시트 처리 |
| `TestLoadText` | 5 | TXT/MD/LOG, cp949, 빈 파일 처리 |
| `TestLoadImage` | 3 | OCR 추출, 빈 이미지, pytesseract 미설치 예외 |
| `TestLoadDataDir` | 3 | 혼합 파일 일괄 로드, 미지원 확장자 무시 |

> `TestLoadImage`는 `pytesseract`가 설치되지 않은 환경에서 자동으로 skip됩니다.

---

## 3. 실습 1 — 임베딩 체험

```bash
python embedding_playground.py
```

임베딩 차원 수와 문장 간 코사인 유사도가 출력됩니다.

---

## 4. 실습 2 — 다중 파일 Q&A 봇

`data/` 디렉토리에 지원 파일을 준비합니다:

```bash
mkdir -p data
cp /path/to/데이터.csv data/
cp /path/to/데이터.xlsx data/
cp /path/to/문서.txt data/
```

실행:

```bash
python rag_pipeline.py
```

- 첫 실행: 파일 로드 → 청킹 → 임베딩 → `chroma_db/`에 저장
- 재실행: `chroma_db/` 재사용 (임베딩 재계산 없음)

인덱스 초기화:

```bash
rm -rf chroma_db/
python rag_pipeline.py
```
