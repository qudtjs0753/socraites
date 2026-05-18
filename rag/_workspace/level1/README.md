# Level 1 실습 — 기초 RAG

## 빠른 시작 (회사 첫 실행)

```bash
# 1. 가상환경 생성 + 활성화
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 열어서 사내 LLM 서버 정보 + 모델 경로 입력

# 4. CSV 로딩 테스트 (모델 없어도 실행 가능)
pytest test_load_csv.py -v
```

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `01_embedding_playground.py` | 임베딩 + 코사인 유사도 체험 |
| `02_csv_excel_rag.py` | CSV/Excel → Chroma → Q&A 봇 |

---

## 1. 최초 셋업

### 1-1. BGE-M3 모델 설치 (외부 PC → 폐쇄망 전달)

인터넷 가능한 외부 PC에서 실행:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='./models/BAAI/bge-m3')
"
# ./models/ 압축 후 폐쇄망으로 전달
```

전달받은 `models/` 디렉토리를 이 디렉토리(`level1/`) 아래에 배치:

```
level1/
└── models/
    └── BAAI/
        └── bge-m3/
```

### 1-2. 가상환경 + 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1-3. 환경변수 설정

```bash
cp .env.example .env
```

`.env`를 열어 사내 값으로 수정:

```
LLM_BASE_URL=http://사내-llm-서버/v1
LLM_API_KEY=사내키
LLM_MODEL=모델명
EMBED_MODEL=BAAI/bge-m3
EMBED_CACHE_DIR=./models
```

> direnv 사용 시 `direnv allow` 한 번 실행하면 디렉토리 진입 시 `.env` + 가상환경이 자동 활성화됩니다.

---

## 2. 실습 1 — 임베딩 체험

```bash
python 01_embedding_playground.py
```

임베딩 차원 수와 문장 간 코사인 유사도가 출력됩니다. 의미가 가까운 문장일수록 점수가 높게 나오면 정상입니다.

---

## 3. 실습 2 — CSV/Excel Q&A 봇

CSV 또는 Excel 파일을 `data/` 디렉토리에 준비합니다:

```bash
mkdir -p data
cp /path/to/데이터.csv data/
cp /path/to/데이터.xlsx data/
```

실행:

```bash
python 02_csv_excel_rag.py
```

- 첫 실행: 각 행을 청킹 + 임베딩 후 `chroma_db/`에 저장
- 두 번째 실행부터: `chroma_db/` 재사용 (임베딩 재계산 없음)

파일을 교체하거나 다시 인덱싱하려면 `chroma_db/`를 삭제하고 재실행합니다:

```bash
rm -rf chroma_db/
python 02_csv_excel_rag.py
```
