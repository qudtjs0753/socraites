"""
실습 2: PDF → Chroma → Q&A 봇
핵심 개념: Load → Split → Embed → Store → Retrieve → Generate
실행: python 02_pdf_rag.py
준비: data/ 디렉토리에 PDF 파일 1개 이상 배치
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PDF_DIR = "./data"
PERSIST_DIR = "./chroma_db"

# (1) PDF 로딩 + 청킹
pdf_files = glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True) + glob.glob(f"{PDF_DIR}/*.pdf")
if not pdf_files:
    raise FileNotFoundError(f"{PDF_DIR}/ 디렉토리에 PDF 파일이 없습니다.")

pages = []
for path in pdf_files:
    pages.extend(PyPDFLoader(path).load())

chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
).split_documents(pages)
print(f"[Index] PDF {len(pdf_files)}개 / 페이지 {len(pages)}개 → 청크 {len(chunks)}개")

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

# (4) 커스텀 프롬프트: 컨텍스트 외 답변 금지 + 출처 표기
prompt = PromptTemplate(
    template="""다음 컨텍스트만 사용하여 한국어로 답하세요.

컨텍스트:
{context}

질문: {question}

규칙:
- 컨텍스트에 없으면 "문서에서 찾을 수 없습니다."라고만 답하세요.
- 답변 끝에 [출처: 페이지 번호] 형식으로 근거를 표시하세요.

답변:""",
    input_variables=["context", "question"],
)

# (5) RAG 체인: 사내 LLM 서버 연결
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        temperature=0,
    ),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# (6) 질문 루프
print("\nPDF Q&A 봇 준비 완료. 질문을 입력하세요.")
while True:
    q = input("\n질문 (exit 입력 시 종료): ").strip()
    if q.lower() in {"exit", "quit", ""}:
        break
    out = qa.invoke({"query": q})
    print(f"\n[답변]\n{out['result']}")
    print("\n[근거]")
    for doc in out["source_documents"]:
        page = doc.metadata.get("page", "?")
        source = os.path.basename(doc.metadata.get("source", "?"))
        snippet = doc.page_content[:100].replace("\n", " ")
        print(f"  - {source} p.{page}: {snippet}...")
