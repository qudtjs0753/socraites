---
name: rag-implementation-guide
description: "Python으로 RAG 파이프라인을 구현하는 가이드 스킬. LangChain, LlamaIndex, 임베딩, RAG 코드, Self-RAG, CRAG, HyDE, RAGAS 평가, 파이썬 RAG 예제, 랭체인 튜토리얼, 라마인덱스 실습 요청 시 반드시 이 스킬을 사용하라. '코드 보여줘', '어떻게 구현해', 'LangChain으로 RAG', 'RAGAS로 평가' 등 모든 RAG Python 구현 요청에 이 스킬을 사용."
---

# RAG Python 구현 가이드 스킬

LangChain과 LlamaIndex를 활용한 RAG 파이프라인 구현 가이드.

## Level 1: 기초 RAG 구현

### 핵심 컴포넌트

```
문서 → [Document Loader] → [Text Splitter] → [Embedding] → [Vector Store]
                                                                    ↓
질문 → [Embedding] → [Similarity Search] → [Retrieved Docs] → [LLM] → 답변
```

### 최소 RAG 파이프라인 (LangChain)

```python
# pip install langchain langchain-community langchain-chroma chromadb requests python-dotenv
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from internal_llm import InternalChatLLM, InternalEmbeddings  # 공통 모듈

load_dotenv()

llm = InternalChatLLM(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model_name=os.getenv("LLM_MODEL"),
    temperature=0,
)
embeddings = InternalEmbeddings(
    base_url=os.getenv("EMBED_BASE_URL"),
    api_key=os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", "")),
    model=os.getenv("EMBED_MODEL"),
)

# 1. 문서 로딩
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. 청킹 — chunk_size와 overlap은 문서 특성에 맞게 조정
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 임베딩 + 벡터스토어 저장
vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
)

# 5. 질문
result = qa_chain.invoke("RAG란 무엇인가요?")
print(result["result"])
```

## Level 2: 하이브리드 검색 구현

### BM25 + 벡터 앙상블

```python
# pip install rank-bm25 langchain-community
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 벡터 리트리버 (이미 존재하는 vectorstore 사용)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 리트리버 (렉시컬 검색)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 앙상블 — 벡터 60%, BM25 40% 가중치
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)

results = ensemble_retriever.invoke("한국어 문서 검색")
```

## Level 3: 고급 RAG 패턴

### Self-RAG (LangGraph 활용)

```python
# pip install langgraph
from langgraph.graph import StateGraph
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_relevant: bool

def retrieve(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

def grade_documents(state: RAGState) -> RAGState:
    # 검색 문서가 질문과 관련 있는지 평가
    relevant = grader_llm.invoke(
        f"질문: {state['question']}\n문서: {state['documents'][0]}\n관련성: yes/no"
    )
    return {"is_relevant": "yes" in relevant.content.lower()}

def generate(state: RAGState) -> RAGState:
    answer = qa_chain.invoke({
        "question": state["question"],
        "context": "\n".join(state["documents"])
    })
    return {"generation": answer}

# 그래프 구성
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)
workflow.add_conditional_edges(
    "grade",
    lambda s: "generate" if s["is_relevant"] else "retrieve",
)
app = workflow.compile()
```

### HyDE (Hypothetical Document Embeddings)

```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 가상 문서 생성 (질문에 대한 이상적인 답변처럼 작성)
hyde_prompt = PromptTemplate.from_template(
    "다음 질문에 대한 가상의 학술 문서 단락을 작성하세요:\n질문: {question}\n단락:"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

hypothetical_doc = hyde_chain.invoke({"question": "RAG의 한계는?"})

# 2. 가상 문서로 벡터 검색 (실제 질문 대신)
docs = vectorstore.similarity_search(hypothetical_doc, k=4)
```

## Level 3: RAGAS 평가

```python
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

# 평가 데이터셋 구성
eval_data = {
    "question": ["RAG란?", "임베딩이란?"],
    "answer": [generated_answers[0], generated_answers[1]],
    "contexts": [retrieved_contexts[0], retrieved_contexts[1]],
    "ground_truth": ["RAG는...", "임베딩은..."]
}

dataset = Dataset.from_dict(eval_data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(result)
# Output: {'faithfulness': 0.85, 'answer_relevancy': 0.78, 'context_recall': 0.90}
```

## 구현 시 공통 체크리스트

- [ ] `.env` 파일에 `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL` 설정
- [ ] `internal_llm.py` (InternalChatLLM + InternalEmbeddings) 공통 모듈 준비
- [ ] 청크 크기 실험 (200~1000 토큰)
- [ ] `k` 파라미터 조정 (검색 결과 수)
- [ ] 벡터스토어 영속성 설정 (`persist_directory`)

## 관련 references

- Level별 상세 가이드는 `../rag-study-orchestrator/references/` 참조
