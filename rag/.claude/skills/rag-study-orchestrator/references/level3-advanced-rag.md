# Level 3: 고급 RAG (3주)

## 목차
- [학습 목표](#학습-목표)
- [6주차: Self-RAG / CRAG](#6주차-self-rag--crag)
- [7주차: HyDE + Multi-hop RAG](#7주차-hyde--multi-hop-rag)
- [8주차: RAGAS 평가 프레임워크](#8주차-ragas-평가-프레임워크)
- [핵심 실습 프로젝트](#핵심-실습-프로젝트)
- [체크리스트](#체크리스트)

---

## 학습 목표

3주 후 달성 목표:
- Self-RAG로 검색 결과의 관련성을 LLM이 스스로 판단하는 파이프라인 구현
- HyDE로 쿼리 품질 개선, Multi-hop으로 복잡한 질문 처리
- RAGAS로 RAG 시스템 품질을 자동화 측정

---

## 6주차: Self-RAG / CRAG

### 기존 RAG의 문제

기본 RAG는 항상 검색 결과를 사용한다. 검색 결과가 질문과 관련 없어도 강제로 사용하여 "헛소리(hallucination)" 발생.

**Self-RAG 아이디어:**
1. 검색이 필요한가? (Retrieval Decision)
2. 검색 결과가 관련 있는가? (Relevance Grading)
3. 생성된 답변이 문서에 근거하는가? (Faithfulness Check)
4. 답변이 질문을 해결하는가? (Answer Grading)

### LangGraph 기반 Self-RAG

```python
# pip install langgraph langchain langchain-core langchain-community requests
import os
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from internal_llm import InternalChatLLM  # 공통 모듈

load_dotenv()

# --- 상태 정의 ---
class RAGState(TypedDict):
    question: str
    documents: List[str]
    generation: Optional[str]
    retrieval_needed: bool
    documents_relevant: bool
    answer_grounded: bool
    loop_count: int  # 무한 루프 방지

# --- LLM 그레이더 ---
llm = InternalChatLLM(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model_name=os.getenv("LLM_MODEL"),
    temperature=0,
)

def check_retrieval_needed(state: RAGState) -> RAGState:
    """이 질문이 검색을 필요로 하는가?"""
    prompt = ChatPromptTemplate.from_template(
        "다음 질문이 외부 문서 검색이 필요한가요? 단순한 계산이나 일반 상식은 불필요.\n"
        "질문: {question}\n"
        "답변 (yes/no만):"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    return {"retrieval_needed": "yes" in result.lower()}

def retrieve(state: RAGState) -> RAGState:
    """벡터 DB에서 관련 문서 검색"""
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

def grade_documents(state: RAGState) -> RAGState:
    """검색된 문서가 질문과 관련 있는가?"""
    prompt = ChatPromptTemplate.from_template(
        "질문: {question}\n"
        "문서: {document}\n\n"
        "이 문서가 질문에 답하는 데 관련 있나요? (yes/no만):"
    )
    chain = prompt | llm | StrOutputParser()
    
    # 모든 문서 평가, 관련 있는 것만 유지
    relevant_docs = []
    for doc in state["documents"]:
        result = chain.invoke({"question": state["question"], "document": doc})
        if "yes" in result.lower():
            relevant_docs.append(doc)
    
    return {
        "documents": relevant_docs,
        "documents_relevant": len(relevant_docs) > 0
    }

def generate(state: RAGState) -> RAGState:
    """관련 문서 기반 답변 생성"""
    context = "\n\n".join(state["documents"])
    prompt = ChatPromptTemplate.from_template(
        "다음 컨텍스트만 사용하여 질문에 답하세요.\n\n"
        "컨텍스트: {context}\n\n"
        "질문: {question}\n\n"
        "답변:"
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    return {"generation": generation}

def grade_generation(state: RAGState) -> RAGState:
    """생성된 답변이 문서에 근거하는가?"""
    prompt = ChatPromptTemplate.from_template(
        "문서: {documents}\n\n"
        "답변: {generation}\n\n"
        "이 답변이 오직 문서 내용에 근거하나요? (yes/no만):"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "documents": "\n".join(state["documents"]),
        "generation": state["generation"]
    })
    return {"answer_grounded": "yes" in result.lower()}

# --- 라우팅 함수 ---
def route_after_retrieval_check(state: RAGState) -> str:
    if state["retrieval_needed"]:
        return "retrieve"
    return "generate_without_retrieval"

def route_after_grading(state: RAGState) -> str:
    if state["documents_relevant"]:
        return "generate"
    elif state["loop_count"] < 2:
        return "retrieve"  # 재검색 시도
    return "generate"  # 2번 시도 후 포기

def route_after_generation(state: RAGState) -> str:
    if state["answer_grounded"] or state["loop_count"] >= 2:
        return END
    return "generate"  # 재생성 시도

# --- 그래프 구성 ---
workflow = StateGraph(RAGState)

workflow.add_node("check_retrieval", check_retrieval_needed)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation)

workflow.set_entry_point("check_retrieval")
workflow.add_conditional_edges("check_retrieval", route_after_retrieval_check)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_after_grading)
workflow.add_edge("generate", "grade_generation")
workflow.add_conditional_edges("grade_generation", route_after_generation)

app = workflow.compile()

# 실행
result = app.invoke({
    "question": "LangChain에서 RetrievalQA는 어떻게 사용하나요?",
    "documents": [],
    "generation": None,
    "retrieval_needed": False,
    "documents_relevant": False,
    "answer_grounded": False,
    "loop_count": 0
})
print(result["generation"])
```

### CRAG (Corrective RAG)

CRAG는 검색 결과가 불충분할 때 **웹 검색**으로 보완하는 패턴.

```python
from langchain_community.tools import TavilySearchResults

web_search = TavilySearchResults(max_results=3)

def crag_retrieve(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["question"])
    # 관련성 평가
    if all(doc_grade == "no" for doc_grade in grade_all(docs, state["question"])):
        # 벡터 DB 결과 불충분 → 웹 검색으로 보완
        web_results = web_search.invoke(state["question"])
        return {"documents": [r["content"] for r in web_results]}
    return {"documents": [d.page_content for d in docs]}
```

---

## 7주차: HyDE + Multi-hop RAG

### HyDE (Hypothetical Document Embeddings)

**문제:** 짧은 질문("RAG란?")의 임베딩과 긴 답변 문서의 임베딩은 공간이 다르다.
**해결:** 질문에 대한 가상의 답변을 생성 → 그 답변으로 검색.

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from internal_llm import InternalChatLLM, InternalEmbeddings  # 공통 모듈

load_dotenv()
llm = InternalChatLLM(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model_name=os.getenv("LLM_MODEL"),
    temperature=0,
)

# 1단계: 가상 답변 생성
hyde_prompt = ChatPromptTemplate.from_template(
    "다음 질문에 대한 가상의 학술 문서 단락을 작성하세요.\n"
    "실제 사실이 아니어도 됩니다. 검색에 사용할 가상의 내용입니다.\n\n"
    "질문: {question}\n\n"
    "가상 단락 (200-300자):"
)
hyde_chain = hyde_prompt | llm | StrOutputParser()

def hyde_retrieve(question: str, vectorstore, k: int = 5):
    # 가상 문서 생성
    hypothetical_doc = hyde_chain.invoke({"question": question})
    print(f"가상 문서: {hypothetical_doc[:100]}...")
    
    # 가상 문서의 임베딩으로 검색
    results = vectorstore.similarity_search(hypothetical_doc, k=k)
    return results

# 사용
results = hyde_retrieve("Self-RAG의 핵심 아이디어는 무엇인가요?", vectorstore)
```

### Multi-hop RAG

**문제:** "A의 창업자가 다니던 학교의 위치는?" 같은 다단계 추론 질문은 단일 검색으로 해결 불가.

```python
def multi_hop_retrieve(question: str, vectorstore, max_hops: int = 3):
    """질문을 분해하여 단계적으로 검색"""
    
    # 1단계: 질문 분해
    decompose_prompt = ChatPromptTemplate.from_template(
        "복잡한 질문을 순서대로 답해야 하는 단순한 하위 질문들로 분해하세요.\n"
        "질문: {question}\n\n"
        "하위 질문들 (번호 목록으로):"
    )
    decompose_chain = decompose_prompt | llm | StrOutputParser()
    sub_questions_text = decompose_chain.invoke({"question": question})
    
    # 하위 질문 파싱
    sub_questions = [
        line.strip("0123456789. ") 
        for line in sub_questions_text.split("\n") 
        if line.strip()
    ]
    
    # 2단계: 각 하위 질문 순차 검색
    context_so_far = ""
    all_docs = []
    
    for i, sub_q in enumerate(sub_questions[:max_hops]):
        # 이전 답변을 컨텍스트로 포함한 검색
        search_query = f"{sub_q}\n이전 컨텍스트: {context_so_far}" if context_so_far else sub_q
        docs = vectorstore.similarity_search(search_query, k=3)
        all_docs.extend(docs)
        
        # 중간 답변 생성 (다음 hop의 컨텍스트로 사용)
        if i < len(sub_questions) - 1:
            mid_answer = qa_chain.invoke({"query": sub_q, "context": docs})
            context_so_far = mid_answer["result"]
    
    return all_docs, context_so_far
```

---

## 8주차: RAGAS 평가 프레임워크

### RAGAS란?

RAG 시스템의 품질을 자동으로 측정하는 평가 프레임워크. LLM을 사용하여 4가지 지표를 측정한다.

```
Faithfulness:      답변이 검색 문서에 얼마나 충실한가? (hallucination 측정)
Answer Relevancy:  답변이 질문에 얼마나 관련 있는가?
Context Recall:    정답에 필요한 정보가 검색 문서에 있는가?
Context Precision: 검색 문서 중 실제 답변에 사용된 비율
```

### RAGAS 평가 실행

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# 평가 데이터셋 구성 (수동 또는 자동 생성)
eval_dataset = {
    "question": [
        "RAG에서 청킹이 중요한 이유는?",
        "하이브리드 검색이란 무엇인가?",
        "RAGAS는 무엇을 측정하는가?",
    ],
    "answer": [
        # RAG 시스템이 생성한 답변
        generated_answers[0],
        generated_answers[1],
        generated_answers[2],
    ],
    "contexts": [
        # 각 질문에 검색된 문서 목록
        [doc.page_content for doc in retrieved_docs[0]],
        [doc.page_content for doc in retrieved_docs[1]],
        [doc.page_content for doc in retrieved_docs[2]],
    ],
    "ground_truth": [
        # 정답 (수동 작성)
        "청킹은 검색 단위를 결정하므로...",
        "벡터 검색과 BM25를 결합하여...",
        "Faithfulness, Answer Relevancy, Context Recall, Context Precision을 측정한다.",
    ]
}

dataset = Dataset.from_dict(eval_dataset)

# 평가 실행 — 사내 LLM/임베딩 인스턴스 전달
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=llm,           # InternalChatLLM 인스턴스
    embeddings=embeddings,  # InternalEmbeddings 인스턴스
)

print(result)
# Output 예시:
# {'faithfulness': 0.8833, 'answer_relevancy': 0.8012, 'context_recall': 0.8667, 'context_precision': 0.8500}
```

### RAGAS 결과 해석 및 개선 방향

| 지표 낮을 때 | 원인 | 개선 방법 |
|-----------|------|---------|
| Faithfulness 낮음 | 답변이 문서에 없는 내용 포함 (hallucination) | 프롬프트에 "컨텍스트만 사용" 강조 |
| Answer Relevancy 낮음 | 답변이 질문에서 벗어남 | 프롬프트 개선, 온도 낮춤 |
| Context Recall 낮음 | 검색이 정답 문서를 놓침 | 청킹 전략 변경, 하이브리드 검색 |
| Context Precision 낮음 | 관련 없는 문서가 많이 검색됨 | k 줄이기, Reranking 추가 |

### 자동화 평가 파이프라인

```python
import pandas as pd
from datetime import datetime

def run_rag_evaluation(qa_chain, retriever, eval_questions, ground_truths):
    """RAG 시스템 전체 평가 파이프라인"""
    answers, contexts = [], []
    
    for question in eval_questions:
        # 답변 및 컨텍스트 생성
        result = qa_chain.invoke({"query": question})
        answers.append(result["result"])
        contexts.append([d.page_content for d in result["source_documents"]])
    
    # RAGAS 평가
    dataset = Dataset.from_dict({
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    df = pd.DataFrame([dict(scores)])
    df.to_csv(f"ragas_results_{timestamp}.csv", index=False)
    
    return scores

# 개선 전후 비교
baseline_scores = run_rag_evaluation(basic_chain, basic_retriever, questions, truths)
improved_scores = run_rag_evaluation(improved_chain, hybrid_retriever, questions, truths)

print("개선 전:", baseline_scores)
print("개선 후:", improved_scores)
```

---

## 핵심 실습 프로젝트

**"RAGAS로 측정한 RAG 품질 개선 실험"**

1. Level 2 하이브리드 RAG를 베이스라인으로 RAGAS 측정
2. Self-RAG 적용 후 재측정 (Faithfulness 개선 확인)
3. HyDE 적용 후 재측정 (Context Recall 개선 확인)
4. 개선 전후 비교 보고서 작성

**기대 개선폭:**
- Faithfulness: 0.75 → 0.88 (Self-RAG)
- Context Recall: 0.72 → 0.85 (HyDE + Hybrid)

---

## 체크리스트

**개념 이해**
- [ ] Self-RAG의 4단계 판단 과정 설명 가능
- [ ] HyDE가 왜 검색 품질을 높이는지 설명 가능
- [ ] RAGAS 4개 지표가 각각 무엇을 측정하는지 설명

**구현 능력**
- [ ] LangGraph StateGraph 구성 가능
- [ ] Self-RAG 파이프라인 완성 (검색 결정, 관련성 평가, 답변 평가)
- [ ] RAGAS 평가 파이프라인 완성

**실험 결과**
- [ ] 3가지 RAG 전략 RAGAS 점수 비교 완료
- [ ] Faithfulness 0.85 이상 달성
- [ ] 개선 보고서 작성 완료
