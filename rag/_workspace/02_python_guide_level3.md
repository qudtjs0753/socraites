# Level 3: 고급 RAG — Self-RAG, HyDE, Multi-hop, RAGAS 평가

> **환경**: Python 3.10+, LangChain, LangGraph, Chroma, Ollama 로컬 LLM  
> **기간**: 3주  
> **목표**: Self-RAG + HyDE + Multi-hop 구현, RAGAS로 자동 품질 측정

---

## 목차

1. [Level 3 시작 전 — Level 2의 한계](#1-level-3-시작-전--level-2의-한계)
2. [LangGraph 기초 — 상태 기계 개념](#2-langgraph-기초--상태-기계-개념)
3. [Self-RAG — 검색 결과를 스스로 판단](#3-self-rag--검색-결과를-스스로-판단)
4. [CRAG — Corrective RAG](#4-crag--corrective-rag)
5. [HyDE — Hypothetical Document Embeddings](#5-hyde--hypothetical-document-embeddings)
6. [Multi-hop RAG — 다단계 추론](#6-multi-hop-rag--다단계-추론)
7. [RAGAS 평가 프레임워크](#7-ragas-평가-프레임워크)
8. [RAGAS 자동화 평가 파이프라인](#8-ragas-자동화-평가-파이프라인)
9. [전체 실습 프로젝트: RAG 품질 개선 실험](#9-전체-실습-프로젝트-rag-품질-개선-실험)
10. [테스트 작성](#10-테스트-작성)
11. [체크리스트 및 다음 단계](#11-체크리스트-및-다음-단계)

---

## 1. Level 3 시작 전 — Level 2의 한계

### 1.1 여전히 남은 문제들

Level 2에서 하이브리드 검색과 Reranking으로 많은 개선을 했지만, 다음 문제들이 남아 있다.

**문제 1: 관련 없는 검색 결과를 무조건 사용**

```
상황:
  문서: "사내 장애 보고서 2024년 1~6월"
  질문: "2024년 7월 서버 장애 내역은?"

Level 2의 동작:
  1. 하이브리드 검색: 관련성이 낮은 6월 장애 보고서 검색
  2. Reranking: "6월 장애 보고서"가 3위로 선정
  3. LLM: 이 문서로 "7월 장애 내역"을 답변 → 잘못된 답변 (hallucination)

원하는 동작:
  "제공된 문서에 2024년 7월 데이터가 없습니다."
```

**문제 2: 짧은 쿼리의 임베딩 품질 저하**

```
질문 임베딩 vs 문서 임베딩의 공간 불일치:
  
  질문 벡터: "HyDE란?"
  → 매우 짧고 추상적 → 벡터 공간에서 불안정
  
  문서 벡터: "HyDE(Hypothetical Document Embeddings)는 짧은 질문의 
              임베딩 품질을 개선하기 위해 가상의 답변을 생성하여..."
  → 긴 문장 → 안정적인 벡터
  
  결과: 질문 "HyDE란?"의 벡터가 관련 문서 벡터와 
        예상보다 거리가 멀 수 있음
```

**문제 3: 다단계 추론 불가**

```
질문: "우리 팀에서 가장 많은 장애를 일으킨 서비스의 담당자는 누구인가?"

필요한 추론 단계:
  1단계: "어떤 서비스가 가장 많은 장애를 일으켰는가?" 검색
  2단계: "그 서비스의 담당자는 누구인가?" 검색
  
  단일 검색으로는 해결 불가 — 두 단계가 순차적으로 필요
```

**문제 4: 품질 측정 불가**

```
RAG 개선 작업을 하고 있는데...
  "오늘 변경 사항이 실제로 품질을 개선했는가?"
  "Faithfulness가 0.75에서 0.88로 올랐는가?"
  
이를 객관적으로 측정하는 방법이 없음 → RAGAS 필요
```

### 1.2 Level 3 해결 로드맵

| 문제 | Level 3 해결책 |
|------|--------------|
| 관련 없는 검색 결과 사용 | Self-RAG (검색 결과 관련성 자체 평가) |
| 짧은 쿼리 임베딩 불안정 | HyDE (가상 답변으로 임베딩) |
| 다단계 추론 불가 | Multi-hop RAG (질문 분해 + 순차 검색) |
| 품질 측정 불가 | RAGAS (Faithfulness, Recall, Precision, Relevancy) |

---

## 2. LangGraph 기초 — 상태 기계 개념

### 2.1 왜 LangGraph가 필요한가?

Self-RAG는 단순한 선형 파이프라인(`질문 → 검색 → 생성`)이 아니다. 조건에 따라 흐름이 달라지는 **그래프** 구조다:

```
                  ┌──────────────┐
                  │ check_needed │  ← 검색 필요한가?
                  └──────┬───────┘
                    yes  │  no
         ┌───────────────┘  └─────────────────┐
         ▼                                    ▼
   ┌───────────┐                    ┌──────────────────┐
   │  retrieve  │                    │ generate_direct  │
   └─────┬─────┘                    └──────────────────┘
         │
         ▼
   ┌─────────────┐
   │ grade_docs  │  ← 문서가 관련 있는가?
   └─────┬───────┘
    yes  │  no
         │  └─────→ retrieve (재검색)
         ▼
   ┌──────────────┐
   │   generate   │
   └──────┬───────┘
          │
          ▼
   ┌─────────────────┐
   │ grade_generation │  ← 답변이 근거 있는가?
   └────────┬────────┘
      yes   │  no
            │  └─────→ generate (재생성)
            ▼
           END
```

이런 복잡한 흐름을 일반 Python 코드로 구현하면 읽기 어렵고 유지보수가 힘들다. LangGraph는 이를 **노드(node)**와 **엣지(edge)**로 명시적으로 표현한다.

### 2.2 LangGraph 핵심 개념

```python
# pip install langgraph
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# 1. State: 파이프라인 전체에서 공유하는 상태
class MyState(TypedDict):
    question: str           # 변하지 않음
    documents: List[str]    # 검색 과정에서 업데이트
    answer: Optional[str]   # 생성 후 설정
    loop_count: int         # 무한 루프 방지

# 2. Node: 상태를 입력받아 업데이트된 상태를 반환하는 함수
def my_node(state: MyState) -> MyState:
    # state 읽기
    question = state["question"]
    
    # 작업 수행
    result = do_something(question)
    
    # 업데이트할 필드만 반환
    return {"documents": result}

# 3. 그래프 구성
workflow = StateGraph(MyState)
workflow.add_node("my_node", my_node)
workflow.set_entry_point("my_node")
workflow.add_edge("my_node", END)

# 4. 컴파일 (실행 가능한 앱으로 변환)
app = workflow.compile()

# 5. 실행
result = app.invoke({
    "question": "질문 텍스트",
    "documents": [],
    "answer": None,
    "loop_count": 0,
})
```

**조건부 엣지 (분기):**

```python
def decide_next_step(state: MyState) -> str:
    """
    현재 상태를 보고 다음 노드 이름을 반환한다.
    """
    if state["documents"]:
        return "generate"     # 문서가 있으면 생성
    else:
        return "retrieve"     # 없으면 재검색

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "check_docs",           # 이 노드 실행 후
    decide_next_step,       # 이 함수로 다음 노드 결정
    {
        "generate": "generate",    # 반환값 → 노드 이름
        "retrieve": "retrieve",
    }
)
```

---

## 3. Self-RAG — 검색 결과를 스스로 판단

### 3.1 Self-RAG 4단계 이론

원래 논문(2023, Asai et al.)의 핵심 아이디어:

```
기존 RAG:   질문 → 무조건 검색 → 무조건 LLM → 답변

Self-RAG:   질문
              │
              ▼
           [판단 1] 검색이 필요한가?
           "2+2는?" → NO → 직접 답변
           "서버 장애 기록은?" → YES → 검색
              │
              ▼ (검색 결과)
           [판단 2] 이 문서들이 관련 있는가?
           각 문서를 개별 평가: "yes/no"
           관련 없는 문서 제거
              │
              ▼ (관련 문서만)
           [생성] 관련 문서로 답변 생성
              │
              ▼
           [판단 3] 이 답변이 문서에 근거하는가? (Faithfulness)
           "no" → 재생성 (최대 2회)
              │
              ▼
           [판단 4] 이 답변이 질문을 해결하는가? (Utility)
           최종 답변 출력
```

### 3.2 Self-RAG 완전 구현

```python
# 파일: self_rag.py
# 목적: LangGraph 기반 Self-RAG 파이프라인
# pip install langgraph langchain langchain-core

import os
from typing import TypedDict, List, Optional, Annotated
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

from ollama_llm import create_llm, create_embeddings
from loaders import load_directory
from chunking_strategies import strategy_fixed_size

load_dotenv()


# ─────────────────────────────────────────────
# 1. 상태 정의
# ─────────────────────────────────────────────

class SelfRAGState(TypedDict):
    """Self-RAG 파이프라인 전체에서 공유하는 상태"""
    question: str                       # 원본 질문 (변하지 않음)
    documents: List[str]               # 검색된 문서 내용 목록
    document_sources: List[str]        # 문서 출처 목록 (metadata)
    generation: Optional[str]          # 생성된 답변
    retrieval_needed: bool             # 검색이 필요한가?
    documents_relevant: bool           # 문서가 관련 있는가?
    answer_grounded: bool              # 답변이 근거 있는가?
    loop_count: int                    # 재시도 횟수 (무한 루프 방지)
    final_answer: Optional[str]        # 최종 답변


# ─────────────────────────────────────────────
# 2. LLM 그레이더 함수들
# ─────────────────────────────────────────────

def build_nodes(retriever, llm):
    """
    Self-RAG의 각 노드(함수)를 생성한다.

    Args:
        retriever: Chroma 또는 앙상블 리트리버
        llm: OllamaLLM 인스턴스
    """

    # ─────────────────────────────────────────
    # Node 1: 검색 필요 여부 판단
    # ─────────────────────────────────────────

    retrieval_check_prompt = ChatPromptTemplate.from_template("""다음 질문이 외부 문서 검색을 필요로 하는지 판단하세요.

판단 기준:
- 사실/수치 확인이 필요한 질문 → 검색 필요 (yes)
- 사내 정보, 장애 기록, 특정 데이터 → 검색 필요 (yes)
- 단순 계산, 일반 상식 → 검색 불필요 (no)

질문: {question}

답변 (yes 또는 no만 출력):""")

    retrieval_check_chain = retrieval_check_prompt | llm | StrOutputParser()

    def check_retrieval_needed(state: SelfRAGState) -> SelfRAGState:
        """이 질문이 문서 검색을 필요로 하는가?"""
        result = retrieval_check_chain.invoke({"question": state["question"]})
        needed = "yes" in result.lower().strip()

        print(f"  [검색 필요?] {state['question'][:40]}... → {'필요' if needed else '불필요'}")

        return {
            "retrieval_needed": needed,
            "loop_count": state.get("loop_count", 0),
        }

    # ─────────────────────────────────────────
    # Node 2: 문서 검색
    # ─────────────────────────────────────────

    def retrieve(state: SelfRAGState) -> SelfRAGState:
        """벡터 DB / 앙상블 리트리버에서 관련 문서 검색"""
        docs = retriever.invoke(state["question"])

        doc_contents = [d.page_content for d in docs]
        doc_sources = [
            d.metadata.get("source", "unknown") for d in docs
        ]

        print(f"  [검색] {len(docs)}개 문서 검색됨")

        return {
            "documents": doc_contents,
            "document_sources": doc_sources,
            "loop_count": state.get("loop_count", 0) + 1,
        }

    # ─────────────────────────────────────────
    # Node 3: 문서 관련성 평가
    # ─────────────────────────────────────────

    grade_doc_prompt = ChatPromptTemplate.from_template("""다음 질문과 문서가 있습니다.
이 문서가 질문에 답하는 데 도움이 되는지 판단하세요.

질문: {question}

문서 내용:
{document}

이 문서가 질문과 관련 있나요? (yes 또는 no만 출력):""")

    grade_doc_chain = grade_doc_prompt | llm | StrOutputParser()

    def grade_documents(state: SelfRAGState) -> SelfRAGState:
        """
        검색된 각 문서의 관련성을 평가하여 관련 있는 문서만 유지한다.

        모든 문서를 개별 평가하므로 문서 수만큼 LLM 호출 발생.
        비용 절감을 위해 k를 작게 유지하는 것이 중요.
        """
        relevant_docs = []
        relevant_sources = []

        for doc_content, doc_source in zip(
            state["documents"], state["document_sources"]
        ):
            result = grade_doc_chain.invoke({
                "question": state["question"],
                "document": doc_content[:500],  # 긴 문서는 앞부분만
            })

            is_relevant = "yes" in result.lower().strip()
            print(f"  [관련성] {doc_source}: {'관련 있음' if is_relevant else '관련 없음'}")

            if is_relevant:
                relevant_docs.append(doc_content)
                relevant_sources.append(doc_source)

        documents_relevant = len(relevant_docs) > 0
        print(f"  [관련 문서] {len(relevant_docs)}/{len(state['documents'])}개 선택")

        return {
            "documents": relevant_docs,
            "document_sources": relevant_sources,
            "documents_relevant": documents_relevant,
        }

    # ─────────────────────────────────────────
    # Node 4: 답변 생성
    # ─────────────────────────────────────────

    generate_prompt = ChatPromptTemplate.from_template("""다음 컨텍스트만 사용하여 질문에 답하세요.

컨텍스트:
{context}

질문: {question}

규칙:
- 컨텍스트에 없는 정보는 사용하지 마세요.
- 컨텍스트에 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
- 한국어로 답하세요.

답변:""")

    generate_chain = generate_prompt | llm | StrOutputParser()

    def generate(state: SelfRAGState) -> SelfRAGState:
        """관련 문서를 기반으로 답변 생성"""
        context = "\n\n".join([
            f"[{src}]: {content}"
            for content, src in zip(
                state["documents"],
                state["document_sources"],
            )
        ])

        generation = generate_chain.invoke({
            "context": context or "관련 문서 없음",
            "question": state["question"],
        })

        print(f"  [생성] 답변 생성 완료 ({len(generation)}자)")
        return {"generation": generation}

    def generate_direct(state: SelfRAGState) -> SelfRAGState:
        """검색 없이 LLM 일반 지식으로 답변"""
        from langchain_core.messages import HumanMessage

        response = llm.invoke([
            HumanMessage(content=f"다음 질문에 한국어로 답하세요.\n질문: {state['question']}")
        ])
        print(f"  [직접 생성] 검색 없이 답변 생성")
        return {"generation": response.content}

    # ─────────────────────────────────────────
    # Node 5: 답변 근거 평가 (Faithfulness)
    # ─────────────────────────────────────────

    faithfulness_prompt = ChatPromptTemplate.from_template("""다음 답변이 제공된 문서에만 근거하는지 판단하세요.

문서:
{documents}

답변: {generation}

판단 기준:
- 답변의 모든 주장이 문서에서 확인 가능 → yes
- 문서에 없는 내용이 답변에 포함됨 → no

답변 (yes 또는 no만 출력):""")

    faithfulness_chain = faithfulness_prompt | llm | StrOutputParser()

    def grade_generation(state: SelfRAGState) -> SelfRAGState:
        """생성된 답변이 문서에 근거하는가?"""
        if not state.get("documents"):
            # 문서 없이 생성된 경우 평가 건너뜀
            return {"answer_grounded": True}

        result = faithfulness_chain.invoke({
            "documents": "\n".join(state["documents"][:3]),
            "generation": state.get("generation", ""),
        })

        grounded = "yes" in result.lower().strip()
        print(f"  [근거 평가] {'근거 있음' if grounded else '근거 없음'}")

        return {"answer_grounded": grounded}

    # ─────────────────────────────────────────
    # Node 6: 최종 답변 설정
    # ─────────────────────────────────────────

    def finalize(state: SelfRAGState) -> SelfRAGState:
        """최종 답변을 설정하고 출처 정보를 추가한다."""
        answer = state.get("generation", "답변을 생성할 수 없습니다.")
        sources = state.get("document_sources", [])

        if sources:
            sources_text = ", ".join(set(sources))
            final = f"{answer}\n\n(출처: {sources_text})"
        else:
            final = answer

        return {"final_answer": final}

    return {
        "check_retrieval": check_retrieval_needed,
        "retrieve": retrieve,
        "grade_documents": grade_documents,
        "generate": generate,
        "generate_direct": generate_direct,
        "grade_generation": grade_generation,
        "finalize": finalize,
    }


# ─────────────────────────────────────────────
# 3. 라우팅 함수
# ─────────────────────────────────────────────

def route_after_retrieval_check(state: SelfRAGState) -> str:
    """검색 필요 여부에 따라 다음 노드 결정"""
    if state["retrieval_needed"]:
        return "retrieve"
    return "generate_direct"


def route_after_grading(state: SelfRAGState) -> str:
    """문서 관련성에 따라 다음 노드 결정"""
    if state["documents_relevant"]:
        return "generate"
    elif state.get("loop_count", 0) < 2:
        print("  [재검색] 관련 문서 없음, 재검색 시도")
        return "retrieve"  # 쿼리 재변환 후 재검색
    else:
        # 2번 시도했는데 관련 문서 없으면 포기
        print("  [포기] 관련 문서를 찾을 수 없음")
        return "generate"


def route_after_generation(state: SelfRAGState) -> str:
    """답변 근거 여부에 따라 다음 노드 결정"""
    if state["answer_grounded"]:
        return "finalize"
    elif state.get("loop_count", 0) >= 3:
        print("  [강제 종료] 최대 재시도 도달")
        return "finalize"
    else:
        print("  [재생성] 근거 없는 답변, 재생성 시도")
        return "generate"


# ─────────────────────────────────────────────
# 4. 그래프 구성 및 실행
# ─────────────────────────────────────────────

def build_self_rag_graph(retriever, llm):
    """
    Self-RAG 그래프를 구성하고 컴파일한다.
    """
    nodes = build_nodes(retriever, llm)

    workflow = StateGraph(SelfRAGState)

    # 노드 추가
    for name, fn in nodes.items():
        workflow.add_node(name, fn)

    # 시작점
    workflow.set_entry_point("check_retrieval")

    # 엣지 설정
    workflow.add_conditional_edges(
        "check_retrieval",
        route_after_retrieval_check,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "retrieve": "retrieve"},
    )
    workflow.add_edge("generate", "grade_generation")
    workflow.add_edge("generate_direct", "finalize")
    workflow.add_conditional_edges(
        "grade_generation",
        route_after_generation,
        {"finalize": "finalize", "generate": "generate"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


def run_self_rag(question: str, retriever, llm) -> dict:
    """Self-RAG 파이프라인 실행"""
    app = build_self_rag_graph(retriever, llm)

    initial_state = {
        "question": question,
        "documents": [],
        "document_sources": [],
        "generation": None,
        "retrieval_needed": False,
        "documents_relevant": False,
        "answer_grounded": False,
        "loop_count": 0,
        "final_answer": None,
    }

    print(f"\n=== Self-RAG 실행: '{question}' ===")
    result = app.invoke(initial_state)

    return {
        "question": question,
        "answer": result.get("final_answer", result.get("generation", "")),
        "documents": result.get("documents", []),
        "sources": result.get("document_sources", []),
        "loop_count": result.get("loop_count", 0),
    }


# ─────────────────────────────────────────────
# 5. 실행 예시
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size
    from hybrid_retriever import build_ensemble_retriever

    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    ensemble = build_ensemble_retriever(chunks, k=5)
    llm = create_llm()

    test_questions = [
        "503 오류 발생 시 대응 절차는?",         # 관련 문서 있음
        "2024년 8월 장애 내역을 알려줘",          # 관련 문서 없음 (기간 불일치)
        "2 더하기 2는?",                          # 검색 불필요
    ]

    for q in test_questions:
        result = run_self_rag(q, ensemble, llm)
        print(f"\n질문: {result['question']}")
        print(f"답변: {result['answer']}")
        print(f"루프 횟수: {result['loop_count']}")
```

**직접 작성해보세요 — 실습 과제 1:**

```python
# 실습 1-1: Self-RAG에 쿼리 재작성 추가
# 관련 문서가 없을 때 단순히 재검색하는 것 대신,
# LLM이 쿼리를 다르게 표현하여 재검색하는 노드를 추가하라.

def rewrite_query(state: SelfRAGState, llm) -> SelfRAGState:
    """
    관련 문서를 찾지 못했을 때 쿼리를 다르게 표현한다.

    예시:
    원래 쿼리: "서버 다운 이유"
    재작성:    "서버 응답 없음 원인" 또는 "503 오류 발생 원인"
    """
    rewrite_prompt = ChatPromptTemplate.from_template("""
다음 질문을 검색에 더 적합하게 다시 작성하세요.
동의어를 사용하거나 더 구체적으로 표현하세요.

원래 질문: {question}

재작성된 질문 (한 줄만):""")

    chain = rewrite_prompt | llm | StrOutputParser()
    new_question = chain.invoke({"question": state["question"]})

    print(f"  [쿼리 재작성] '{state['question']}' → '{new_question.strip()}'")

    # TODO: 재작성된 질문으로 state를 업데이트하고 반환
    pass

# 실습 1-2: 그래프에 쿼리 재작성 노드 통합
# grade_documents에서 관련 문서가 없을 때
# "retrieve"로 바로 가는 대신 "rewrite_query" → "retrieve" 경로를 추가하라.

# TODO: route_after_grading 함수와 workflow 엣지를 수정하라
```

---

## 4. CRAG — Corrective RAG

### 4.1 CRAG 개념

CRAG(Corrective RAG)는 Self-RAG의 변형으로, 로컬 벡터 DB 검색이 실패했을 때 대체 소스(웹 검색)로 보완하는 패턴이다.

```
CRAG 흐름:
  질문 → 로컬 벡터 DB 검색
           │
           ▼
        관련성 평가
        ┌─────┬─────┐
      High  Low  Ambiguous
        │     │       │
        ▼     ▼       ▼
      사용   웹검색  웹검색+로컬혼합
```

폐쇄망 환경에서는 웹 검색 대신 내부 추가 데이터 소스(다른 DB, API)를 사용한다.

```python
# 파일: crag.py
# 목적: 폐쇄망 환경의 CRAG (웹 검색 대신 대체 소스 사용)

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ollama_llm import create_llm
from loaders import load_directory, LOADERS

load_dotenv()


# ─────────────────────────────────────────────
# 관련성 등급 평가
# ─────────────────────────────────────────────

def grade_relevance(
    question: str,
    documents: List[Document],
    llm,
    threshold: float = 0.5,
) -> str:
    """
    검색 결과의 전체적 관련성을 평가한다.

    Returns:
        "high"      - 관련 문서 많음, 그대로 사용
        "low"       - 관련 문서 거의 없음, 대체 소스 필요
        "ambiguous" - 관련 있을 수도 없을 수도, 혼합 사용
    """
    if not documents:
        return "low"

    prompt = ChatPromptTemplate.from_template("""다음 질문과 검색 결과를 보고 검색 품질을 평가하세요.

질문: {question}

검색된 문서들 (상위 3개):
{docs}

전체적인 관련성 평가:
- "high": 질문에 답하기에 충분한 관련 정보가 있음
- "low": 관련 정보가 거의 없어 다른 소스가 필요함
- "ambiguous": 일부 관련 있지만 불확실함

평가 (high, low, ambiguous 중 하나만):""")

    chain = prompt | llm | StrOutputParser()

    doc_preview = "\n\n".join([
        f"[{doc.metadata.get('source')}] {doc.page_content[:200]}"
        for doc in documents[:3]
    ])

    result = chain.invoke({
        "question": question,
        "docs": doc_preview,
    })

    result = result.lower().strip()
    if "high" in result:
        return "high"
    elif "low" in result:
        return "low"
    else:
        return "ambiguous"


# ─────────────────────────────────────────────
# 대체 소스 (폐쇄망에서 웹 검색 대신)
# ─────────────────────────────────────────────

def search_fallback_source(query: str, fallback_dir: str = "./fallback_data") -> List[Document]:
    """
    폐쇄망 환경에서 웹 검색 대신 사용하는 대체 소스 검색.

    실제 환경에 맞게 교체:
    - 내부 API 검색
    - 다른 데이터 디렉토리 검색
    - 사내 Wiki API 호출
    - 별도 벡터 DB (Elasticsearch 등)
    """
    if not os.path.exists(fallback_dir):
        print(f"  [CRAG] 대체 소스 없음: {fallback_dir}")
        return []

    fallback_docs = load_directory(fallback_dir)
    if not fallback_docs:
        return []

    # 간단한 키워드 매칭으로 관련 문서 필터링
    keywords = query.lower().split()
    relevant = [
        doc for doc in fallback_docs
        if any(kw in doc.page_content.lower() for kw in keywords)
    ]

    print(f"  [CRAG] 대체 소스에서 {len(relevant)}개 문서 검색됨")
    return relevant[:5]


# ─────────────────────────────────────────────
# CRAG 파이프라인
# ─────────────────────────────────────────────

def crag_retrieve(
    question: str,
    primary_retriever,
    llm,
    fallback_dir: str = "./fallback_data",
) -> List[Document]:
    """
    CRAG: 1차 검색 결과의 관련성에 따라 대체 소스를 사용한다.
    """
    # 1차 검색
    primary_docs = primary_retriever.invoke(question)
    print(f"  [CRAG] 1차 검색: {len(primary_docs)}개 문서")

    # 관련성 평가
    grade = grade_relevance(question, primary_docs, llm)
    print(f"  [CRAG] 관련성 등급: {grade}")

    if grade == "high":
        # 1차 검색 결과 그대로 사용
        return primary_docs

    elif grade == "low":
        # 대체 소스만 사용
        fallback = search_fallback_source(question, fallback_dir)
        return fallback if fallback else primary_docs  # fallback도 없으면 원본 사용

    else:  # ambiguous
        # 1차 검색 + 대체 소스 혼합
        fallback = search_fallback_source(question, fallback_dir)
        # 중복 제거 후 혼합
        combined = primary_docs + [
            d for d in fallback
            if d.page_content not in [p.page_content for p in primary_docs]
        ]
        return combined[:8]  # 최대 8개


if __name__ == "__main__":
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size
    from hybrid_retriever import build_ensemble_retriever

    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    ensemble = build_ensemble_retriever(chunks, k=5)
    llm = create_llm()

    questions = [
        "503 오류 대응 방법",           # primary에 있음
        "2024년 9월 배포 일정은?",       # primary에 없음
    ]

    for q in questions:
        print(f"\n=== CRAG 실행: '{q}' ===")
        docs = crag_retrieve(q, ensemble, llm)
        print(f"최종 문서: {len(docs)}개")
        for doc in docs[:2]:
            print(f"  [{doc.metadata.get('source')}] {doc.page_content[:80]}...")
```

---

## 5. HyDE — Hypothetical Document Embeddings

### 5.1 HyDE 이론

**문제 상황:**

```
임베딩 공간의 불균형:
  짧은 질문: "HyDE란?"
  → 벡터: [0.21, -0.12, 0.08, ...]  (768차원이지만 정보가 희박)
  
  긴 답변 문서: "HyDE(Hypothetical Document Embeddings)는 짧은 질문의
                 임베딩 품질 문제를 해결하기 위해..."
  → 벡터: [0.31, -0.28, 0.19, ...]  (정보가 풍부)
  
  문제: 짧은 질문 벡터와 긴 문서 벡터의 공간이 다르다.
         질문이 짧을수록 검색 정확도 하락.
```

**HyDE 해결책:**

```
1단계: 질문으로 가상의 답변 생성 (LLM 사용)
  "HyDE란?" → LLM → "HyDE는 짧은 질문의 임베딩을 개선하기 위해
                      가상의 문서를 생성하여 그 임베딩으로 검색하는..."
  (실제 사실이 아니어도 됨, 검색을 위한 "형태"만 필요)

2단계: 가상 답변의 임베딩으로 검색
  가상 답변 벡터 ≈ 실제 문서 벡터 (같은 길이, 같은 형태)
  → 검색 정확도 향상
```

```
수식으로 표현:
  일반 RAG:   q_emb = E(question)
  HyDE RAG:   q_emb = E(LLM(question))
              E(LLM(q)) ≈ E(d*)  (d* = 관련 문서)
              d* = argmax_d sim(E(LLM(q)), E(d))
```

### 5.2 HyDE 구현

```python
# 파일: hyde_rag.py
# 목적: HyDE(Hypothetical Document Embeddings) 구현

import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from ollama_llm import create_llm, create_embeddings

load_dotenv()


# ─────────────────────────────────────────────
# HyDE 생성기
# ─────────────────────────────────────────────

HYPOTHETICAL_PROMPT = ChatPromptTemplate.from_template("""다음 질문에 대한 가상의 상세한 답변 문서를 작성하세요.

중요: 실제 사실이 아니어도 됩니다. 형식과 길이가 검색 대상 문서와 비슷하게 만드는 것이 목적입니다.
검색 목적의 가상 문서이므로 자세하게 작성하세요.

질문: {question}

가상 답변 문서 (200~400자):""")


def generate_hypothetical_document(question: str, llm) -> str:
    """
    질문에 대한 가상의 답변 문서를 생성한다.

    이 문서는 실제 정보가 아닌 검색을 위한 "프록시"다.
    LLM이 생성하는 형태는 실제 문서와 비슷한 패턴을 가진다.
    """
    chain = HYPOTHETICAL_PROMPT | llm | StrOutputParser()
    hypothetical = chain.invoke({"question": question})
    return hypothetical.strip()


# ─────────────────────────────────────────────
# HyDE 검색 함수
# ─────────────────────────────────────────────

def hyde_search(
    question: str,
    vectorstore: Chroma,
    llm,
    k: int = 5,
    compare_with_standard: bool = True,
) -> List[Document]:
    """
    HyDE를 사용한 검색.

    1. LLM으로 가상 답변 생성
    2. 가상 답변의 임베딩으로 검색

    Args:
        question: 사용자 질문
        vectorstore: Chroma 벡터 DB
        llm: OllamaLLM 인스턴스
        k: 검색할 문서 수
        compare_with_standard: True면 일반 검색과 결과를 비교 출력

    Returns:
        검색된 Document 목록
    """
    # 1. 가상 답변 생성
    hypothetical_doc = generate_hypothetical_document(question, llm)
    print(f"\n[HyDE] 가상 문서 생성:")
    print(f"  '{hypothetical_doc[:100]}...'")

    # 2. 가상 답변의 임베딩으로 검색
    hyde_results = vectorstore.similarity_search(hypothetical_doc, k=k)

    if compare_with_standard:
        # 비교: 일반 벡터 검색
        standard_results = vectorstore.similarity_search(question, k=k)

        print(f"\n[일반 검색] '{question}'")
        for i, doc in enumerate(standard_results[:3], 1):
            print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:80]}...")

        print(f"\n[HyDE 검색] 가상 문서 임베딩 사용")
        for i, doc in enumerate(hyde_results[:3], 1):
            print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:80]}...")

    return hyde_results


# ─────────────────────────────────────────────
# HyDE RAG 체인
# ─────────────────────────────────────────────

def build_hyde_qa_chain(vectorstore: Chroma, llm):
    """
    HyDE를 사용하는 RAG 체인을 구성한다.

    LangChain의 기본 RetrievalQA를 커스터마이즈하여
    검색 단계에서 HyDE를 사용한다.
    """
    from langchain.schema import BaseRetriever
    from langchain.callbacks.manager import CallbackManagerForRetrieverRun

    class HyDERetriever(BaseRetriever):
        """HyDE를 사용하는 커스텀 리트리버"""
        vectorstore: Chroma
        llm: object
        k: int = 5

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun = None,
        ) -> List[Document]:
            """HyDE를 통해 관련 문서를 검색한다"""
            hypothetical = generate_hypothetical_document(query, self.llm)
            return self.vectorstore.similarity_search(hypothetical, k=self.k)

    hyde_retriever = HyDERetriever(vectorstore=vectorstore, llm=llm, k=5)

    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        template="""컨텍스트를 기반으로 질문에 답하세요.

컨텍스트:
{context}

질문: {question}

규칙: 컨텍스트에 없는 정보는 사용하지 마세요. 한국어로 답변하세요.

답변:""",
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hyde_retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


if __name__ == "__main__":
    from langchain_chroma import Chroma
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size

    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    embeddings = create_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hyde_test",
    )
    llm = create_llm()

    test_questions = [
        "RAG 시스템의 성능 지표",
        "서버 장애 발생 원인",
        "임베딩 벡터 차원",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"질문: {question}")
        hyde_search(question, vectorstore, llm, k=3)
```

**직접 작성해보세요 — 실습 과제 2:**

```python
# 실습 2-1: 다중 가상 문서 HyDE
# 하나의 가상 문서 대신 3개를 생성하고,
# 3개 가상 문서의 평균 임베딩으로 검색하라.
# 이렇게 하면 단일 가상 문서의 편향을 줄일 수 있다.

import numpy as np

def multi_hyde_search(
    question: str,
    vectorstore: Chroma,
    llm,
    num_hypothetical: int = 3,
    k: int = 5,
) -> List[Document]:
    """
    여러 가상 문서의 평균 임베딩으로 검색한다.

    수식: q_emb = mean(E(LLM_1(q)), E(LLM_2(q)), E(LLM_3(q)))
    """
    embeddings = create_embeddings()

    # TODO:
    # 1. generate_hypothetical_document를 num_hypothetical번 호출
    # 2. 각 가상 문서의 임베딩을 계산
    # 3. 임베딩들의 평균을 계산 (numpy 사용)
    # 4. 평균 임베딩으로 유사도 검색

    # 힌트: vectorstore.similarity_search_by_vector(embedding_vector, k=k)
    pass

# 실습 2-2: HyDE vs 일반 검색 Recall 비교
# 위에서 작성한 EVAL_DATASET을 사용하여
# 일반 검색과 HyDE 검색의 Recall@5를 비교하라.
# 어떤 질문 유형에서 HyDE가 더 효과적인가?
```

---

## 6. Multi-hop RAG — 다단계 추론

### 6.1 Multi-hop이 필요한 경우

```
단순 질문 (Single-hop):
  "503 오류 대응 절차는?"
  → 한 번의 검색으로 답변 가능

복합 질문 (Multi-hop):
  "우리 팀에서 가장 많은 장애를 일으킨 서비스의 담당자 연락처는?"
  
  필요한 추론:
  Hop 1: "어떤 서비스가 가장 많은 장애를 일으켰는가?"
         검색 → "API 서버: 12회, DB: 5회, 인증 서버: 3회"
         중간 답: API 서버
  
  Hop 2: "API 서버 담당자는 누구인가?"
         검색 → "API 서버 담당: 홍길동 (010-1234-5678)"
         최종 답: 홍길동, 010-1234-5678
```

### 6.2 Multi-hop RAG 구현

```python
# 파일: multi_hop_rag.py
# 목적: 다단계 추론이 필요한 복합 질문 처리

import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from ollama_llm import create_llm

load_dotenv()


# ─────────────────────────────────────────────
# 1. 질문 분해 (Decomposition)
# ─────────────────────────────────────────────

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template("""복잡한 질문을 순서대로 답해야 하는 단순한 하위 질문들로 분해하세요.

규칙:
- 각 하위 질문은 독립적으로 검색 가능해야 합니다
- 이전 하위 질문의 답변이 다음 질문에 필요한 경우, 순서를 명시하세요
- 최대 3개로 제한하세요

질문: {question}

하위 질문들 (번호 목록으로, 한 줄에 하나씩):""")


def decompose_question(question: str, llm) -> List[str]:
    """
    복합 질문을 순서가 있는 하위 질문 목록으로 분해한다.

    Returns:
        하위 질문 목록 (순서대로)
    """
    chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    # 번호 목록 파싱
    sub_questions = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # "1. 질문 내용", "1) 질문 내용", "- 질문 내용" 등의 패턴 처리
        for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "-"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            sub_questions.append(line)

    return sub_questions[:3]  # 최대 3개


# ─────────────────────────────────────────────
# 2. 중간 답변 생성
# ─────────────────────────────────────────────

INTERMEDIATE_PROMPT = ChatPromptTemplate.from_template("""다음 컨텍스트를 사용하여 질문에 답하세요.

이전 답변들:
{previous_answers}

현재 컨텍스트:
{context}

현재 질문: {question}

답변 (간결하게):""")


def answer_with_context(
    question: str,
    retrieved_docs: List[Document],
    previous_answers: str,
    llm,
) -> str:
    """중간 단계의 질문에 답한다"""
    chain = INTERMEDIATE_PROMPT | llm | StrOutputParser()

    context = "\n\n".join([
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in retrieved_docs
    ])

    return chain.invoke({
        "question": question,
        "context": context,
        "previous_answers": previous_answers or "없음",
    })


# ─────────────────────────────────────────────
# 3. Multi-hop 메인 파이프라인
# ─────────────────────────────────────────────

def multi_hop_retrieve(
    question: str,
    retriever,
    llm,
    max_hops: int = 3,
) -> Tuple[List[Document], str]:
    """
    복합 질문을 단계적으로 검색하고 답변한다.

    Args:
        question: 원본 복합 질문
        retriever: LangChain 리트리버
        llm: OllamaLLM
        max_hops: 최대 hop 수

    Returns:
        (수집된 모든 문서, 최종 답변)
    """
    print(f"\n=== Multi-hop RAG: '{question}' ===")

    # 1. 질문 분해
    sub_questions = decompose_question(question, llm)
    print(f"\n[분해된 하위 질문 ({len(sub_questions)}개)]:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")

    # 2. 순차 검색 및 중간 답변
    all_docs = []
    previous_answers = ""

    for i, sub_q in enumerate(sub_questions[:max_hops], 1):
        print(f"\n[Hop {i}] '{sub_q}'")

        # 이전 답변을 쿼리에 추가하여 더 정확한 검색
        search_query = sub_q
        if previous_answers:
            search_query = f"{sub_q}\n(이전 컨텍스트: {previous_answers[:200]})"

        # 검색
        docs = retriever.invoke(search_query)
        all_docs.extend(docs)
        print(f"  검색된 문서: {len(docs)}개")

        # 중간 답변 생성 (마지막 hop 제외)
        if i < len(sub_questions):
            mid_answer = answer_with_context(
                sub_q, docs, previous_answers, llm
            )
            previous_answers += f"\n[{i}단계 답변] {mid_answer}"
            print(f"  중간 답변: {mid_answer[:100]}...")

    # 3. 최종 답변 생성
    print(f"\n[최종 답변 생성]")
    final_prompt = ChatPromptTemplate.from_template("""다음 단계별 검색 결과와 이전 답변들을 종합하여 원래 질문에 최종 답변하세요.

원래 질문: {original_question}

단계별 답변 요약:
{previous_answers}

전체 수집 문서:
{all_context}

최종 답변 (원래 질문에 대해 종합적으로):""")

    chain = final_prompt | llm | StrOutputParser()

    all_context = "\n\n".join([
        f"[{doc.metadata.get('source')}] {doc.page_content[:200]}"
        for doc in all_docs[:6]  # 최대 6개
    ])

    final_answer = chain.invoke({
        "original_question": question,
        "previous_answers": previous_answers,
        "all_context": all_context,
    })

    return all_docs, final_answer


# ─────────────────────────────────────────────
# 4. 단순 vs Multi-hop 비교
# ─────────────────────────────────────────────

def compare_rag_approaches(question: str, qa_chain, retriever, llm):
    """단순 RAG와 Multi-hop RAG의 결과를 비교한다."""
    print(f"\n{'='*70}")
    print(f"질문: {question}")

    # 단순 RAG
    print("\n[단순 RAG]")
    simple_result = qa_chain.invoke({"query": question})
    print(f"답변: {simple_result['result']}")

    # Multi-hop RAG
    print("\n[Multi-hop RAG]")
    docs, answer = multi_hop_retrieve(question, retriever, llm)
    print(f"\n최종 답변: {answer}")

    print("="*70)


if __name__ == "__main__":
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size
    from hybrid_retriever import build_ensemble_retriever
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    ensemble = build_ensemble_retriever(chunks, k=5)
    llm = create_llm()

    # 단순 RAG 체인
    simple_qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=ensemble, return_source_documents=True
    )

    # Multi-hop 테스트
    complex_question = "서버 장애 발생 원인과 그에 대한 재발 방지 대책을 함께 알려줘"
    docs, answer = multi_hop_retrieve(complex_question, ensemble, llm)
    print(f"\n최종 답변:\n{answer}")
```

**직접 작성해보세요 — 실습 과제 3:**

```python
# 실습 3-1: 질문 분류기 구현
# 질문을 받아 Single-hop 또는 Multi-hop인지 분류하는 함수를 구현하라.
# Multi-hop으로 판단되면 decompose, Single-hop이면 일반 RAG를 사용한다.

def classify_question_complexity(question: str, llm) -> str:
    """
    질문의 복잡도를 분류한다.

    Returns:
        "single": 단순 질문 (한 번의 검색으로 해결 가능)
        "multi":  복합 질문 (다단계 추론 필요)
    """
    prompt = ChatPromptTemplate.from_template("""다음 질문이 단순 검색(single)으로 해결 가능한지,
다단계 추론(multi)이 필요한지 분류하세요.

단순(single): "503 오류 대응 방법은?", "임베딩이란 무엇인가?"
다단계(multi): "가장 많이 발생한 오류 유형과 그 원인을 모두 알려줘",
               "두 서비스의 장애 빈도를 비교하고 더 안정적인 서비스는?"

질문: {question}

분류 (single 또는 multi만):""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    return "multi" if "multi" in result.lower() else "single"

# TODO: classify_question_complexity를 사용하여
# 자동으로 단순 RAG 또는 Multi-hop RAG를 선택하는 라우터를 구현하라.
```

---

## 7. RAGAS 평가 프레임워크

### 7.1 RAGAS 4개 지표 이론

RAGAS(RAG Assessment)는 RAG 시스템의 품질을 LLM을 사용하여 자동으로 측정한다.

```
4개 지표:

1. Faithfulness (신뢰성, 0~1)
   = 답변의 주장 중 검색 문서에서 지지되는 비율
   
   측정 방법:
   답변을 여러 주장으로 분해 → 각 주장이 문서에 있는지 확인
   
   예시:
   답변: "503 오류는 커넥션 풀 고갈(1)로 발생하며, 재시작으로 해결(2)된다"
   문서: "커넥션 풀 고갈이 원인" (주장 1 지지)
         "재시작 전 원인 파악 필요" (주장 2는 지지 안 됨)
   Faithfulness = 1/2 = 0.5

2. Answer Relevancy (답변 관련성, 0~1)
   = 생성된 답변이 질문과 얼마나 관련 있는가
   
   측정 방법 (역방향 접근):
   답변을 보고 "이 답변이 나왔을 법한 질문들"을 생성
   → 원래 질문과 생성된 질문들의 임베딩 유사도 평균
   
   예시:
   질문: "503 오류 대응 방법은?"
   답변: "커넥션 풀을 200으로 늘리세요"
   역생성 질문: "커넥션 풀 크기를 늘리는 방법은?" (유사도 높음)
   Answer Relevancy ≈ 0.85

3. Context Recall (컨텍스트 재현율, 0~1)
   = 정답에 필요한 정보가 검색된 문서에 얼마나 포함되어 있는가
   
   측정 방법:
   ground_truth를 문장으로 분해
   → 각 문장이 검색 문서에서 도출 가능한지 확인
   
   ground_truth 없이는 측정 불가.

4. Context Precision (컨텍스트 정밀도, 0~1)
   = 검색된 문서 중 실제 답변에 유용한 문서의 비율
   
   낮으면: 노이즈가 많이 검색됨 → k 줄이기, Reranking 추가

요약:
  Faithfulness 낮음 → LLM이 hallucination 발생 → 프롬프트 강화
  Answer Relevancy 낮음 → 답변이 질문에서 벗어남 → 프롬프트 개선
  Context Recall 낮음 → 검색이 필요한 문서를 놓침 → 청킹/검색 개선
  Context Precision 낮음 → 관련 없는 문서가 많이 검색됨 → k 줄이기/Reranking
```

### 7.2 RAGAS 설치 및 기본 사용

```python
# pip install ragas datasets

# 중요: RAGAS는 내부적으로 OpenAI API를 사용하는 경우가 많다.
# Ollama 로컬 LLM을 사용하려면 커스텀 LLM/임베딩을 전달해야 한다.

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset
```

### 7.3 Ollama와 RAGAS 통합

```python
# 파일: ragas_evaluation.py
# 목적: Ollama LLM/임베딩으로 RAGAS 평가 실행
# pip install ragas datasets langchain-core

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()


# ─────────────────────────────────────────────
# RAGAS용 LangChain 래퍼
# ─────────────────────────────────────────────

def get_ragas_llm():
    """
    RAGAS가 사용할 LLM을 반환한다.
    RAGAS는 내부적으로 LangChain 인터페이스를 사용하므로
    OllamaLLM을 그대로 전달할 수 있다.
    """
    from ollama_llm import create_llm
    return create_llm()


def get_ragas_embeddings():
    """RAGAS가 사용할 임베딩을 반환한다."""
    from ollama_llm import create_embeddings
    return create_embeddings()


# ─────────────────────────────────────────────
# 평가 데이터셋 구성
# ─────────────────────────────────────────────

def build_eval_dataset(
    qa_chain,
    eval_questions: List[str],
    ground_truths: List[str],
) -> Dataset:
    """
    RAG 체인을 실행하여 RAGAS 평가 데이터셋을 구성한다.

    Args:
        qa_chain: return_source_documents=True인 RetrievalQA 체인
        eval_questions: 평가 질문 목록
        ground_truths: 각 질문의 정답 목록 (Context Recall 측정용)

    Returns:
        RAGAS Dataset
    """
    answers = []
    contexts = []

    print(f"[RAGAS] {len(eval_questions)}개 질문 처리 중...")

    for i, question in enumerate(eval_questions, 1):
        print(f"  [{i}/{len(eval_questions)}] {question[:40]}...")

        result = qa_chain.invoke({"query": question})

        answers.append(result["result"])
        contexts.append([
            doc.page_content
            for doc in result.get("source_documents", [])
        ])

    dataset_dict = {
        "question": eval_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    return Dataset.from_dict(dataset_dict)


# ─────────────────────────────────────────────
# RAGAS 평가 실행
# ─────────────────────────────────────────────

def run_ragas_evaluation(
    dataset: Dataset,
    metrics: Optional[List] = None,
) -> Dict[str, float]:
    """
    RAGAS 평가를 실행하고 결과를 반환한다.

    Args:
        dataset: RAGAS Dataset (question, answer, contexts, ground_truth 컬럼)
        metrics: 사용할 지표 목록 (기본: 4개 전체)

    Returns:
        {"faithfulness": 0.85, "answer_relevancy": 0.82, ...}
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    print(f"\n[RAGAS] 평가 실행 중...")
    print(f"  데이터셋 크기: {len(dataset)}")
    print(f"  지표: {[m.name for m in metrics]}")

    # RAGAS 평가 실행
    # llm과 embeddings를 전달하여 Ollama 사용
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    scores = dict(result)
    print("\n[RAGAS] 평가 결과:")
    for metric_name, score in scores.items():
        emoji = "✓" if score >= 0.8 else ("△" if score >= 0.6 else "✗")
        print(f"  {emoji} {metric_name}: {score:.4f}")

    return scores


# ─────────────────────────────────────────────
# 지표 해석
# ─────────────────────────────────────────────

def interpret_ragas_scores(scores: Dict[str, float]) -> List[str]:
    """
    RAGAS 점수를 해석하고 개선 방향을 제시한다.
    """
    recommendations = []

    faithfulness = scores.get("faithfulness", 1.0)
    if faithfulness < 0.75:
        recommendations.append(
            f"Faithfulness={faithfulness:.2f} 낮음 → "
            "프롬프트에 '컨텍스트만 사용'을 강조하거나, "
            "Self-RAG로 Faithfulness 체크를 추가하세요."
        )

    answer_rel = scores.get("answer_relevancy", 1.0)
    if answer_rel < 0.75:
        recommendations.append(
            f"Answer Relevancy={answer_rel:.2f} 낮음 → "
            "프롬프트에 '질문에 직접 답변'을 요청하거나 "
            "temperature를 낮추세요."
        )

    ctx_recall = scores.get("context_recall", 1.0)
    if ctx_recall < 0.75:
        recommendations.append(
            f"Context Recall={ctx_recall:.2f} 낮음 → "
            "검색이 필요한 문서를 놓치고 있습니다. "
            "청킹 전략을 변경하거나 하이브리드 검색을 추가하세요."
        )

    ctx_precision = scores.get("context_precision", 1.0)
    if ctx_precision < 0.75:
        recommendations.append(
            f"Context Precision={ctx_precision:.2f} 낮음 → "
            "관련 없는 문서가 많이 검색됩니다. "
            "k 값을 줄이거나 Reranking을 추가하세요."
        )

    if not recommendations:
        recommendations.append("모든 지표가 0.75 이상입니다. 잘 동작하는 RAG 시스템입니다.")

    return recommendations
```

**직접 작성해보세요 — 실습 과제 4:**

```python
# 실습 4-1: 평가 데이터셋 수동 작성
# 현재 data/ 디렉토리의 문서를 읽고
# 최소 5개의 질문-정답 쌍을 직접 작성하라.

EVAL_QA_PAIRS = [
    {
        "question": "...",      # 실제 문서 내용에서 파생된 질문
        "ground_truth": "...",  # 해당 질문의 정답 (문서에서 찾을 수 있어야 함)
    },
    # TODO: 최소 4개 더 추가
]

# 실습 4-2: 단계별 RAGAS 비교 실험
# Level 1 기본 RAG와 Level 2 하이브리드 RAG의 RAGAS 점수를 비교하라.
# 어떤 지표에서 개선이 있었는가?

def compare_rag_systems(
    basic_qa_chain,         # Level 1: 단순 벡터 검색
    enhanced_qa_chain,      # Level 2: 하이브리드 검색
    eval_questions: List[str],
    ground_truths: List[str],
):
    """두 RAG 시스템의 RAGAS 점수를 비교한다."""
    print("=== Level 1 기본 RAG 평가 ===")
    basic_dataset = build_eval_dataset(basic_qa_chain, eval_questions, ground_truths)
    basic_scores = run_ragas_evaluation(basic_dataset)

    print("\n=== Level 2 하이브리드 RAG 평가 ===")
    enhanced_dataset = build_eval_dataset(enhanced_qa_chain, eval_questions, ground_truths)
    enhanced_scores = run_ragas_evaluation(enhanced_dataset)

    print("\n=== 개선 비교 ===")
    for metric in basic_scores:
        before = basic_scores[metric]
        after = enhanced_scores.get(metric, 0)
        delta = after - before
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {metric}: {before:.3f} → {after:.3f} ({arrow}{abs(delta):.3f})")

    # TODO: 구현하라
    pass
```

---

## 8. RAGAS 자동화 평가 파이프라인

```python
# 파일: ragas_pipeline.py
# 목적: RAG 시스템 자동 평가 파이프라인
# 기능: 여러 RAG 변형 비교, 결과 CSV 저장, 개선 방향 제시

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()


class RAGEvaluationPipeline:
    """
    여러 RAG 시스템 변형을 체계적으로 비교·평가하는 파이프라인.

    사용 방법:
        pipeline = RAGEvaluationPipeline(eval_questions, ground_truths)
        pipeline.add_system("basic", basic_qa_chain)
        pipeline.add_system("hybrid", hybrid_qa_chain)
        pipeline.add_system("self_rag", self_rag_chain)
        results = pipeline.run()
        pipeline.save_results("experiment_results.json")
        pipeline.print_summary()
    """

    def __init__(
        self,
        eval_questions: List[str],
        ground_truths: List[str],
    ):
        self.eval_questions = eval_questions
        self.ground_truths = ground_truths
        self.systems: Dict[str, Any] = {}  # name → qa_chain
        self.results: Dict[str, Dict] = {}

    def add_system(self, name: str, qa_chain):
        """평가할 RAG 시스템 추가"""
        self.systems[name] = qa_chain
        print(f"[평가 파이프라인] 시스템 등록: '{name}'")

    def run(
        self,
        metrics: Optional[List] = None,
    ) -> Dict[str, Dict]:
        """등록된 모든 시스템을 평가한다"""
        from ragas_evaluation import build_eval_dataset, run_ragas_evaluation, interpret_ragas_scores

        print(f"\n{'='*70}")
        print(f"RAG 평가 파이프라인 시작")
        print(f"시스템 수: {len(self.systems)}")
        print(f"질문 수: {len(self.eval_questions)}")
        print(f"{'='*70}")

        for system_name, qa_chain in self.systems.items():
            print(f"\n[{system_name}] 평가 시작...")
            start_time = time.time()

            dataset = build_eval_dataset(
                qa_chain,
                self.eval_questions,
                self.ground_truths,
            )
            scores = run_ragas_evaluation(dataset, metrics)
            elapsed = time.time() - start_time
            recommendations = interpret_ragas_scores(scores)

            self.results[system_name] = {
                "scores": scores,
                "elapsed_seconds": elapsed,
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }

        return self.results

    def save_results(self, output_path: str = "ragas_results.json"):
        """평가 결과를 JSON으로 저장"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n[결과 저장] {output_path}")

    def save_csv(self, output_path: str = "ragas_results.csv"):
        """평가 결과를 CSV로 저장"""
        import csv

        if not self.results:
            return

        metrics = list(next(iter(self.results.values()))["scores"].keys())
        fieldnames = ["system"] + metrics + ["elapsed_seconds"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for system_name, data in self.results.items():
                row = {"system": system_name}
                row.update(data["scores"])
                row["elapsed_seconds"] = f"{data['elapsed_seconds']:.1f}"
                writer.writerow(row)

        print(f"[CSV 저장] {output_path}")

    def print_summary(self):
        """평가 결과 요약 출력"""
        if not self.results:
            print("평가 결과 없음")
            return

        metrics = list(next(iter(self.results.values()))["scores"].keys())

        print(f"\n{'='*70}")
        print("평가 결과 요약")
        print(f"{'='*70}")

        # 헤더
        header = f"{'시스템':20}"
        for m in metrics:
            header += f" {m[:12]:>12}"
        print(header)
        print("-" * 70)

        # 각 시스템 결과
        for system_name, data in self.results.items():
            row = f"{system_name:20}"
            for m in metrics:
                score = data["scores"].get(m, 0)
                row += f" {score:>12.4f}"
            print(row)

        # 최고 성능 시스템
        print(f"\n{'─'*70}")
        print("[베스트 시스템 (평균 점수 기준)]:")

        avg_scores = {
            name: sum(data["scores"].values()) / len(data["scores"])
            for name, data in self.results.items()
        }
        best_system = max(avg_scores, key=avg_scores.get)
        print(f"  '{best_system}': 평균 {avg_scores[best_system]:.4f}")

        # 개선 방향
        print(f"\n[개선 방향]:")
        for system_name, data in self.results.items():
            print(f"\n  [{system_name}]:")
            for rec in data["recommendations"]:
                print(f"    - {rec}")


# ─────────────────────────────────────────────
# 샘플 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size
    from hybrid_retriever import build_ensemble_retriever
    from ollama_llm import create_llm, create_embeddings
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # 샘플 평가 데이터
    EVAL_QUESTIONS = [
        "503 오류 발생 시 커넥션 풀 조정 방법은?",
        "디스크 사용률 알람 임계값은?",
        "RAG 시스템의 Recall@k 목표치는?",
        "임베딩 벡터의 차원 수는?",
        "서버 재시작 전 확인해야 할 사항은?",
    ]

    GROUND_TRUTHS = [
        "커넥션 풀 크기를 50에서 200으로 증가시킨다.",
        "85% 초과 시 알람을 설정한다.",
        "k=5 기준으로 0.7 이상을 목표로 한다.",
        "768차원 벡터를 생성한다.",
        "병목 지점을 모니터링 대시보드에서 먼저 확인한다.",
    ]

    # 문서 및 청킹
    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    embeddings = create_embeddings()
    llm = create_llm()

    # System 1: 기본 벡터 검색
    basic_vectorstore = Chroma.from_documents(
        chunks, embeddings, collection_name="basic_eval"
    )
    basic_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=basic_vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    # System 2: 하이브리드 검색
    ensemble = build_ensemble_retriever(chunks, k=5)
    hybrid_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble,
        return_source_documents=True,
    )

    # 평가 실행
    pipeline = RAGEvaluationPipeline(EVAL_QUESTIONS, GROUND_TRUTHS)
    pipeline.add_system("basic_vector", basic_chain)
    pipeline.add_system("hybrid_search", hybrid_chain)

    results = pipeline.run()
    pipeline.save_results("ragas_comparison.json")
    pipeline.save_csv("ragas_comparison.csv")
    pipeline.print_summary()
```

---

## 9. 전체 실습 프로젝트: RAG 품질 개선 실험

```python
# 파일: level3_project.py
# 목적: Level 3 완성 프로젝트 - RAGAS로 측정한 RAG 품질 개선 실험
# 실행: python level3_project.py

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def run_full_experiment():
    """
    Level 3 완성 실험:
    1. Level 2 하이브리드 RAG를 베이스라인으로 RAGAS 측정
    2. HyDE 적용 후 재측정
    3. Self-RAG 적용 후 재측정
    4. 개선 전후 비교 출력
    """
    from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size, strategy_markdown_aware
    from hybrid_retriever import build_ensemble_retriever
    from ollama_llm import create_llm, create_embeddings
    from langchain_chroma import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    from ragas_evaluation import build_eval_dataset, run_ragas_evaluation
    from ragas_pipeline import RAGEvaluationPipeline
    from self_rag import build_self_rag_graph, SelfRAGState
    from hyde_rag import HyDERetriever

    # ─────────────────────────────────────────
    # 공통 설정
    # ─────────────────────────────────────────
    chunks = strategy_fixed_size(SAMPLE_DOCUMENTS, chunk_size=300)
    embeddings = create_embeddings()
    llm = create_llm()

    EVAL_QUESTIONS = [
        "503 오류 발생 시 대응 절차는?",
        "디스크 용량 부족 시 조치 방법은?",
        "커넥션 풀 고갈의 원인은?",
        "임베딩 차원 수는 몇 개인가?",
        "RAG 시스템 성능 목표치는?",
    ]

    GROUND_TRUTHS = [
        "모니터링 대시보드 확인 후 커넥션 풀 상태 점검, 필요시 재시작.",
        "오래된 로그 파일 압축/삭제 후 디스크 증설 요청.",
        "데이터베이스 커넥션 풀 고갈.",
        "nomic-embed-text 기준 768차원.",
        "Recall@5 기준 0.7 이상.",
    ]

    # ─────────────────────────────────────────
    # System 1: Baseline (기본 벡터 검색)
    # ─────────────────────────────────────────
    basic_vs = Chroma.from_documents(chunks, embeddings, collection_name="baseline")
    basic_prompt = PromptTemplate(
        template="""컨텍스트:\n{context}\n\n질문: {question}\n\n답변:""",
        input_variables=["context", "question"],
    )
    baseline_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=basic_vs.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": basic_prompt},
        return_source_documents=True,
    )

    # ─────────────────────────────────────────
    # System 2: 하이브리드 검색
    # ─────────────────────────────────────────
    ensemble = build_ensemble_retriever(chunks, k=5)
    enhanced_prompt = PromptTemplate(
        template="""컨텍스트만 사용하여 답변하세요. 없으면 "모름"이라고 하세요.

컨텍스트:\n{context}\n\n질문: {question}\n\n답변:""",
        input_variables=["context", "question"],
    )
    hybrid_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble,
        chain_type_kwargs={"prompt": enhanced_prompt},
        return_source_documents=True,
    )

    # ─────────────────────────────────────────
    # System 3: HyDE + 하이브리드
    # ─────────────────────────────────────────
    hyde_vs = Chroma.from_documents(chunks, embeddings, collection_name="hyde")
    hyde_retriever_instance = HyDERetriever(
        vectorstore=hyde_vs, llm=llm, k=5
    )
    hyde_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=hyde_retriever_instance,
        chain_type_kwargs={"prompt": enhanced_prompt},
        return_source_documents=True,
    )

    # ─────────────────────────────────────────
    # 평가 실행
    # ─────────────────────────────────────────
    pipeline = RAGEvaluationPipeline(EVAL_QUESTIONS, GROUND_TRUTHS)
    pipeline.add_system("1_baseline", baseline_chain)
    pipeline.add_system("2_hybrid", hybrid_chain)
    pipeline.add_system("3_hyde_hybrid", hyde_chain)

    print("\n" + "="*70)
    print("Level 3 RAG 품질 개선 실험")
    print("="*70)

    results = pipeline.run()
    pipeline.save_results("level3_experiment.json")
    pipeline.save_csv("level3_experiment.csv")
    pipeline.print_summary()

    return results


if __name__ == "__main__":
    run_full_experiment()
```

---

## 10. 테스트 작성

```python
# 파일: test_advanced_rag.py
# 실행: pytest test_advanced_rag.py -v

import pytest
from unittest.mock import MagicMock, patch
from typing import List
from langchain.schema import Document

from rrf_demo import ScoredDocument
from chunking_strategies import SAMPLE_DOCUMENTS, strategy_fixed_size
from reranker import llm_rerank


# ─────────────────────────────────────────────
# Mock LLM (API 호출 없이 테스트)
# ─────────────────────────────────────────────

class MockLLM:
    """테스트용 Mock LLM - API 호출 없이 예측 가능한 응답 반환"""

    def __init__(self, response: str = "yes"):
        self.response = response
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.response)


# ─────────────────────────────────────────────
# HyDE 테스트
# ─────────────────────────────────────────────

class TestHyDE:
    def test_hypothetical_doc_generation(self):
        """가상 문서가 문자열로 반환된다"""
        from hyde_rag import generate_hypothetical_document

        mock_llm = MockLLM(response="가상의 상세한 답변 문서입니다.")
        result = generate_hypothetical_document("테스트 질문", mock_llm)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_hypothetical_doc_is_not_empty(self):
        """LLM 응답이 비어 있어도 처리된다"""
        from hyde_rag import generate_hypothetical_document

        mock_llm = MockLLM(response="  ")
        result = generate_hypothetical_document("질문", mock_llm)
        assert isinstance(result, str)


# ─────────────────────────────────────────────
# Multi-hop 테스트
# ─────────────────────────────────────────────

class TestMultiHop:
    def test_decompose_returns_list(self):
        """질문 분해가 리스트를 반환한다"""
        from multi_hop_rag import decompose_question

        mock_llm = MockLLM(response="1. 하위 질문 1\n2. 하위 질문 2")
        sub_qs = decompose_question("복합 질문", mock_llm)

        assert isinstance(sub_qs, list)
        assert len(sub_qs) >= 1

    def test_decompose_max_3(self):
        """하위 질문이 최대 3개로 제한된다"""
        from multi_hop_rag import decompose_question

        mock_llm = MockLLM(response="1. q1\n2. q2\n3. q3\n4. q4\n5. q5")
        sub_qs = decompose_question("복합 질문", mock_llm)

        assert len(sub_qs) <= 3

    def test_decompose_empty_lines_ignored(self):
        """빈 줄이 있어도 올바르게 파싱된다"""
        from multi_hop_rag import decompose_question

        mock_llm = MockLLM(response="1. 질문1\n\n2. 질문2\n")
        sub_qs = decompose_question("질문", mock_llm)

        assert len(sub_qs) == 2


# ─────────────────────────────────────────────
# LLM 재순위 테스트
# ─────────────────────────────────────────────

class TestLLMRerank:
    def test_rerank_returns_documents(self):
        """재순위가 Document 목록을 반환한다"""
        docs = [
            Document(page_content="문서 A", metadata={"source": "a.txt"}),
            Document(page_content="문서 B", metadata={"source": "b.txt"}),
            Document(page_content="문서 C", metadata={"source": "c.txt"}),
        ]
        mock_llm = MockLLM(response="2,1,3")

        result = llm_rerank("질문", docs, mock_llm, top_k=2)

        assert isinstance(result, list)
        assert len(result) <= 2
        assert all(isinstance(d, Document) for d in result)

    def test_rerank_empty_docs(self):
        """빈 문서 목록에서도 안전하게 동작한다"""
        mock_llm = MockLLM(response="")
        result = llm_rerank("질문", [], mock_llm, top_k=3)
        assert result == []

    def test_rerank_invalid_llm_response(self):
        """LLM이 잘못된 응답을 해도 원본 순서를 유지한다"""
        docs = [
            Document(page_content=f"문서 {i}", metadata={})
            for i in range(3)
        ]
        mock_llm = MockLLM(response="잘못된 응답 형식")

        result = llm_rerank("질문", docs, mock_llm, top_k=3)

        # 파싱 실패해도 원본 문서를 반환해야 함
        assert isinstance(result, list)


# ─────────────────────────────────────────────
# Self-RAG 상태 테스트
# ─────────────────────────────────────────────

class TestSelfRAGState:
    def test_initial_state_valid(self):
        """초기 상태가 올바르게 구성된다"""
        from self_rag import SelfRAGState

        state: SelfRAGState = {
            "question": "테스트 질문",
            "documents": [],
            "document_sources": [],
            "generation": None,
            "retrieval_needed": False,
            "documents_relevant": False,
            "answer_grounded": False,
            "loop_count": 0,
            "final_answer": None,
        }

        assert state["question"] == "테스트 질문"
        assert state["loop_count"] == 0
        assert not state["retrieval_needed"]


# ─────────────────────────────────────────────
# RAGAS 데이터셋 구성 테스트
# ─────────────────────────────────────────────

class TestRAGASDataset:
    def test_build_eval_dataset_structure(self):
        """RAGAS 데이터셋이 필요한 컬럼을 가진다"""
        from ragas_evaluation import build_eval_dataset

        # Mock QA 체인
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "모의 답변",
            "source_documents": [
                Document(page_content="근거 문서", metadata={"source": "a.txt"})
            ]
        }

        questions = ["질문1", "질문2"]
        ground_truths = ["정답1", "정답2"]

        dataset = build_eval_dataset(mock_chain, questions, ground_truths)

        assert "question" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "contexts" in dataset.column_names
        assert "ground_truth" in dataset.column_names
        assert len(dataset) == 2

    def test_contexts_is_list_of_lists(self):
        """contexts가 리스트의 리스트 형태여야 한다"""
        from ragas_evaluation import build_eval_dataset

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "답변",
            "source_documents": [
                Document(page_content="문서1", metadata={}),
                Document(page_content="문서2", metadata={}),
            ]
        }

        dataset = build_eval_dataset(mock_chain, ["질문"], ["정답"])

        # contexts는 List[List[str]] 형태
        assert isinstance(dataset["contexts"][0], list)
        assert all(isinstance(c, str) for c in dataset["contexts"][0])
```

### 10.1 requirements.txt (Level 3 추가분)

```
# requirements.txt (Level 1 + 2 + 3 추가)
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-chroma>=0.1.0
langchain-experimental>=0.0.60
langgraph>=0.1.0
chromadb>=0.5.0
rank-bm25>=0.2.2
FlagEmbedding>=1.2.0
huggingface_hub>=0.20.0
ragas>=0.1.0
datasets>=2.14.0
openpyxl>=3.1.0
pillow>=10.0.0
pytesseract>=0.3.10
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
pytest>=8.0.0
```

---

## 11. 체크리스트 및 다음 단계

### 11.1 Level 3 완성 체크리스트

**개념 이해**

- [ ] Self-RAG의 4단계(검색 필요성, 문서 관련성, 답변 근거, 유용성)를 설명할 수 있다
- [ ] HyDE가 왜 짧은 쿼리의 검색 품질을 높이는지 수식으로 설명할 수 있다
- [ ] Multi-hop RAG에서 질문 분해의 필요성을 예시로 설명할 수 있다
- [ ] RAGAS 4개 지표(Faithfulness, Answer Relevancy, Context Recall, Context Precision)가 각각 무엇을 측정하는지 설명할 수 있다
- [ ] 각 지표가 낮을 때의 원인과 개선 방향을 설명할 수 있다
- [ ] LangGraph의 StateGraph, 노드, 조건부 엣지 개념을 설명할 수 있다

**구현 능력**

- [ ] LangGraph `StateGraph`로 조건부 분기가 있는 파이프라인을 구현할 수 있다
- [ ] Self-RAG의 4개 그레이더 노드를 직접 작성할 수 있다
- [ ] HyDE를 사용하는 커스텀 리트리버를 LangChain BaseRetriever로 구현할 수 있다
- [ ] Multi-hop RAG에서 질문 분해 → 순차 검색 → 최종 답변 생성 흐름을 구현할 수 있다
- [ ] RAGAS 평가 파이프라인을 설정하고 Ollama LLM을 사용하여 평가를 실행할 수 있다
- [ ] 여러 RAG 시스템을 비교하는 평가 스크립트를 작성할 수 있다

**실험 결과**

- [ ] 3가지 RAG 전략(Baseline, Hybrid, HyDE+Hybrid)의 RAGAS 점수 비교 완료
- [ ] Faithfulness 0.80 이상 달성 (Self-RAG 또는 강화된 프롬프트)
- [ ] Context Recall 0.80 이상 달성 (HyDE + 하이브리드)
- [ ] 개선 전후 비교 보고서 작성 완료

### 11.2 Level 3 이후 확장 주제

Level 3을 완성했다면, 다음 심화 주제를 탐구할 수 있다:

**검색 심화**
- Elasticsearch + Nori 형태소 분석기 (진정한 한국어 BM25)
- Dense Passage Retrieval (DPR) — 질문/문서를 별도 인코더로 임베딩
- ColBERT — 토큰 수준의 정밀 검색

**생성 심화**
- RAG with Citations — 인용 포함 답변 생성
- Speculative RAG — 병렬 검색으로 속도 향상
- Adaptive RAG — 질문 복잡도에 따른 자동 전략 선택

**인프라**
- K8s 기반 RAG 배포 (Level 4 AIOps)
- Kind 클러스터에서 Chroma StatefulSet
- Prometheus/Grafana RAG 모니터링

### 11.3 RAGAS 목표 점수 요약

```
지표               | Level 1 | Level 2 | Level 3 (목표)
─────────────────────────────────────────────────────
Faithfulness       |  ~0.70  |  ~0.75  |  ≥ 0.85
Answer Relevancy   |  ~0.72  |  ~0.78  |  ≥ 0.82
Context Recall     |  ~0.65  |  ~0.75  |  ≥ 0.80
Context Precision  |  ~0.68  |  ~0.76  |  ≥ 0.80
```
