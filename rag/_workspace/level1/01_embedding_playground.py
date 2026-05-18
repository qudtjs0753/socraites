"""
실습 1: 임베딩 + 코사인 유사도 직접 체험
핵심 개념: 텍스트 → 1024차원 벡터(BGE-M3) → 코사인 유사도
실행: python 01_embedding_playground.py
"""

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

# 비교할 문장들 — 의미가 비슷한 그룹과 다른 그룹을 섞어 봅니다
texts = [
    "RAG는 외부 문서를 활용해 LLM의 답변을 보강하는 기법입니다",
    "검색 증강 생성은 LLM에 최신 지식을 주입하는 방법입니다",   # 1번과 유사 (의미)
    "리트리벌은 벡터 DB에서 관련 청크를 가져오는 단계입니다",   # 1번과 약한 관련
    "오늘 서울 날씨가 맑고 기온은 22도입니다",                  # 1번과 무관
    "쿠버네티스 Pod이 CrashLoopBackOff 상태입니다",            # 1번과 무관
]

# 배치로 한 번에 임베딩 생성
vectors = [np.array(v) for v in embeddings.embed_documents(texts)]

print(f"임베딩 차원: {len(vectors[0])}")  # BGE-M3: 1024
print(f"벡터 첫 5개 값(샘플): {vectors[0][:5]}\n")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"[기준 문장] {texts[0]}\n")
for i in range(1, len(texts)):
    sim = cosine_similarity(vectors[0], vectors[i])
    bar = "#" * int(sim * 40)
    print(f"  유사도 {sim:.3f} | {bar}")
    print(f"           {texts[i]}\n")
