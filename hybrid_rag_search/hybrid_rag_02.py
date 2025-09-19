import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import text_splitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.load import dumps, loads

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = WebBaseLoader(
    web_paths=("https://news.naver.com/section/101",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("sa_item_SECTION_HEADLINE", "sa_text")
        )
    ),
)
docs = loader.load()
# print(docs)

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=300,
  chunk_overlap=50
)

splits = text_splitter.split_documents(docs)

# 임베딩 생성 및 벡터 스토어 설정
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
chroma_retriever = vectorstore.as_retriever(
  search_type='mmr',
  search_kwargs={"k": 1, 'fetch_k': 4}
)

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k=2

# 알고리즘 조합
ensemble_retriever = EnsembleRetriever(
  retrievers=[bm25_retriever, chroma_retriever],
  weights=[0.2, 0.8]
)

# 질문 쿼리
query = "대기업 고용 현황에 대해 알려주세요."

# 관련 문서 검색
docs = ensemble_retriever.invoke(query)

# print(docs)

# 사용자의 질문에 대해 여러 개의 쿼리를 생성하는 체인 구성

template = """
당신은 AI 언어 모델 조수 입니다. 당신의 임무는 주어진 사용자 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다.
사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는데 도움을 주는 것이 목표입니다.
각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

# 템플릿을 바탕으로 사용자가 입력한 질문에 대해 4개의 서로 다른 쿼리 생성
generate_queries = (
  prompt_perspectives
  | ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
  | StrOutputParser()
  | (lambda x: x.split("\n")) # 전달받은 템플릿 문자열을 줄바꿈 기준으로 분리
)

# RRF 적용 함수
def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
  """
  여러개의 순위가 매겨진 문서 리스트를 받아, RRF 공식을 사용해 문서의 최종 순위 계산
  Args:
    results: 순위가 매겨진 문서 리스트
    k: 공식에 사용되는 상수
    top_n: 반환할 우선순위가 높은 문서의 개수
  Returns:
    reranked_results: 알고리즘에 따라 재정렬된 문서 리스트
  """
  fused_scores = {}

  for docs in results:
    for rank, doc in enumerate(docs):
      doc_str = dumps(doc)
      if doc_str not in fused_scores:
        fused_scores[doc_str] = 0
      fused_scores[doc_str] += 1 / (k + rank)

  reranked_results = [
    (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
  ]

  return reranked_results[:top_n]

# RAG_Fusion
retrieval_chain_rag_fusion = generate_queries | ensemble_retriever.map() | reciprocal_rank_fusion

# 질문에 대해 검색된 문서 호출
question = "대기업 고용 현황에 대해 알려주세요."
docs = retrieval_chain_rag_fusion.invoke({'question': question})

# print(len(docs))
# print('=' * 50)
# print(docs)

template = """
다음 맥락을 바탕으로 질문에 답변해 주세요:
{context}
질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
final_rag_chain = (
  {'context': retrieval_chain_rag_fusion, 'question': RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

print(final_rag_chain.invoke({'question': question}))