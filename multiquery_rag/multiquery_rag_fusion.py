# RAG Fusion 적용
# 여러 개의 검색 쿼리에서 반환된 결과를 결합하여 최종적으로 가장 관련성 높은 문서를 선택하는 기법
# 이 방법은 다양한 쿼리로부터 도출된 문서들을 종합함으로써 중복된 문서를 제거하고 정확도를 높이는데 유용
# 이 기법의 핵심은 다양한 검색 결과를 결합해 순위를 재조정하고 이 과정에서 최종적인 관련 문서를 선별하는 것
# 사용하는 알고리즘은 RRF(Reciprocal Rank Fusion)
# 이 알고리즘은 여러 순위 리스트를 결합하는 효과적인 방법
# 순위가 높은 문서에 더 높은 가중치를 부여하며 여러 검색 쿼리의 결과를 종합하여 문서의 최종 순위를 계산
# RRF = 1 / k + 순위
# k는 정해진 상수로 일반적으로 60으로 설정.
# 순위가 낮을 수록(분모에 더해지는 값이 작을 수록 = 문서가 상위에 있을수록) 더 높은 점수를 부여

# 필요 모듈
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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

# split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=300,
  chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

# 벡터 스토어 생성
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 멀티 쿼리 생성
# 주어진 질문을 바탕으로 여러 개의 검색 쿼리를 생성
# 단일 쿼리로는 포착하기 어려운 다양한 문서를 검색할 수 있도록 함
template = """
당신은 주어진 하나의 질문을 기반으로 여러 검색 쿼리를 생성하는 유용한 조수입니다. \n
다음 질문과 관련된 여러 검색 쿼리를 생성하세요: {question} \n
출력 (4개의 쿼리):
"""
prompt_lag_fusion = ChatPromptTemplate.from_template(template)

# 템플릿을 바탕으로 사용자가 입력한 질문에 대해 4개의 서로 다른 쿼리 생성
generate_queries = (
  prompt_lag_fusion
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

  # 각 고유한 문서에 대한 점수를 저장할 변수 딕셔너리 초기화
  fused_scores = {}

  # 순위가 매겨진 문서 리스트를 반복 순회
  for docs in results:
    # 각 문서와 순위 추출
    for rank, doc in enumerate(docs):
      # 문서를 문자열 형식으로 직렬화
      doc_str = dumps(doc)
      # 해당 문서가 비어 있다면 초기 점수를 0으로 추가
      if doc_str not in fused_scores:
        fused_scores[doc_str] = 0
      # RRF 공식: 1 / (k + 순위)
      fused_scores[doc_str] += 1 / (k + rank)

  # 계산된 점수에 따라 내림차순으로 정렬하여 최종 재정렬 결과 반환
  reranked_results = [
    (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
  ]

  # 재정렬된 결과에서 우선 순위가 높은 top_n개 문서 반환
  return reranked_results[:top_n]

# Multiquery + RAG-Fusion 체인 구성
# 다양한 검색 쿼리를 생성하고 이 쿼리를 통해 얻은 문서들을 결합으로 최종적으로 가장 유의미한 문서들을 선정
# 1. 쿼리 생성: 주어진 질문에 대해 여러 검색 쿼리를 생성. 하나의 질문을 여러 방향으로 확장하여 검색 범위를 넓힘
# 2. 문서 검색: 생성된 각 쿼리에 대해 관련 문서들을 검색. 각 쿼리로부터 다양한 문서들이 반환되며, 이 문서들은 잠재적으로 중복되거나 유사한 내용이 포함될 수 있음
# 3. RRF 알고리즘 적용: 여러 쿼리로부터 검색된 문서들을 RRF 알고리즘을 통해 결합하여 최종 순위를 계산. 각 문서의 순위를 평가하고, 가장 중요한 문서들을 재정렬
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

# 질문에 대해 검색된 문서 호출
question = "SK그룹 채용에 대해 알려주세요"
docs = retrieval_chain_rag_fusion.invoke({'question': question})

# 문서 개수
# print('문서 개수: ', len(docs))

# print("=" * 50)

# 고유 문서
print('고유 문서', docs)

# 최종 RAG 체인 구성 및 실행
# 사용자가 입력한 질문에 대한 최종 답변 생성 구성
# RAG Fusion 과정을 통해 검색된 문서들을 바탕으로 LLM을 사용하여 답변을 생성하는 전체 프로세스를 자동화

template = """
다음 맥락을 바탕으로 질문에 답변해 주세요:
{context}
질문: {question}
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
final_rag_chain = (
  {'context': retrieval_chain_rag_fusion, 'question': RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

print(final_rag_chain.invoke({'question': question}))