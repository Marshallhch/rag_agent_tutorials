# Unique-union 전략은 Multi-query를 통해 수집한 여러 결과 중 중복되는 문서를 제거하고 유일한 문서만 남기는 방식
# 이를 통해 유사한 내용을 반복하지 않고 풍부한 정보만 남길 수 있다.
# 또한 GPT가 중복된 내용을 읽고 중요한 정보라고 잘못 판단하는 현상을 방지한다.
# 참조: https://whatthaburger.tistory.com/143

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

# 프롬프트 템플릿 생성
template = """
당신은 AI 언어 모델 조수 입니다. 당신의 임무는 주어진 사용자 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다.
사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는 것이 목표입니다.
각 질문은 세 줄로 구분하여 제공하세요. 원본 질문: {question}
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
  prompt_perspectives
  | ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
  | StrOutputParser()
  | (lambda x: x.split("\n")) # 전달받은 템플릿 문자열을 줄바꿈 기준으로 분리
)

# generate_queries = generate_queries.invoke("코스피 전망은?")

# 입력된 질문과 관련하여 다양한 각도에서 5가지 질문이 생성
# print(generate_queries)

# 여러 리스트에 포함된 문서들 중에서 중복된 문서를 제거하고, 고유한 문서들만 반환
def get_unique_union(documents: list[list]):
  """고유한 문서들의 합집합을 생성"""

  # 리스트를 평탄화하고, 각 문서의 문자열로 직렬화
  flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]

  # 중복된 문서를 제거하고 고유한 문서만 추출
  unique_docs = list(set(flattened_docs))

  # 고유한 문서를 원래의 문서 객체로 변환
  return [loads(doc) for doc in unique_docs]

question = '코스피 전망은?'

# generate_queries: 사용자가 입력한 질문을 바탕으로 다양한 관점에서 다섯가지의 검색 쿼리 생성
# retriever.map(): 생성된 각 쿼리를 사용하여 관련 문서를 검색. 각 쿼리는 개별적으로 문서 검색을 수행하며, 결과로 문서 리스트 반환
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# retrieval_chain 실행하고 고유한 문서를 반환. <- 중복이 제거된 문서 리스트 생성
docs = retrieval_chain.invoke({'question': question})
# print(len(docs))

# final_rag_chain
# 사용자가 입력한 질문과 관련된 문서를 검색하고, 이를 바탕으로 LLM을 사용하여 답변을 생성 자동화
# 프롬프트 템플릿을 사용하여 LLM에 적절한 입력을 제공하고 검색된 문서와 결합된 정보를 바탕으로 답변을 생성
# 최종적으로 사용자는 질문에 대한 관련성 높은 답변을 받을 수 있음

template = """
다음 맥락을 바탕으로 질문에 답하세요:
{context}
질문: {question}
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
final_rag_chain = (
  {'context': retrieval_chain, 'question': RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

print(final_rag_chain.invoke(question))

# 정리
# 1. retrival_chain: 사용자 질문을 바탕으로 다양한 관점에서 다섯가지의 검색 쿼리 생성. context로 설정
# 2. RunnablePassthrough(): 입력 변수를 그대로 전달하는 역할. 즉, 사용자 질문을 그대로 전달. question으로 설정
# 3. 프롬프트 생성: 설정된 context와 question을 사용하여 프롬프트 생성. 이 프롬프트는 앞서 정의한 템플릿을 기반으로 만들어지며, LLM에게 전달
# 4. 답변 생성: 생성된 프롬프트는 llm에 전달되며, 이를 바탕으로 답변을 생성
# 5. 출력 파싱: 최종적으로 생성된 답변은 StrOutputParser()를 사용하여 문자열로 변환