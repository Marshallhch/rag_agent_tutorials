# 벡터: 
# 문자를 특정한 규칙에 의해 숫자로 변환한 것을 말한다. 이는 괄로 안에 콤마를 기준으로 나열 되며 주로 2차 또는 3차를 사용한다.
# 문자열을 벡터로 변환하면 단어 자체가 일치하지 않아도 질문과 문서의 의미를 특정한 알고리즘에 의해 이어줄 수 있다.

# 벡터 데이터베이스: 
# 벡터를 효율적으로 저장하고 쉽게 찾기 위한 장소를 말한다. 
# 일반 관계형 데이터베이스 또는 NoSQL 데이터에이스에도 저장할 수 있지만 이는 특화된 데이터베이스는 아니다.
# 일반적으로 벡터 데이터만을 효율적으로 저장할 수 있는 데이터베이스가 필요하다.
# 벡터 데이터베이스는 벡터 간의 거리 계산이나 유사도 계산을 효율적으로 수행할 수 있도록 설계됐다.
# 벡터 데이터베이스는 사용자 질문과 문서의 유사도를 계산하여 가장 연관도가 높은 문서를 빠르게 찾을 수 있다.
# 알려진 벡터 데이터베이스로는 Chroma, FAISS, Qdrant, Milvus 등이 있다.

# 임베딩:
# 텍스트, 이미지, 오디오 등의 데이터를 벡터로 변환하여 컴퓨터가 인식할 수 있게 만드는 과정이다.

# 데이터: 공유마당 - https://gongu.copyright.or.kr/gongu/main/main.do

# 텍스트 분할 -> 임베딩 -> 벡터 데이터베이스 저장 -> 검색기 설정 -> 프롬프트 구성 -> 체인 생성 -> 답변

# 모듈 설치
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

# 현재 파일 위치 기준으로 data 폴더 경로 설정
current_file = Path(__file__)
data_dir = current_file.parent / 'data'
pdf_path = data_dir / 'unsu.pdf'

# pdf 파일 로드
loader = PyPDFLoader(str(pdf_path))

# 페이지 분할
pages = loader.load_and_split()
# print(pages[0])

# 텍스트 분할기
# 임베딩된 텍스트는 고정된 크기의 벡터로 변환되는 과정이다.
# PDF 파일 내부의 한 페이지를 한꺼번에 임베딩하게 되면 관련 정보를 찾기 어려워질 수 있다.
# 따라서 의미있는 정보가 묶인 덩어리, 청크로 문서를 분할해야 한다.
# 랭체인에서는 문서를 분할하기 위하 방식으로 텍스트 분할기를 제공한다.
# 참조 문서: https://python.langchain.com/docs/how_to/recursive_text_splitter/

# RecursiveCharacterTextSplitter()
# params:
# - chunk_size: 한 청크의 최대 크기
# - chunk_overlap: 청크 간 중복 범위
# - length_function: 청크 크기를 결정하는 함수
# - is_separator_regex: 청크 분할 기준 문자열(False로 설정할 경우 구분자를 단순한 문자열로 해석)

# text_splitter.split_documents(): pages 리스트에 저장된 문서를 분할한다.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loader
loader = PyPDFLoader(str(pdf_path))
pages = loader.load_and_split()

# splitter
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=300,
  chunk_overlap=20,
  length_function=len,
  is_separator_regex=False
)

texts = text_splitter.split_documents(pages)
# print(texts[0])

# 임베딩 모델
# 임베딩 모델을 사용하여 텍스트 분할기로 분할한 문서 청크를 벡터로 변환한다.
# 임베딩 모델은 문서에서 연관된 정보를 정확하게 추출하기 위한 기반이다.
# 랭체인에서는 표준 임베딩 인터페이스를 통해 OpenAI, Cohere 등의 상용 임베딩 모델 뿐 아니라
# huggingface  등 오픈 소스 임베딩 모델까지 연결할 수 있다.
# 참조 문서: https://python.langchain.com/docs/integrations/text_embedding/

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 임베딩 모델 가격 참조: https://platform.openai.com/docs/pricing#embeddings

from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(
  model="text-embedding-3-large"
)

# 벡터 저장소
# 임베딩 벡터는 벡터 저장소를 통해 관리된다.
# 랭체인에서는 Chroma, Pinecone, Lance 등 여러 가지 벡터 저장소와 연동 기능을 제공한다.
# 참조 문서: https://python.langchain.com/docs/integrations/vectorstores/

from langchain_chroma import Chroma

db = Chroma.from_documents(texts, embedding_model) # 첫번째는 임베딩 대상 문자열, 두번째는 임베딩 실행 모델

# 검색기
# 벡터 저장소에서 문서를 검색하기 위한 검색기가 필요하다.
# 즉 임베딩된 PDF 문서에서 사용자 질문과 가장 관련 있는 정보만 가져오도록 해야 한다.
# 검색 성능을 높이기 위해 뉘앙스가 미묘하게 다른 단어도 의미를 잘 반영하기 위해
# 랭체인에서는 다중 질문 검색기 알고리즘을 지원한다.
# LLM을 사용하여 사용자 질문을 다양한 가짓수로 생성함으로써, 보다 관련선이 높은 결과를 제공하도록 구현한다.
# 참조 문서: https://python.langchain.com/docs/how_to/MultiQueryRetriever/

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# 텍스트 생성기 관련 모듈
from langchain import hub # 생성기 탬플릿
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough # 질의한 문장열을 그대로 전달하는 클래스

# Retriever
question = "아내가 먹고 싶어하는 음식은 뭐야?"
llm = ChatOpenAI(temperature=0) # temperature: 0 ~ 2 사이의 값 - 높으면 자유도가 높아진다.

# MultiQueryRetriever.from_llm() 함수로 MultiQueryRetriever 객체를 생성한다.
# 함수의 첫 번째 파라미터인 retriever는 db.as_retriever()로 Chroma 저장소에 대한 Retriever 인스턴스를 생성한 것이다.
# 두 번째 파라미터인 llm에는 앞서 만든 LLM 인스턴스(llm)를 전단한다.
# 이후 retriever_from_llm.invoke() 함수를 호출하여 사용자 질문에 대한 연관 정보를 가져온다.

retriever_from_llm = MultiQueryRetriever.from_llm(
  retriever=db.as_retriever(),
  llm=llm
)

# docs = retriever_from_llm.invoke(question)
# print(len(docs))
# print("=" * 50)
# print(docs)

# 현재 코드는 검색 단계와 생성 단계로 구분된다.
# 검색 단계에서는 사용자 질문과 연관된 정보를 검색하고,
# 생성 단계에서는 검색된 정보를 기반으로 답변을 생성한다.
# 이러한 구조를 RAG(Retrieval-Augmented Generation) 기법이라고 한다.

# 생성기
# 참조 문서: https://python.langchain.com/docs/tutorials/rag/#preview

# 사용자 질문에 대한 답변을 생성하기 위한 프롬프트 템플릿을 정의한다.
# 랭체인에서는 다른 사람들이 올린 프롬프트를 탐색하고 활용할 수 있는 프롬프트 허브를 제공한다.
# 프롬프트 허브를 사용하려면 langchain 라이브러리에서 hub 클래스를 불러온 뒤
# hub.pull() 함수의 인자로 프롬프트 경로("rlm/rag-prompt")를 전달한다.

# prompt template hub
prompt = hub.pull('rlm/rag-prompt')
# print(prompt)

# generate: 검색된 문서를 합치고 답변을 생성하는 LLM 체인을 만든다.
# - format_docs()를 통해 검색된 문서를 2개 줄바꿈으로 구분하여 결합한다.
# - LLM 체인을 만들기 위해 프롬프트 인스턴스(prompt), LLM 인스턴스(llm), 출력파서(StrOutputParser())를 연결한다.
# - context에 검색된 문서를 결합한 결과(retriever_from_llm | format_docs)를 question에 사용자 질문(RunnablePassthrough())을 전달한다.

# RunnablePassthrough(): 랭체인의 문법으로 invoke() 함수의 사용자 입력을 그래도 전달하는 역할을 한다.
# 예를 즐어, rag_chain.inovie('아내가 먹고 싶어하는 음식은 무엇이야?') 와 같이 사용자 질문을 전달하면
# '아내가 먹고 싶어하는 음식은 무엇이야?' 라는 문자열을 LLM 체인에 그대로 전달된다.

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
  {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

# rag_chain.invoke()를 통해 LLM 체인을 실행하여 답변 생성
result = rag_chain.invoke('아내가 먹고 싶어하는 음식은 뭐야?')
print(result)