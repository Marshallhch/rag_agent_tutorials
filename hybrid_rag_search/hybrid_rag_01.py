# 서로 다른 검색 알고리즘을 조합하여 검색 정확도와 효율성을 높이는 기법
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import text_splitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF 문서 로드 후 텍스트로 변환. 
pdf_path = os.path.join('hybrid_rag_search', 'sample.pdf')
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# print(pages)

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=300,
  chunk_overlap=50
)
texts = text_splitter.split_documents(pages)

# print(texts)

# Embedding
embeddings_model = OpenAIEmbeddings()

# 벡터 스토어 생성
vectorstore = Chroma.from_documents(texts, embeddings_model)
chroma_retriever = vectorstore.as_retriever(
  search_type='mmr',
  search_kwargs={"k": 1, 'fetch_k': 4}
)

# BM25(Best Matching 25) 검색기
# 전통적인 정보 검색 모델로, TF_IDF와 유사한 방식으로 작동한다.
# 사용자가 입력한 검색 질의와 문서 간의 관련성을 점수화하여 문서의 순위를 매긴다.
# 문서 내 검색 단어의 빈도와 문서의 전체 길이를 고려하여 과도하게 긴 문서가 높은 점수를 받지 않도록 한다.

# 주요 특징
# 1. 단어 빈도(TF): 특정 단어가 문서에서 얼마나 자주 등장하는지 계산. 특정 단어가 많이 등장할수록 그 단어는 문서의 주요 주제를 나타낸다고 판단
# 2. 역문서 빈도(IDF): 특정 단어가 전체 문서 집합에서 얼마나 흔하지 않은지 측정. 흔하지 않은 단어일수록 그 단어는 중요한 정보로 간주
# 3. 문서 길이 정규화: 문서 길이를 보정하는 기능. 너무 긴 문서는 불리하게, 너무 짧은 문서는 유리하게 가중치 조절. 문서가 길어질수록 단지 단어 수가 많다는 이유만으로 쿼리 단어가 더 자주 등장할 수 있어, 실제 관련성과 무관하게 높은 점수를 받을 위험을 방지

# 이 알고리즘은 검색 엔진, 정보 검색 시스템 등에서 사용되고 있다.
# 예를 들어 네이버, 구글과 같은 대형 검색 엔진에서도 사용되며 특히 대규모 문서 데이터베이스에서 검색어의 관련성을 빠르고 정확하게 평가하는데 도움을 준다.

# 하이브리드 검색 시스템 구현
# EnsembleRetriever: 여러 개의 검색기를 결합하여 다양한 검색 기법의 장점을 모두 활용할 수 있도록 하는 역할
# retrievers=[bm25_retriever, chroma_retriever]: 두 가지 검색 기법, 즉 BM25 검색기와 Chroma 기반 벡터 검색기 결합
# weights=[0.2, 0.8]: 각 검색기의 가중치를 설정. 즉, 0.2는 BM25 검색기의 가중치, 0.8은 Chroma 검색기의 가중치를 의미
# 이 가중치는 검색 결과에 미치는 영향을 결정하며, 높은 가중치는 해당 검색기의 결과가 더 강하게 반영됨을 의미

from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k=2

# 알고리즘 조합
ensemble_retriever = EnsembleRetriever(
  retrievers=[bm25_retriever, chroma_retriever],
  weights=[0.2, 0.8]
)

# 예시 문의
query = "에코프로에 대해 알려줘"

# 관련 문서 검색
docs = ensemble_retriever.invoke(query)

# print(docs)
# for doc in docs:
#   print(doc.page_content)

# 프롬프트 생성
template = """
다음 맥락을 바탕으로 질문에 답변해 주세요:
{context}
질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

# 전처리
def format_docs(docs):
  formatted = "\n\n".join(doc.page_content for doc in docs)
  return formatted

final_rag_chain = (
  {'context': ensemble_retriever, 'question': RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

# 답변 출력
print(final_rag_chain.invoke(query))
