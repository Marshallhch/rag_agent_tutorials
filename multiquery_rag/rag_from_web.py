# 필요 모듈
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# web based loader 문서: https://wikidocs.net/231644
# 예시 URL: https://news.naver.com/section/101
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

# 문서를 토큰 단위로 분할
# ticktoken 인코더 사용
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=300,
  chunk_overlap=50
)

splits = text_splitter.split_documents(docs)

# 임베딩 후 벡터 스토어 저장
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# MMR(Maximal Marginal Relevance) 알고리즘은 정보 검색에서 다양성을 확보하면서도 사용자 쿼리와 가장 관련성이 높은 문서를 선택
# 이 알고리즘은 유사성이 높지만 서로 중복되지 않는 문서를 선택하여 정보의 다양성을 보장한다.
# search_type="mmr": 이 설정은 검색 과정에서 선택된 문서들 사이의 다양성을 높이면서도 사용자 쿼리에 가장 관련성 높은 문서를 우선순위로 둔다.
# 'fetch_k': 전체 문서 중에서 50개의 문서를 먼저 선택한 후 이 중에서 최종적으로 반환할 k개의 문서를 결정
# 이 과정에서 다양성과 관련성을 모두 고려하여 가장 적합한 문서를 반환

# 최종 50개 문서를 고려하여 검색 수행 후 5개 문서 반환
# retriever = vectorstore.as_retriever(
#   search_type='mmr',
#   search_kwargs={'k': 5, 'fetch_k': 50}
# )

# 유사도 점수가 0.8 이상인 문서들만 검색하여 반환
# 정확도가 높은 문서만 가져오고자 할 때 유용
retriever = vectorstore.as_retriever(
  search_type='similarity_score_threshold',
  search_kwargs={"score_threshold": 0.8}
)

# 질문에 대한 답변 생성
prompt = PromptTemplate(
  template="""질문-답변 작업을 돕는 비서입니다. 질문에 답하기 위해 다음과 같은 검색 결과를 사용하세요. 답을 모르는 경우에는 모른다고 명시하세요. 답변은 세 문장 이내로 간결하게 유지해 주세요.
  질문: {question}
  검색 결과: {context}
  답변:""",
    input_variables=["question", "context"]
)

# RAG 체인 구성
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

def format_docs(docs):
  formatted = "\n\n".join(doc.page_content for doc in docs)
  return formatted

# 체인
rag_chain = (
  {'context': retriever | format_docs, 'question': RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

# 질문에 대한 답변 생성
answer = rag_chain.invoke("현대건설 관련 기사 검색해줘")

# 검색된 원본 문서
docs = retriever.invoke("현대모비스 관련 기사 검색해줘")
print(docs)