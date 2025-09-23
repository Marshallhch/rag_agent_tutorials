# 랭체인 에이전트는 LLM이 주어진 목적을 달성하기 위해 여러 개의 도구(Tool)을 순차적으로 선택하고 실행하도록 구성된 동적 실행 엔진
# 에이전트는 사용자의 입력에 따라 어떤 도구를 사용할지 스스로 결정하고, 각 도구에서 얻은 정보를 바탕으로 다음 행동을 계획하여 최종 응답을 생성

# 시스템의 주요 구성 요소

# 1. LLM: 시스템의 중심으로, 사용자가 입력한 질문을 이해하고 적절한 도구를 호출해 데이터를 처리한다. 
# 2. Parser: LLM이 사용자 질문을 분석한 후 적절한 도구를 호출하는 역할을 한다. 예를 들어 논문 검색과 관련된 질문을 받으면 지정된 논문 API를 호출하는 방식이다.
# 3. Tools: 시스템에서 활용되는 실제 API들이며 각각의 툴은 특정한 목적을 가지고 데이터를 처리한다.
# 4. Observation: 툴이 반환한 결과를 LLM이 다시 분석하고 해석되는 단계다. 각 툴이 제공한 데이터를 바탕으로 LLM은 최종적인 답변을 만든다.
# 5. 최종 답변: LLM이 도구로부터 받은 데이터를 바탕으로 사용자의 질문에 대한 답변을 생성하는 단계다. 사용자가 원하는 정보를 얻을 때까지, 이 과정은 툴 호출과 관찰 단계를 반복할 수 있다.

# LangChain Hub
# 프롬프트, 체인, 에이전트 등 랭체인의 핵심 구성 요소들을 공유하고 탐색하며 관리할 수 있는 오픈 커뮤니티 플랫폼이다.
# 이를 통해 사용자는 실무에 활용 가능한 다양한 프롬프트와 체인을 빠르게 확인하고 직접 다운로드하거나, 자신이 작성한 프롬프트를 업로드하여 기여할 수 있으며, 파이썬 또는 다른 SDK를 통해 손쉽게 push/pull 기능을 사용할 수 있다.

# 필요 모듈 설치

# agent 생성을 취한 모듈
from langchain.agents import AgentExecutor

# 벡터 DB를 agent에게 전달하기 위한 tool
from langchain.agents import create_openai_tools_agent

# 랭체인 허브
from langchain import hub

# arXiv 논문 검색을 위한 tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

# 네이버 뉴스에서 최신 기사를 로드하고 이를 벡터화하여 검색이 가능한 형태로 만든다
# OpenAI의 임베딩을 사용하여 FAISS 벡터 스토어에 저장하고, 이 데이터를 검색할 수 있다.

# 벡터 DB 구축 및 검색 도구
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 벡터 DB 모듈 및 웹 크롤링 모듈
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# 질의 주제 검색: 위키피디아
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# OpenAI 모듈
from langchain_openai import ChatOpenAI

# OpenAI Key
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 구성
openai = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=OPENAI_API_KEY)

# 프롬프트 탬플릿은 사용자의 입력을 처리하고 그에 따라 툴을 호출할 수 있도록 에이전트의 동작 방식을 정의한다.
# agent 시뮬레이션을 위한 prompt 참조
# hub에서 가져온 prompt를 agent에게 전달하기 위한 prompt 생성
# LangChain Hub = "GitHub + HuggingFace + Registry" 개념의 중앙 저장소
# hub.pull("username/item-name") = 특정 유저가 만든 리소스를 LangChain 환경으로 불러옴
# "hwchase17": 이 리소스를 만든 사용자의 이름 (LangChain 창시자 Harrison Chase의 유저명)
# "openai-functions-agent": 해당 사용자가 공유한 체인의 이름 (OpenAI 함수 호출 기반 Agent 설정)

# 프롬트트 구성
prompt = hub.pull("hwchase17/openai-functions-agent")

# 1. wikipedia 도구 툴
# WikipediaAPIWrapper를 사용하여 API 호출 시 반환할 결과를 개수(top_k_results)와 문서의 최대 본문길이(doc_content_char_max)를 각각 1과 200으로 제한
# 설정한 API 래퍼를 인자로하여 WikipediaQueryRun을 초기화
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
# print(wiki.name)

# 2. 네이버 뉴스 로딩 및 문서 분할
loader = WebBaseLoader('https://news.naver.com')
docs = loader.load()

# 문서를 1000자 청크로 나누고 중첩은 200글자
documents = RecursiveCharacterTextSplitter(
  chunk_size=2000,
  chunk_overlap=200
).split_documents(docs)

# 3. 벡터 DB 생성 및 검색기 생성
# 위에서 분할한 문서를 OpenAI의 임베딩 모델을 사용하여 벡터로 변환한 후, 이를 FAISS 벡터 스토어에 저장. 이후 저장된 벡터 데이터베이스를 통해 검색할 수 있도록 retriever 검색기로 변환
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever() # 리트리버 검색기

# 4. 검색 도구 정의 툴
# 검색된 네이버 뉴스 데이터를 처리하기 위한 툴 정의 이 툴은 사용자가 질문했을 때 해당 DB에서 검색 결과를 반환.
# create_retriever_tool 함수는 랭체인에서 검색 기반 툴을 생성할 때 사용된다. 질문에 대한 내용을 벡터 데이터베이스에서 유사 문서를 검색하는 retriever 객체를 활용해 관련 뉴스 기사를 찾아 반환한다.
# naver_news_search는 이 툴의 내부 식별자로, 에이전트가 툴을 선택할 때 사용되며 마지막 문자열은 사용자 질문과 연결될 설명으로써 이 툴이 어떤 용도이며 언제 사용해야 하는지 에이전트에게 명확히 안내하는 역할을 한다.
retriever_toos = create_retriever_tool(
  retriever, 
  "naver_news_search",
  "네이버 뉴스 정보가 저장된 벡터 DB 입니다. 당일 기사에 대해 궁금하면 이 툴을 사용하세요."
)

# 5. arXiv 논문 검색 툴 설정
# arXiv는 최신 논문을 검색할 수 있는 오픈 데이터베이스다. 이 API를 사용해 최신 논문을 검색하고 필요한 정보를 검색하는 도구를 구현한다. arXiv API를 호출할 때 최대 1개의 결과만 반환하고 반환된 문서의 본문 길이를 200자로 제한한다.
# arXiv API: top_k_results=결과수, doc_content_char_max=문서 길이 제한
arxiv_wrapper = ArxivAPIWrapper(
  top_k_results=1,
  doc_content_chars_max=200,
  load_all_available_meta=False # 메타데이터는 가져오지 않음
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# 6. 에이전트 툴 통합
# LLM 모델을 사용해 사용자가 입력한 질문을 처리하고, 필요한 도구들을 호출하여 데이터를 검색한 후 최종 답변을 반환하는 에이전트 정의
# 여기서 사용하는 도구는 Wikipedia, 네이버 뉴스 검색기, arXiv 도구를 통합
tools = [wiki, retriever_toos, arxiv]

# llm 모델을 openai로 정의하고, tools, prompt를 입력하여 agent를 완성한다.
agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)

# agent를 실행하여 사용자의 질문에 대해 답변을 생성
# verbose=True로 설정하여 에이전트가 각 단계를 어떻게 처리하는지 출력
# agent Execute 정의 부분
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_result = agent_executor.invoke({"input": "오늘 코스피 관련 소식을 알려주세요."})

# 7. 결과 출력
print(agent_result)