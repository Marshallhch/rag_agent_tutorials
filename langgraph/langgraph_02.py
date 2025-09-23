# 다양한 도구와 프롬프트를 활용해 정보를 검색, 가공, 응답하는 RAG 시스템을 그래프 형태로 구성

# Agentic RAG 흐름
# 사용자의 질문이 입력되면 에이전트는 적절한 도구(tools)를 사용하여 정보를 검색(retrieve)하고, 이를 기반으로 답변을 생성(generate)한다. 
# 에이전트는 검색 결과의 관련성을 평가한 후, 경우에 따라 질문을 재작성(rewrite)하여 더욱 정확한 정보를 제공할 수 있다.

# start -> agent -> tools -> retrieve -> generate -> end
# start -> agent -> tools -> retrieve -> rewrite -> agent -> tools -> retrieve -> generate -> end
# agent: 사용자의 질문을 받아 처리하는 추체로 여러 도구와 상호작용한다.
# tools: 정보 검색에 사용되는 도구들로, 데이터베이스나 외부 API에서 관련 정보를 검색한다.
# retrieve: 도구를 활용해 사용자의 질문과 관련된 문서를 검색하는 과정이다.
# rewrite: 검색된 결과를 바탕으로 질문을 재구성해 보다 정확한 답변을 생성하는 단계다
# generate: 최종적으로 사용자의 질문에 대한 답변을 생성하는 단계다.

# 위 과정에서 에이전트는 사용자의 질문을 여러 번 처리하고, 필요한 경우 관련된 질문을 재작성하여 보다 정교한 결과를 제공하게 된다.
# 이를 통해 사용자는 더욱 정확하고 관련성 높은 응답을 받는다
# 또한 에이전트는 동적으로 자신의 행동을 제어하고 모니터링하여 신뢰성 있는 질의응답 시스템을 구현할 수 있다.

# OpenAI Key
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 금융 정보를 제공하는 웹 피이지를 크롤링하고 텍스트를 분할하여 벡터 스토어에 저장한다.
# RAG 관련 모듈들
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 에이전트의 상태를 정의하고, 문서 검색을 위한 도구를 생성
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# 크롤링할 웹페이지 목록
urls = [
  "https://finance.naver.com/",
  "https://finance.yahoo.com/",
  "https://finance.daum.net/",
]

# 각 URL에서 문서 로드
docs = [WebBaseLoader(url).load() for url in urls]
print(docs)
