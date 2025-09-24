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
docs_list = [item for sublist in docs for item in sublist]

# 문서 분할 설정
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=300,
  chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 벡터 스토어에 문서 추가
vectorstore = Chroma.from_documents(
  documents=doc_splits,
  collection_name='rag-chroma',
  embedding=OpenAIEmbeddings()
)

# 검색기 생성: 에이전트 툴에 전달하여 사용
retrivever = vectorstore.as_retriever()

# 에이전트 상태를 나타내는 데이터 구조 정의
class AgentState(TypedDict):
  # add_messages 함수는 업데이트가 어떻게 처리되어야 하는지를 정의. 기본은 대체, add_messages는 추가
  messages: Annotated[Sequence[BaseMessage], add_messages]

# 검색 도구 툴에 전달 생성
retriever_tool = create_retriever_tool(
  retrivever,
  'retrive_blog_posts',
  '네이버, 야후, 다음의 금융 관련 정보를 검색하고 반환합니다.'
)

tools = [retriever_tool]

# 검색된 문서가 질문과 관련성이 있는지를 평가하는 함수
def grade_documents(state) -> Literal['generate', 'rewrite']:
  """
  Args:
    state(messages): 현재 상태
  Returns:
    str: 문서의 관련성에 따라 다음 노드 결정('generate' or 'rewrite')
  """

  # 데이터의 모델 타입 정의
  class grade(BaseModel):
    """
    관련성 평가를 위한 이진 점수
    """
    # Field: 데이터 모델의 필드를 정의
    binary_score: str = Field(description="관련성 점수 'yes' 또는 'no'")

  # LLM 모델
  model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=OPENAI_API_KEY, streaming=True)

  # LLM에 모델 데이터 타입 적용
  llm_with_tool = model.with_structured_output(grade)

  # 프롬프트 구성
  prompt = PromptTemplate(
    template="""당신은 사용자 질문에 대한 검색된 문서의 관련성을 평가하는 평가자입니다.\n
    여기 검색된 문서가 있습니다:\n\n{context}\n\n
    여기 사용자 질문이 있습니다: {question}\n
    문서가 사용자 질문과 관련된 키워드 또는 의미를 포함하면 관련성이 있다고 평가하세요.\n
    문서가 질문과 관련이 있는지 여부를 나타내기 위해 'yes' 또는 'no'로 이진 점수를 주세요.""",
    input_variables=["context", "question"],
  )

  # 체인 생성
  chain = prompt | llm_with_tool

  messages = state["messages"]
  last_message = messages[-1]

  # 질의
  question = messages[0].content
  docs = last_message.content

  # 질문 점수 평가
  scored_result = chain.invoke({"question": question, "context": docs})
  score = scored_result.binary_score # yes or no

  if score == 'yes':
    print("--- 결정: 문서 관련성 있음 ---")
    return 'generate'
  else:
    print("--- 결정: 문서 관련성 없음 ---")
    return 'rewrite'

# 현재 상태를 기반으로 에이전트 모델을 호출하여 응답을 생성
# 주어진 질문에 따라 검색 도구를 사용하여 검색을 수행하거나 단순히 종료를 결정하는 함수
def agent(state):
  """
  Args:
    state: 현재 상태
  Returns:
    dict: 메시지에 에이전트 응답이 추가된 업데이트 상태
  """

  print("--- 에이전트 호출 ---")
  messages = state['messages']

  # 메시지가 제대로 전달되고 있는지 확인
  print('에이전트로 전달된 메시지: ', messages)

  # 모델 구성
  model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=OPENAI_API_KEY, streaming=True)
  model = model.bind_tools(tools)
  response = model.invoke(messages)

  # 응답을 상태에 추가
  state['messages'].append(response)
  return state

# 질문을 재작성하여 더 명확하고 효과적인 질문을 생성하여 LLM을 통한 질문을 개선하는 함수
def rewrite(state):
  """
  Args: 
    state: 현재 상태
  Returns:
    dict: 재구성된 질문으로 업데이트된 상태
  """

  print("--- 질문 변경 ---")
  messages = state['messages']
  question = messages[0].content

  # 새로운 질문에 따른 내용을 모델에 전달해서 다시 응답을 생성
  msg = [
    HumanMessage(
      content=f"""
        다음 입력을 보고 근본적인 의도나 의미를 파악해 주세요.\n
        초기 질문은 다음과 같습니다:
        \n----------\n
        {question}
        \n----------\n
        개선된 질문을 만들어 주세요:
      """
    )
  ]

  # 새로운 질문 평가
  model = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=OPENAI_API_KEY, streaming=True)
  response = model.invoke(msg)

  # 반환되는 메시지가 올바른지 확인
  print('Rewrite 단계에서의 응답: ', response)

  # 상태 업데이트
  state['messages'].append(response)
  return state

# 관련성 있는 문서를 기반으로 최종 답변을 생성하는 함수
def generate(state):
  """
  Args:
    state: 현재 상태
  Returns:
    dict: 재구성된 질문으로 최종 답변 업데이트
  """

  print("--- 최종 생성 ----")
  messages = state['messages']
  question = messages[0].content
  last_message = messages[-1]

  docs = last_message.content

  # 프롬프트 정의
  prompt = PromptTemplate(
  template="""당신은 질문-답변 작업을 위한 어시스턴트입니다. 
  아래 제공된 문맥을 사용하여 질문에 답변해주세요. 
  답을 모를 경우 '모르겠습니다'라고 말해주세요. 답변은 최대 3문장으로 간결하게 작성하세요.
  
  질문: {question}
  문맥: {context}
  답변: """,
  input_variables=["context", "question"],
  )

  # 모델 생성
  llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=OPENAI_API_KEY, streaming=True)

  # 체인 구성
  rag_chain = prompt | llm | StrOutputParser()

  # 답변 실행
  response = rag_chain.invoke({'context': docs, 'question': question})
  return {"messages": [response]}

# ================ workflow 구성 및 답변 생성 ======================#

# 순환 그래프 정의 초기화
workflow = StateGraph(AgentState)

# 순화 노드 추가
workflow.add_node('agent', agent) # 에이전트 노드

retrieve = ToolNode([retriever_tool])

# 검색 도구 노드 추가
workflow.add_node('retrieve', retrieve)

# 질문 재작성 노드 추가
workflow.add_node('rewrite', rewrite)

# 관련성 있다고 판단했을 때 최종 응답 노드 추가
workflow.add_node('generate', generate)

# 초기 진입점 설정
workflow.add_edge(START, 'agent')

# 검색 여부를 결정
workflow.add_conditional_edges(
  'agent',
  tools_condition, # 에이전트 결정 평가
  {
    'tools': 'retrieve',
    END: 'generate'
  }
)

# 검색 후 문서 관련성 평가
workflow.add_conditional_edges(
  'retrieve',
  grade_documents, 
  {
    'generate': 'generate',
    'rewrite': 'rewrite'
  }
)

workflow.add_edge('generate', END)
workflow.add_edge('rewrite', 'agent')

# 그래프 컴파일
graph = workflow.compile()

# 그래프 실행 및 결과 확인
import pprint

inputs = {
  "messages": [
    ("user", "코스피 전망에 대한 기사를 요약해 주세요.")
  ]
}

for output in graph.stream(inputs):
  for key, value in output.items():
    pprint.pprint(f'노드 "{key}" 출력 결과: ')
    pprint.pprint("--------")
    pprint.pprint(value, indent=2, width=80, depth=None)
  pprint.pprint("\n-----------\n")