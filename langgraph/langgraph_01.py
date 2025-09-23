# 랭그래프: 
# - 단일 에이전트, 다중 에이전트, 계층적 또는 순차적 작업 흐름을 쉽게 구현힐 수 있는 유연한 API를 제공한다.
# - 이를 통해 복잡한 작업들을 안정적으로 처리할 수 있다.
# - 특히 에이전트의 행동을 세밀하게 제어하고, 오류나 비정상적인 동작을 방지하기 위한 모니터링 및 품질 보증 루프를 간편하게 추가할 수 있다.
# - 랭체인과 랭스미스와의 통합을 지원하지만 독립적으로 사용할 수도 있다.
# - 내장된 지속성 기능을 통해 상태를 유지하고 Human-in-the-Loop 기능을 활성화해 더욱 정교한 에이전트 시스템을 구축할 수 있다.

# -- 사이클 및 분기: 애플리케이션에서 루프와 조건문을 구현할 수 있다.
# -- 지속성: 그래프의 각 단계 후 자동으로 상태를 저장한다. 그래프 실행을 언제든지 일시 중지하고 다시 시작하여 오류 복구, Human-in-the-Loop 워크플로, 시간 여행 등을 지원한다.
# -- Human-in-the-Loop: 개발 및 운영 과정에 인간이 참여하여 시스템의 성능을 개선하고 신뢰성을 높이는 방식. 에이전트가 계획한 다음 작업을 승인하거나 수정하기 위해 그래프 생성을 중단할 수 있다.
# -- 스트리밍 지원: 각 노드에서 생성된 출력(토큰 스트리밍 포함)을 실시간으로 스트리밍할 수 있다.

# OpenAI Key
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# - Annotated: 타입에 메타데이터를 추가할 수 있게 해주는 도구 (예: Pydantic 모델, Tool 정의 등에서 사용)
# - Literal: 리터럴 값 중 하나만 허용하는 타입 지정 (예: Literal["yes", "no"])
# - TypedDict: 딕셔너리의 키-값 구조에 타입을 지정해주는 클래스 (정적 타입 검사에 유용)
from typing import Annotated, Literal, TypedDict

# - HumanMessage: LangChain에서 사람이 보낸 메시지를 나타내는 클래스
#   예: 대화형 LLM 입력에서 사용됨 (LLMChain, Agent 등)
from langchain_core.messages import HumanMessage

# - ChatOpenAI: OpenAI의 Chat API (gpt-3.5, gpt-4 등)를 사용하는 LangChain용 래퍼 클래스
#   예: LLM 대화 수행 시 이 객체를 사용
from langchain_openai import ChatOpenAI

# 메모리 체크포인터 - 메모리 체크포인터는 상태 그래프의 각 단계 후 자동으로 상태를 저장하고, 그래프 실행을 언제든지 일시 중지하고 다시 시작하여 오류 복구, Human-in-the-Loop 워크플로, 시간 여행 등을 지원한다.
from langgraph.checkpoint.memory import MemorySaver 

# - @tool: 특정 함수(또는 클래스)를 LLM이 사용할 수 있는 "Tool"로 등록하는 데 사용
#   예: 에이전트가 계산기, 검색기 등을 사용할 수 있도록 연결
from langchain_core.tools import tool

# - END: 상태 머신 그래프에서 "종료 상태"를 나타냄
# - StateGraph: 상태 기반 흐름을 정의할 수 있는 LangGraph의 핵심 클래스
# - MessagesState: 메시지 기반 상태 (대화 흐름을 메시지 중심으로 관리할 때 사용)
from langgraph.graph import END, StateGraph, MessagesState

# - ToolNode: 미리 정의된 도구 실행 노드 (LangGraph 내에서 툴 호출을 쉽게 처리)
#   예: 상태 그래프 내에서 LLM이 특정 Tool을 호출할 수 있도록 연결
from langgraph.prebuilt import ToolNode

# 에이전트가 사용할 도구 정의
def recommand_recipe(dish: str):
  """
  주어진 요리에 대해 간단한 레시피 제공
  Args:
    dish: 요리이름
  Returs:
   str: 레시피 설명
  """
  recipes = {
    "파스타": "재료: 스파게티 면, 토마토 소스, 올리브 오일, 마늘. 면을 삶고 소스를 부어주세요.",
    "불고기": "재료: 소고기, 간장, 설탕, 마늘. 고기를 양념에 재워 볶아주세요.",
    "샐러드": "재료: 양상추, 토마토, 오이, 드레싱. 채소를 썰어 드레싱과 버무려주세요."
  }
  return recipes.get(dish, "죄송하지만 해당 요리의 레시피를 찾을 수 없습니다.")

# 도구를 구성하여 저장
tools = [recommand_recipe]

# ToolNode 생성
tool_node = ToolNode(tools)

# 워크 플로 그래프 생성 및 상태 관리
model = ChatOpenAI(model='gpt-4o-mini', api_key=OPENAI_API_KEY, temperature=0).bind_tools(tools)

# 검색을 계속 지속할지 여부를 결정
def should_continue(state: MessagesState):
  message = state['messages']
  last_message = message[-1]

  # tool calling: llm이 도구를 호출하면 tools 노드로 라우팅
  if last_message.tool_calls:
    return 'tools'
  return END

# 모델을 호출하는 함수
def call_model(state: MessagesState):
  messages = state['messages']
  response = model.invoke(messages)

  # 결과는 기존 목록에 누적해 추가되기 때문에 딕셔너리를 반환
  return {"messages": [response]}

# 워크플로우 생성: 그래프 정의
workflow = StateGraph(MessagesState)

# 검색 사이클을 순환할 두개 노드 정의
workflow.add_node("agent", call_model)
workflow.add_node('tools', tool_node)

# 첫 진입점을 agent로 설정: 첫 번째로 호출되는 노드
workflow.set_entry_point('agent')

# 정보 검색 조건무 경로 추가
workflow.add_conditional_edges(
  'agent', # 첫 진입 노드
  should_continue # 다음으로 어느 노드가 호출될지 결정
)

# tools에서 agent로의 일반 경로를 추가한다.
# 이는 tools가 호출된 후 agent 노드가 다음에 호출된다는 것을 의미
workflow.add_edge('tools', 'agent')

# 그래프 실행 간 상태를 유지하기 위해 메모리 체크포인터를 설정하고 워크플로를 컴파일한다.
# 그래프 실행 간 상태를 유지하기 위해 메모리 초기화
checkpointer = MemorySaver()

# 워크플로우에 체크포인터를 전달하여 컴파일한다.
app = workflow.compile(checkpointer=checkpointer)

# 에이전트와 상호작용하여 레시피를 요청하고 결과를 확인
# 첫 번째 질문: 파스타 레시피 요청
print('==== 첫번째 질문 ===')
final_state = app.invoke(
  {"messages": [HumanMessage(content='파스타 만드는 방법을 알려주세요.')]},
  config={"configurable": {"thread_id": 100}}
)

print("AI 응답: ", final_state['messages'][-1].content)
print("\n현재 대화 기록 개수: ", len(final_state['messages']))

# 두 번째 질문: 이전 대화 내용 참조하는 질문
print('==== 두번째 질문: 대화 기억 확인 ===')
final_state = app.invoke(
  {"messages": [HumanMessage(content='방금 알려준 레시피는 어떤 요리를 만드는 거야?.')]},
  config={"configurable": {"thread_id": 100}}
)

print("AI 응답: ", final_state['messages'][-1].content)
print("\n현재 대화 기록 개수: ", len(final_state['messages']))

# 세 번째 질문: 또 다른 대화 기억 테스트
print('==== 세번째 질문: 추가 대화 기억 확인 ===')
final_state = app.invoke(
  {"messages": [HumanMessage(content='그 요리의 주재료는 뭐야?')]},
  config={"configurable": {"thread_id": 100}}
)

print("AI 응답: ", final_state['messages'][-1].content)
print("\n현재 대화 기록 개수: ", len(final_state['messages']))