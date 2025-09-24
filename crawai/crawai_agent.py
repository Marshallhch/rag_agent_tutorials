# 인공지능 에이전트의 특성
# - AI 에이전트는 사용자나 다른 프로그램을 대신하여 자율적으로 또는 준자율적으로 작업을 수행하도록 설계된 소프트웨어다
# - 이러한 에이전트들은 인공지능을 활용하여 결정을 내리고 조치를 취하며, 다른 시스템과 상호작용한다.

# 에이전트의 특성
# - 자율성: 지속적인 인간의 개입 없이 독립적으로 작업을 수행한다.
# - 의사결정: 알고리즘과 AI 모델을 사용하여 최선의 조치를 선택한다.
# - 학습: 머신 러닝을 통해 과거 경험에서 배우고 새로운 상황에 적응한다.
# - 상호작용: 다른 에이전트나 시스템과 소통하고 협력한다.
# - 전문화: 특정 작업이나 도메인에 특화될 수 있다
# - 목표지향성: 구체적인 목표를 달성하기 위해 노력한다.

# CrewAI
# - 역할 수행, 자율적인 AI 에이전트를 조율하기 위한 프레임워크로, 에이전트들이 협업하여 복잡한 작업을 원활하게 수행할 수 있도록 지원한다.
# - 각 에이전트는 특정 역할과 목표를 가지고 있으며, 함께 협력하여 공통의 목표를 달성한다.

# - 에이전트(Agent): 특정 작업을 수행하도록 프로그래밍된 독립적인 단위다
# - 작업(Tesk): 에이전트가 수행해야 할 구체적인 할당 또는 작업이다
# - 크루(Crew): 공통의 목표를 향해 함께 작업하는 에이전트들의 팀

# 필요 모듈
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# OpenAI Key
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=OPENAI_API_KEY)

# =============== 에이전트 정의 ======================
# Agent Parameters
# role: 역할 설정
# goal: 역할에 따른 목표 설정
# backstory: 역할이 해야할 배경(할일)
# llm: 언어 모델
# allow_delegation: 다른 에이전트에게 현재 작업을 위임할지 여부(True, False)
# verbose: 로그 출력 여부

# 1. 콘텐츠 기획자 에이전트
planner = Agent(
  role="콘텐츠 기획자",
  goal="{topic}에 대한 흥미롭고 사실적인 콘텐츠를 기획합니다.",
  backstory=(
    "당신은 {topic}에 대한 블로그 기사를 기획하고 있습니다."
    "청중이 무언가를 배우고 정보에 입각한 결정을 내릴 수 있도록 도와주는 정보를 수집합니다."
    "블로그 게시물의 일부가 되어야 하는 자세한 개요와 관련된 주제 및 하위 주제를 준비해야 합니다."
    "당신의 작업은 이 주제에 대한 기사를 작성하는 콘텐츠 작가의 기초가 됩니다."
  ),
  llm=llm,
  allow_delegation=False,
  verbose=True
)

# 2. 콘텐츠 작가 에이전트
writer = Agent(
  role="콘텐츠 작가",
  goal="주제: {topic}에 대한 통찰력 있고 사실적인 의견 기사를 작성합니다",
  backstory=(
    "당신은 {topic}에 대한 새로운 의견 기사를 작성하고 있습니다."
    "당신의 글은 콘텐츠 기획자의 작업을 기반으로 하며, 콘텐츠 기획자는 개요와 관련된 맥락을 제공합니다."
    "콘텐츠 기획자가 제공한 개요의 주요 목표와 방향을 따릅니다."
    "또한 콘텐츠 기획자가 제공한 정보로 뒷받침되는 객관적이고 공정한 통찰력을 제공합니다."
    "의견 진술과 객관적 진술을 구분하여 의견 기사에 반영합니다."
  ),
  allow_delegation=False,
  llm=llm,
  verbose=True
)

# 3. 편집자 에이전트
editor = Agent(
  role="편집자",
  goal="주어진 블로그 게시물을 블로그 글쓰기 스타일에 맞게 편집합니다.",
  backstory=(
    "당신은 콘텐츠 작가로부터 블로그 게시물을 받는 편집자입니다."
    "당신의 목표는 블로그 게시물이 저널리즘의 모범 사례를 따르고,"
    "의견이나 주장 시 균형 잡힌 관점을 제공하며,"
    "가능하다면 주요 논란이 되는 주제나 의견을 피하도록 검토하는 것입니다."
  ),
  llm=llm,
  allow_delegation=False,
  verbose=True
)

# 4. 번역가 에이전트
translator_kor = Agent(
  role="translator",
  goal="Translate to korean",
  verbose=True,
  memory=True,
  backstory=(
      "언어를 감지해서 한국어로 바꾸어서 작성해줘"
  ),
  allow_delegation=False,
  llm=llm
)

# =============== 태스크 정의 ========================
# Task Parameters:
# description: 작업자가 수행해야 할 작업을 상세히 기술
# expected_output: 기대되는 출력물 명시
# agent: 이 작업을 수행할 에이전트 연결 지정

# 1. 기획 작업
plan = Task(
  description=(
    "1. {topic}에 대한 최신 동향, 주요 인물, 주목할 만한 뉴스를 우선순위에 둡니다.\n"
    "2. 대상 청중을 식별하고 그들의 관심사와 어려움을 고려합니다.\n"
    "3. 소개, 주요 포인트, 행동 촉구를 포함한 자세한 콘텐츠 개요를 개발합니다.\n"
    "4. SEO 키워드와 관련 데이터 또는 소스를 포함합니다."
  ),  # 작업에 대한 상세 설명
  expected_output=(
    "개요, 청중 분석, SEO 키워드, 리소스를 포함한 포괄적인 콘텐츠 계획 문서."
  ),  # 기대 출력물
  agent=planner,  # 이 작업을 수행할 에이전트 지정
)

# 2. 작성 작업
write = Task(
  description=(
    "1. 콘텐츠 계획을 사용하여 {topic}에 대한 매력적인 블로그 게시물을 작성합니다.\n"
    "2. SEO 키워드를 자연스럽게 통합합니다.\n"
    "3. 섹션/부제목은 매력적인 방식으로 적절하게 명명됩니다.\n"
    "4. 매력적인 소개, 통찰력 있는 본문, 요약 결론으로 구조화되었는지 확인합니다.\n"
    "5. 문법 오류와 브랜드의 음성에 맞게 교정합니다.\n"
  ),
  expected_output=(
    "마크다운 형식의 잘 작성된 블로그 게시물로, 각 섹션은 2~3개의 단락으로 구성되어 있으며, 출판 준비가 되어 있습니다."
  ),
  agent=writer,
)

# 3. 편집 작업
edit = Task(
  description=(
    "주어진 블로그 게시물을 문법 오류와 브랜드의 음성에 맞게 교정합니다."
  ),
  expected_output=(
    "마크다운 형식의 잘 작성된 블로그 게시물로, 각 섹션은 2~3개의 단락으로 구성되어 있으며, 출판 준비가 되어 있습니다."
  ),
  agent=editor
)

# 4. 번역 작업
translate = Task(
  description=(
      "주제에 대해 연구원이 작성해준 보고서를 기반으로 2단락 5000자로 요약하여서 한국어 콘텐츠를 작성합니다."),
  expected_output="주제에 대해 연구원이 작성해준 보고서를 기반으로 한국어 콘텐츠를 작성합니다.",
  agent=translator_kor,
  async_execution=False, # 비동기 실행 여부 - 비동기 실행을 사용하면 작업이 완료되기 전에 다음 작업을 시작할 수 있다.
  output_file= "translated-blog.md"
)

# =============== 크루 정의 ==========================
crew = Crew(
  agents = [planner, writer, editor, translator_kor],
  tasks = [plan, write, edit, translate],
  verbose = True
)

# 입력
inputs = {'topic': '인기 프로그래밍 언어 순위 소개'}

# 작업 실행
result = crew.kickoff(inputs=inputs)

# 결과 출력
print(result)