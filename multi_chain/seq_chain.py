# SequentialChain과 LLM 체인으로 전체 흐름을 구축
# 이후 LCEL 버전으로 리팩토링

# SequentialChain
# - 여러 개의 LLM 체인 객체를 한 줄로 엮어, 앞 단계 출력을 그 다음 단계의 입력으로 자동 전달하는 오케스트레이터
# - 즉, 데이터 흐름을 명시적으로 관리해 주는 컨베이어 벨트에 해당한다.
# - 정의한 순서대로 체인을 생성하며 각 단계 결과를 키-값 딕셔너리에 저장해 이어지는 단계에 주입하므로
# - 개발자는 중간 데이터를 따로 수집하거나 전달하는 코드를 작성할 필요가 없다.

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 초기화
openai = ChatOpenAI(
  temperature=0.7,
  model="gpt-4o-mini",
  api_key=OPENAI_API_KEY
)

# 프롬프트 템플릿 정의
# 1. Prompt1: 리뷰를 한 문장으로 요약하는 작업(리뷰 요약)
# 2. Prompt2: 리뷰를 바탕으로 0 ~ 10점 사이의 점수를 평가(긍정/부정 점수 평가)
# 3. prompt3: 요약된 리뷰에 대해 공손한 답변을 작성(리뷰에 대한 공손한 답변 생성)
prompt1 = PromptTemplate.from_template("다음 식당 리뷰를 한 문장으로 요약해 주세요. \n\n{review}")

# 1번 체인 생성
chain1 = LLMChain(llm=openai, prompt=prompt1, output_key='summary')

prompt2 = PromptTemplate.from_template("다음 식당 리뷰를 읽고 0점부터 10점 사이에서 긍정/부정 점수를 평가해 주세요. 숫자로만 답해 주세요. \n\n{review}")

# 2번 체인 생성
chain2 = LLMChain(llm=openai, prompt=prompt2, output_key='sentiment_score')

prompt3 = PromptTemplate.from_template("다음 식당 리류 요약에 대해 공손한 답변을 작성해 주세요. \n리뷰요약: {summary}")

# 3번 체인 생성
chain3 = LLMChain(llm=openai, prompt=prompt3, output_key='reply')

# 세 개의 체인을 연결하여 하나의 전체적인 작업 흐름 구성
all_chain = SequentialChain(
  chains=[chain1, chain2, chain3],
  input_variables=['review'],
  output_variables=['summary', 'sentiment_score', 'reply']
)

# 리뷰 작성
review = """
  이 식당은 맛도 좋고 분위기도 좋았습니다. 가격 대비 만족 합니다.
  하지만 서비스 속도가 너무 느려서 조금 실망스럽습니다.
  종합적으로 다시 방문할 의사는 있습니다.
"""

# 체인 실행 결과 출력
try:
  result = all_chain.invoke(input={'review': review})
  print('요약 결과: ', result['summary'])
  print('감정 결과: ', result['sentiment_score'])
  print('응답 결과: ', result['reply'])
except Exception as e:
  print('error: ', e)