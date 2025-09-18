# 어떤 리뷰를 요청. 그 요청 결과를 출력
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

prompt_template = "이 음식 리뷰 '{review}'에 대해 '{rating1}'점부터 '{rating2}'점까지의 점수를 평가해 주고, 이유를 기술해 주세요."

prompt = PromptTemplate(
  input_variables=['review', 'rating1', 'rating2'],
  template=prompt_template
)

openai = ChatOpenAI(
  temperature=0.7,
  model="gpt-4o-mini",
  api_key=os.getenv("OPENAI_API_KEY")
)

# 체인 구성
chain = prompt | openai | StrOutputParser()

# 사용자 리뷰와 점수 범위를 입력하여 모델에게 평가 요청
try:
  response = chain.invoke({
    'review': '맛은 있었지만 배달 포장이 부족하여 아쉬웠습니다.',
    'rating1': '1',
    'rating2': '5'
  })

  print('평가 결과: ', response)
except Exception as e:
  print('error: ', e)