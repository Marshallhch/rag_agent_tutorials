# LCEL(Language Chain Execution Language)은 랭체인에서 제공하는 간결하고 선언적인 체인 구성 언어
# 기존에는 LLM, 프롬프트, 문서 검색기, 출력 파서 등을 연결하려면 클래스를 일일이 생상하고 매서드를 호출해야 했다.
# LCEL을 사용하면 연산자 기반의 직관적인 방식으로 LLM 워크플로를 구성할 수 있다.

# 또한 RunnableSequence나 RunnableParallel 같은 체인을 한 줄 선언으로 작성하여 데이터 흐름을 직관적으로 드러낸다.
# RunnablePassthough.assign()을 사용하면 앞 단계 출력을 새 키로 추가하여 다음 단계로 전달할 수 있다.

# 리뷰를 처리하는 단계별 작업을 정의하기 위해 여섯 가지 프롬프트 템플릿을 구성
# 1. 리뷰번역(prompt1): 영어로 작성된 리뷰를 한글로 번역
# 2. 리뷰요약(prompt2): 번역된 리뷰를 한 문장으로 요약
# 3. 긍/부정 점수 평가(prompt3): 번역된 리뷰를 바탕으로 1 ~ 10점 사이의 점수 평가
# 4. 언어감지(prompt4): 원래 리뷰가 작성된 언어를 감지
# 5. 공손한 답변 생성(prompt5): 요약된 리뷰와 감지된 언어 정보를 기반으로 공손한 답변을 생성
# 6. 답변 번역(prompt6): 생성된 답변을 한국어로 번역

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# 전달 받은 키-값 쌍의 데이터를 추가해 다음 단계로 전달하는 역할
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai = ChatOpenAI(model='gpt-4o-mini', api_key=OPENAI_API_KEY, temperature=0.7)

# 프롬프트 탬플릿 정의
# 1. 리뷰 번역
prompt1 = PromptTemplate(
  input_variables=['review'],
  template="다음 숙박 시설 리뷰를 한글로 번역해 주세요. \n\n{review}"
)

# 2. 리뷰 요약
prompt2 = PromptTemplate(
  template="다음 숙박 시설 리뷰를 한 문장으로 요약해 주세요. \n\n{translation}"
)

# 3. 점수 평가
prompt3 = PromptTemplate(
  template="다음 숙박 시설 리뷰를 읽고 1점부터 10점 사이에서 부정/긍정 점수를 평가해 주세요. 숫자로만 답해주세요. \n\n{translation}"
)

# 4. 언어 식별
prompt4 = PromptTemplate(
  template="다음 숙박 시설 리뷰에 사용된 언어는 무엇인가요? 언어 이름만 답하세요. \n\n{review}"
)

# 5. 답변 생성
prompt5 = PromptTemplate(
  template="다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성해 주세요. \n답변 언어: {language}\n리뷰 요약: {summary}"
)

# 6. 한국어로 번역 답변
prompt6 = PromptTemplate(
  template="다음 생성된 답변을 한국어로 번역해 주세요. \n\n{reply}"
)

# LCEL을 사용한 체인 구성
# 각 단계는 프롬프트 | LLM | 출력 파서

# 1단계: 리뷰 번역 제안
translation_chain = prompt1 | openai | StrOutputParser()

# 2단계: 리뷰 요약
summarize_chain = prompt2 | openai | StrOutputParser()

# 3단계: 점수 평가
sentiment_score_chain = prompt3 | openai | StrOutputParser()

# 4단계: 언어 식별
language_chain = prompt4 | openai | StrOutputParser()

# 5단계: 답변 생성
reply1_chain = prompt5 | openai | StrOutputParser()

# 6단계: 답변 생성(한국어)
reply2_chain = prompt6 | openai | StrOutputParser()

# 위 6개의 체인을 combined_lcel_chain으로 연결하여 하나의 전체적인 작업흐름을 구성한다.
# 여기서는 RunnablePassthrough.assign()을 사용해 리뷰 분석과 응답 생성 과정을 여러 단계로 나눠 연결한다.

# 첫번째 assign
# - 입력값('review': 원본 리뷰 텍스트)
# - translate_chain이 review 값을 받아 한극 번역 결과(translation)을 생성한다.
# - 결과: 딕셔너리에 'translation':'번역된 텍스트' 형식으로 추가된다.

# 두번째 assign
# - 입력값: 딕셔너리 내 기존 값과 translation
# - summarize_chain와 sentiment_score_chain는 번역 결과(translation)를 사용해 각각 요약(summary)과 감성점수(sentiment_score)를 생성
# - language_chain은 원본리뷰를 사용해 언어를 판별(language)한다.
# - 결과: 딕셔너리에 'summary', 'sentiment_score', 'language'가 추가된다.

# 세번째 assign
# - 입력값: 이전까지 누적된 값들(특히 'language', 'summary')
# - reply1_chain이 감지된 언어와 요약을 이용해 첫 번째(원본 언어) 답변(reply)을 생성
# - 결과: 딕셔너리에 reply이 추가

# 네번째 assign
# - 입력값: 딕셔너리 내 'reply'
# - reply2_chain이 첫번째 답변(reply)을 한국어로 번역한 두 번째 답변(reply2)를 생성
# - 결과: 딕셔너리에 'reply2'가 추가

# RunnablePassthrough.assign()을 사용해 각 단계의 출력을 다음 단계의 입력으로 전달하고,
# 중간 결과들을 딕셔너리에 누적한다.

combined_lcel_chain = (
  RunnablePassthrough.assign(
    # 입력: {'review': '원본 리뷰 텍스트'}
    # translate_chain은 'review' 키를 사용하여 호출
    # 출력: {'review': '...', 'translation': '번역된 텍스트'}
    translation=lambda x: translation_chain.invoke({'review': x['review']})
  )
  | RunnablePassthrough.assign(
    # 입력: {'review': '...', 'translation': '번역된 텍스트'}
    # summarize_chain과 sentiment_score_chain은 translation 키를 사용
    # language_chain은 원본 review 키를 사용
    # 출력: {'review': '...', 'sentiment_score': '감성 점수', 'language': '언어'}
    summary = lambda x: summarize_chain.invoke({'translation': x['translation']}),
    sentiment_score = lambda x: sentiment_score_chain.invoke({'translation': x['translation']}),
    language = lambda x: language_chain.invoke({'review': x['review']}) 
  )
  | RunnablePassthrough.assign(
    # 입력: {'review': '...', ..., 'language': '언어', 'summary': '요약'}
    # reply1_chain은 language와 summary 키를 사용
    # 출력: {'review': '...', ..., 'language': '...', 'summary': '...', 'reply': '첫번째 답변'}
    reply1 = lambda x: reply1_chain.invoke({'language': x['language'], 'summary': x['summary']})
  )
  | RunnablePassthrough.assign(
    # 입력: {'review': '...', ..., 'reply': '첫번째 답변'}
    # reply2_chain은 reply2 키를 사용
    # 출력: {'review': '...', ..., 'reply1': '...', 'reply2': '두번째 답변(한국어)'}
    reply2 = lambda x: reply2_chain.invoke({'reply': x['reply1']})
  )
)

# 위 코드에서 translation = lambda x: translation_chain.invoke({'review': x['review']})은 현재 컨텍스트 딕셔너리 x에서 review 값을 꺼내 번역 체인을 실행하고, 얻은 결과를 translation 키에 돌려주는 한 줄짜리 콜백이다.
# 이 람다를 RunnablePassthrough.assign()에 전달하면 람다가 실행되어 번역 결과를 기존 컨텍스트에 합쳐 새 딕셔너리를 만들어준다.
# 따라서 각 단계가 '필요한 입력 추출 -> 하위 체인 호출 -> 새로운 키 자동추가' 과정을 반복하게 된다.
# 결과적으로 중간 데이터를 직접 주고받을 코드가 사라져 체인 구조가 간결해지고 의존성이 명확해진다.

# 숙박시설 리뷰를 입력하고, 체인을 실행하여 결과를 출력
# 숙박시설 리뷰 입력
review_text = """
  The hotel was clean and the staff were very helpful. The location was convenient, close to many attractions.
  Hoever, the room was a bit small and the breakfast options were limited. 
  Overall, a decent stay but there is room for improvement.
"""

# 체인 실행 및 결과 출력
try:
  # invoke() 메서드에 초기 입력을 딕셔너리 형태로 전달
  result = combined_lcel_chain.invoke({'review': review_text})

  # 결과 딕셔너리에서 각 키를 사용해 값을 출력
  print('translation: ', result.get('translation'))
  print('summary: ', result.get('summary'))
  print('sentiment_score: ', result.get('sentiment_score'))
  print('language: ', result.get('language'))
  print('reply1: ', result.get('reply1'))
  print('reply2: ', result.get('reply2'))
except Exception as e:
  print('error: ', e)