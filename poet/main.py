from langchain.chat_models import init_chat_model # 챗모델 초기화 - llm 초기 연결 지정
from langchain_core.prompts import ChatPromptTemplate # 프롬프트 지정 클래스
from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서

from dotenv import load_dotenv
load_dotenv()

# ChatOpenAI 초기화
llm = init_chat_model("gpt-4o-mini", model_provider='openai')

# 프롬프트 탬플릿 생성
prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a helpful assistant."),
  ("user", "{input}")
])

# 문자열 출력 파서
output_parsers = StrOutputParser()

# llm 체인 구성
chain = prompt | llm | output_parsers

# invoke 체인 실행
# content = '비오는 날'
# result = chain.invoke({"input": content + '에 대한 시를 써줘'})
# print(result)

# streamlit 참조: https://blog.zarathu.com/posts/2023-02-01-streamlit/
import streamlit as st

st.title('인공지능 시인')

# 시 주제 입력 필드
content = st.text_input('시 주제를 제시해 주세요.')
st.write('시 주제는 ', content)

# 작성 요청
if st.button('작성 요청'):
  with st.spinner('생성중...'):
    result = chain.invoke({"input": content + "에 대한 시를 써줘"})
    st.write(result)
