import os 
from pathlib import Path
from dotenv import load_dotenv
import tempfile

from langchain_core.callbacks import BaseCallbackHandler
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# 생성기 관련 모듈
from langchain import hub # 생성기 템플릿
from langchain_core.output_parsers import StrOutputParser # 생성기 출력 파서
from langchain_core.runnables import RunnablePassthrough # 생성기 실행 파서

import streamlit as st

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 현재 파일 위치를 기준으로 data 폴더 경로 설정
current_file = Path(__file__)
data_dir = current_file.parent / 'data'
pdf_path = data_dir / 'unsu.pdf'

# Loader
loader = PyPDFLoader(str(pdf_path))
pages = loader.load_and_split()

######## streamlit ########
st.title('ChatPDF')
st.write('-----------')

# 파일 업로드
uploaded_file = st.file_uploader('PDF 파일을 올려 주세요', type='pdf')
st.write('-----------')

# 업로드 파일을 불러오는 함수를 정의한다.
# 파일을 처리하기 위해 임시파일을 생성한다.
# tempfile.TemporaryDirectory() 함수를 사용하여 임시 디렉토리를 생성한다.
# os.path.join() 함수를 사용하여 임시 디렉토리와 파일 이름을 결합하여 temp_filepath 변수에 저장한다.
# f.write() 함수를 통해 업로드한 파일의 내용을 temp_filepath에 작성한다.

# PyPDFLoader() 함수를 사용하여 파일을 전달한다.

def pdf_to_document(uploaded_file):
  temp_dir = tempfile.TemporaryDirectory()
  temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
  with open(temp_filepath, 'wb') as f:
    f.write(uploaded_file.getvalue())

  loader = PyPDFLoader(temp_filepath)
  pages = loader.load_and_split()
  return pages

# 업로드된 파일 처리
# 사용자가 파일을 업로드 했을 경우메만 서비스 코드를 실행
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)

  # splitter
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
  )

  texts = text_splitter.split_documents(pages)

  # embedding
  embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
  )

  # db
  db = Chroma.from_documents(texts, embedding_model)

  # StreamHandler
  # 사용자가 질문을 입력한 뒤 답변이 오기까지 딜레이가 발생한다.
  # 체감 딜레이 시간을 줄이기 위해 LLM 답변을 토큰 단위로 실시간 출력하는 스트리밍 방식이 필요하다.
  # LLM이 토큰을 생성할 때마다 화면에 표시한 텍스트를 처리하는 StreamHandler를 생성한 뒤
  # LLM 체인의 콜백 함수로 StreamHander를 연결한다.
  # import 키워드를 통해 BaseCallbackHander 함수를 불러온 뒤, 이를 상속 받는 StreamHandler 클래스를 생성한다.
  # 이후 LLM 체인의 콜백 함수로 StreamHandler를 연결한다.
  # __init__ 함수는 StreamHandler 클래스의 생성자로, 객체가 생성될 때 초기화 작업을 수행한다.
  # self.container는 텍스트를 담은 Streamlit 컨테이너이고, self.text는 LLM에서 받은 텍스트를 저장하는 변수로 기본값은 빈문자열이다.
  # on_llm_new_token은 LLM에서 새로운 토큰을 받을 때마다 호출되는 함수다.
  # self.text += token에서는 새로운 토큰을 누적된 텍스트에 추가한다.
  # self.container.markdown(self.text)에서는 지금까지 누적된 텍스트를 마크다운 형식으로 컨터이너에 업데이트 한다.

  class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
      self.container = container
      self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
      self.text += token
      self.container.markdown(self.text)

  # 사용자 입력
  st.header('PDF 내용을 질문해 보세요')
  question = st.text_input('질문을 입력하세요.')

  # 질문 버튼
  if st.button('질문하기'):
    # 로딩 버튼
    with st.spinner("생성중..."):
      # retriever
      llm = ChatOpenAI(temperature=0)

      retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm
      )

      # prompt
      prompt = hub.pull("rlm/rag-prompt")

      # 빈 컨터이너 하나 생성(빈 공간)
      chat_box = st.empty()
      stream_handler = StreamHandler(chat_box)

      # llm 구성
      generate_llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler]
      )

      def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
      
      rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
      )

      result = rag_chain.invoke(question)