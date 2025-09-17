# 필요 모듈
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))
restaurant_text = os.path.join(current_dir, 'data', 'restaurants.txt')
restaurant_faiss = os.path.join(current_dir, 'restaurant-faiss')

# 생성된 인덱스가 없을 경우 생성
def create_faiss_index():
  """
  1. TextLoader를 사용한 파일 로드
  2. 텍스트를 300자로 분할 후 겹침 50 설정
  3. 임베딩 생성
  4. FAISS 인덱스 생성 및 저장
  """
  
  # 1. 
  loader = TextLoader(restaurant_text, encoding='utf-8')
  documents = loader.load()

  # 2.
  text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
  chunks = text_splitter.split_documents(documents)

  # 3. 
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

  # 4. 
  db = FAISS.from_documents(chunks, embeddings)
  db.save_local(restaurant_faiss)
  print('FAISS 인덱스 생성 완료!!')

# 생성된 인덱스가 있을 경우 사용
def load_faiss_index():
  """
  1. 임베딩 초기화
  2. 인덱스 로드(기존 생성된 파일)
  """

  # 1. 
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

  # 2.
  load_db = FAISS.load_local(restaurant_faiss, embeddings, allow_dangerous_deserialization=True)
  return load_db

# 검색된 문서들을 하나의 문자열로 합침
# FAISS를 사용하여 검색된 문서들을 하나의 문자열로 합친다.
# 이 과정은 Retrieve된 문서를 AI 모델이 처리할 수 있는 형태로 준비하는 단계다
# 검색된 문서들을 하나의 텍스트 블록으로 결합하여 모델이 이 텍스트를 기반으로 답변을 생성할 수 있도록 한다.
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

# 검색된 텍스트를 프롬프트에 포함시켜 LLM 모델에 전달, 최종 답변 생성
def answer_question(db, query):
  """
  1. LLM 모델 초기화
  2. 사용자 정의 프롬프트 작성
  3. 프롬프트와 LLM 연결 체인 구성하여 문자열 파싱
  4. 질의에 대한 답변 생성

  args:
    db: FAISS 인덱스
    query: 사용자 질문
  returns:
    result: 답변
  """

  # 1. 
  llm = OpenAI(api_key=OPENAI_API_KEY)

  # 2.
  prompt_template = """
  당신은 유능한 AI 비서입니다. 주어진 맥락 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해야 합니다.

  맥락: {context}

  질문: {question}

  답변을 작성할 때 다음 지침을 따르세요:
  1. 주어진 맥락 정보에 있는 내용만을 사용하여 답변하세요.
  2. 맥락 정보에 없는 내용은 답변에 포함하지 마세요.
  3. 질문과 관련이 없는 정보는 제외하세요.
  4. 답변은 간결하고 명확하게 작성하세요.
  5. 불확실한 경우, "주어진 정보로는 정확한 답변을 드릴 수 없습니다."라고 말하세요.

  답변:
  """

  prompt = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'question']
  )

  # 3.
  qa_chain = (
    {
      'context': db.as_retriever() | format_docs,
      'question': RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
  )

  # 4. 
  result = qa_chain.invoke(query)
  return result 

# 위 함수들을 이용하는 메인 함수
def main():
  # 인덱스가 없을 경우 생성
  if not os.path.exists(restaurant_faiss):
    create_faiss_index()

  # 인덱스가 있을 경우
  db = load_faiss_index()

  # 무한 루프 프로그램 실행
  while True:
    query = input('레스토랑에 대해 궁금한 점을 물어보세요 (종료는 "quit")')
    if query.lower() == 'quit':
      break

    answer = answer_question(db, query)
    print('답변: ', answer)

# 최종 실행
if __name__ == "__main__":
  main()