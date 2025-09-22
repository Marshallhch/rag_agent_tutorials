# 데이터 추출
# PDF 파일에서 텍스트, 이미지, 테이블을 추출

# 현재 디렉토리 저장
import os
current_directory = os.path.dirname(os.path.abspath(__file__))

# 필요 패키지 설치
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf # PDF 파일에서 텍스트, 이미지, 테이블을 추출하는 함수

# PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출하는 함수
# 파일의 경로와 이름을 입력하면 PDF의 내용을 블록 단위로 분할하며 이미지를 추출하고 테이블 구조를 분석
# 최대 4000자로 분할되어 저장
def extract_pdf_elements(path, fname):
  """
  Args:
    path: 파일의 경로
    fname: 파일의 이름
  Returns:
    PDF 파일에서 추출된 이미지, 테이블, 텍스트 블록들의 리스트
  """
  
  return partition_pdf(
    filename=os.path.join(path, fname),
    extract_images_in_pdf=False,  # poppler 의존성 문제로 이미지 추출 비활성화
    infer_table_structure=True, # 테이블 구조를 추론
    chunking_strategy="by_title", # 타이틀을 기준으로 텍스트를 블록으로 분할
    max_characters=4000, # 최대 4000자로 텍스트 블록을 제한
    new_after_n_chars=3800, # 3800자 이후에 새로운 블록 생성
    combine_text_under_n_chars=2000, # 2000자 이하의 텍스트는 결합
    image_output_dir_path=path, # 이미지가 저장될 경로 설정
  )

# 추출한 PDF의 요소들은 텍스트와 테이블로 분류된다. 
# categorize_elements 함수는 PDF에서 추출한 요소들을 순회하면서 테이블 요소는 tables 리스트에
# 텍스트 요소는 texts 리스트에 저장된다.
def categorize_elements(raw_pdf_elements):
  """
  PDF에서 추출한 요소들을 테이블과 텍스트로 분류한다.
  raw_pdf_elements: unstructured.documents.elements 리스트
  Args:
    raw_pdf_elements: PDF에서 추출한 요소들
  Returns:
    tables: 테이블 요소들
    texts: 텍스트 요소들
  """
  tables = []
  texts = []
  for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)): # 테이블 요소 타입 확인
      tables.append(str(element))  # 테이블 요소를 저장
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)): # 텍스트 요소 타입 확인
      texts.append(str(element))  # 텍스트 요소를 저장
  return texts, tables

# 파일 경로 설정
fname = 'invest.pdf'
fpath = os.path.join(os.path.dirname(current_directory), "multi_modal", "data")

# 경로 확인 출력
print("현재 스크립트의 위치:", current_directory)
print("pdf 위치:",fpath)

# 데이터 분할
# PDF 파일에서 텍스트와 테이블을 추출한 후, 텍스트를 일정한 크기로 분할하여 데이터 처리를 진행하는 과정
# 특히 텍스트 데이터를 효율적으로 처리하기 위해 텍스트를 2000자 단위로 분할하는 작업 수행
print("PDF 처리 시작...")
raw_pdf_elements = extract_pdf_elements(fpath, fname)

# 텍스트와 테이블 분류
texts, tables = categorize_elements(raw_pdf_elements)

# 텍스트 분할 설정
# CharacterTextSplitter.from_tiktoken_encoder() 함수는 텍스트 데이터를 일정 크기 단위로 나누는 분할기를 설정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=2000, # 2000자 단위로 분할
  chunk_overlap=200, # 200자 중복 허용
)

# 텍스트 결합 및 분할
# 텍스트 리스트에 포함된 모든 텍스트를 하나의 긴 문자열로 결합
# 그 후 결합된 텍스트를 2000자 단위로 분할하여 texts_2k_token 리스트에 저장
joined_texts = "\n".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)

# print(f"분할된 텍스트 개수: {len(texts_2k_token)}")
# print(f"원본 텍스트 요소 개수: {len(texts)}")
# print(f"테이블 요소 개수: {len(tables)}")

# 다중 벡터 검색기
# 텍스트, 표, 이미지 등의 다양한 데이터를 요약하여 인덱싱하고, 원본 데이터를 검색하는 데 사용하는 기술이다.
# 이미지와 같은 멀티모달 데이터를 포함하여 정보 검색을 효율적으로 처리할 수 있으며
# 생성된 요약본은 데이터를 빠르게 검색할 수 있도록 최적화 한다.

# 텍스트 및 표 요약
# 텍스트 및 표 요약은 GPT-4 또는 Llama 3.1 모델을 통해 생성된다.
# 특히 큰 텍스트 블록을 사용하는 경우에는 텍스트 요약이 매우 중요하다.
# 요약된 텍스트와 표 데이터는 검색에 최적화되며, 이를 통해 원본 데이터를 빠르고 정확하게 검색할 수 있다.

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# 텍스트 및 표 요약 함수
def generate_text_summaries(texts, tables, summarize_texts=False):
  """
  텍스트 및 표 데이터를 요약하여 검색에 활용할 수 있는 요약본 생성
  Args:
    texts: 텍스트 데이터
    tables: 표 데이터
    summary_texts: 텍스트 요약 여부
  Returns:
    text_summaries: 텍스트 요약 여부
    tables_summaries: 표 요약 여부
  """
  # Prompt 한국어 버전
  prompt_text_kor = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다. \n 
  이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다. \n 
  주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해 주세요. 요약할 표 또는 텍스트: {element}"""

  prompt = ChatPromptTemplate.from_template(prompt_text_kor)

  # 모델 생성
  model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
  summarize_chain = {'element': lambda x: x} | prompt | model | StrOutputParser()

  text_summaries = []
  tables_summaries = []

  # 텍스트 요약을 활성화 하는 경우
  # max_concurrency: 동시 요약 처리 수 - 병렬 처리의 최대 개수
  if texts and summarize_texts:
    text_summaries = summarize_chain.batch(texts, {'max_concurrency': 5})
  # 텍스트를 요약하지 않는 경우
  elif texts:
    text_summaries = texts

  # 테이블 요약
  if tables:
    tables_summaries = summarize_chain.batch(texts, {'max_concurrency': 5})

  return text_summaries, tables_summaries

text_summaries, table_summaries = generate_text_summaries(texts_2k_token, tables, summarize_texts=True)

print(text_summaries)
print("=" * 50)
print(table_summaries)