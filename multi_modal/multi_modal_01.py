# 데이터 추출
# PDF 파일에서 텍스트, 이미지, 테이블 추출

import os 
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf # PDF에서 텍스트, 이미지, 테이블 추출 함수

# 현재 디렉토리 저장
current_directory = os.path.dirname(os.path.abspath(__file__))

# print(current_directory)

# PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출하는 함수
# 파일의 경로와 이름을 입력하면 PDF의 내용을 블록 단위로 분할하며 이미지를 추출하고 테이블 구조를 분석
# 최대 4000자로 분할되어 저장
def extract_pdf_elements(path, fname):
  """
  Args:
    path: 파일의 경로
    fname: 파일의 이름
  Returns:
    []: PDF파일에서 추출된 이미지, 테이블, 텍스트 블록 리스트
  """
  return partition_pdf(
    filename=os.path.join(path, fname),
    extract_images_in_pdf=False,  # poppler 의존성 문제로 이미지 추출 비활성화
    infer_table_structure=True, # 테이블 구조를 추론
    chunking_strategy="by_title", # 타이틀을 기준으로 텍스트를 블록으로 분한
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
    tables: 테이블 요소들 리스트
    texts: 텍스트 요소들 리스트
  """

  tables = []
  texts = []

  for element in raw_pdf_elements:
    if "unstructured.docuemnts.elements.Table" in str(type(element)): # 테이블 스트링타입 체크
      tables.append(str(element))
    elif "unstructured.docuemnts.elements.CompositeElement" in str(type(element)): # 텍스트 스트링타입 체크
      texts.append(str(element))
  return tables, texts

# 파일 경로 생성
fname = 'invest.pdf'
fpath = os.path.join(os.path.dirname(current_directory),'multi_modal', 'data')

# 데이터 분할 -> 텍스트와 테이블 분류 -> 텍스트 Split -> 텍스트 결합 및 분할 -> 결과 확인 -> LLM을 이용한 질답 생성

# 데이터 분할
# PDF 파일에서 텍스트와 테이블을 추출한 후, 텍스트를 일정한 크기로 분할하여 데이터 처리를 진행하는 과정
# 특히 텍스트 데이터를 효율적으로 처리하기 위해 텍스트를 2000자 단위로 분할하는 작업 수행
print("PDF 처리 시작 ...")
raw_pdf_elements = extract_pdf_elements(fpath, fname)

# 텍스트와 테이블 분류
texts, tables = categorize_elements(raw_pdf_elements)

# 텍스트 분할 설정
# CharacterTextSplitter.from_tiktoken_encoder() 함수는 텍스트 데이터를 일정 크기 단위로 나누는 분할기를 설정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=2000,
  chunk_overlap=200
)

# 텍스트 결합 및 분할
# 텍스트 리스트에 포함된 모든 텍스트를 하나의 긴 문자열로 결합
# 그 후 결합된 텍스트를 2000자 단위로 분할하여 texts_2k_token 리스트에 저장
joined_texts = "\n".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)

# 분할 결과 확인
print('분할된 텍스트 개수: ', len(texts_2k_token))
print('원본 텍스트 요소 개수: ', len(texts))
print('테이블 요소 개수: ', len(tables))