import os
import base64
from langchain_core.messages import HumanMessage # 사용자 메시지 관리 모듈
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 현재 디렉토리 경로
current_directory = os.path.dirname(os.path.abspath(__file__))

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

raw_pdf_elements = extract_pdf_elements(fpath, fname)

# 텍스트와 테이블 분류
texts, tables = categorize_elements(raw_pdf_elements)

# 텍스트 분할 설정
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
  chunk_size=2000,
  chunk_overlap=200
)

joined_texts = "\n".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)

# 텍스트 및 테이블 요약 함수
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

# 이미지 인코딩
def encode_image(image_path):
  with open(image_path, 'rb') as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# OpenAI를 이용해 이미지 요약
def image_summarize(img_base64, prompt):
  chat = ChatOpenAI(model='gpt-4o-mini', max_tokens=1024)

  msg = chat.invoke(
    [
    HumanMessage(
      content=[ 
          {"type": "text", "text": prompt},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64, {img_base64}"}}
        ]
      )
    ]
  )
  return msg.content

# 주어진 경로 내의 이미지 파일들을 base64로 인코딩 후 각 이미지를 요약하여 리스트로 반환
# .jpg, .png, .jpeg 파일만 처리
def generate_img_summaries(path):
  """
  Args: 
    path: Unstructured에 의해 추출된 .jpg 파일의 경로
  Return:
    image_summaries: 이미지 요약 리스트
    img_base64_list: base64로 인코딩된 이미지 리스
  """

  # base64로 인코딩된 이미지 저장 초기화
  img_base64_list = []
  # 이미지 요약 저장 리스트 초기화
  image_summaries = []

  # Prompt_kor 한국어
  prompt_kor = """You are an assistant tasked with summarizing images for retrieval. 
  These summaries will be embedded and used to retrieve the raw image. Provide a concise summary of the image that is well optimized for retrieval. 
  The summary should be written in Korean (Hangul)."""

  # Prompt 영어
  prompt = """You are an assistant tasked with summarizing images for retrieval. 
  These summaries will be embedded and used to retrieve the raw image. Provide a concise summary of the image that is well optimized for retrieval. """

  # 주어진 경로에서 파일 목록을 가져와 정렬한 후, 각 파일을 처리
  for img_file in sorted(os.listdir(path)): # listdir: 디렉토리 내의 파일과 폴더 목록을 리스트로 반환
    # 세 가지 확장자 확인
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
      # 파일의 전체 경로 생성
      img_path = os.path.join(path, img_file)
      # 이미지를 base64로 인코딩하여 문자열로 반환
      base64_image = encode_image(img_path)
      # 생성된 문자열을 리스트에 추가
      img_base64_list.append(base64_image)

      # 한국어로 요약된 결과를 사용
      image_summaries.append(image_summarize(base64_image, prompt_kor))
  return img_base64_list, image_summaries

# figure 디렉토리 경로 설정
figures_directory = os.path.join(current_directory, 'figures')

# 이미지 요약 생성
img_base64_list, image_summaries = generate_img_summaries(figures_directory)

# 다중 벡터 검색기 생성
# 텍스트, 표, 이미지 요약본을 색인화하고 검색 시 원본 이미지를 반환하는 다중 벡터 검색기를 생성
# 이 검색기는 다양한 데이터 유형을 통합적으로 처리할 수 있는 기능을 제공하여 멀티 모달 검색을 가능하게 함

# 다중 벡터 검색을 위한 모듈
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever # 텍스트, 표, 이미지 요약본을 색인화 하고 검색 시 원본이미지 반환하는 모듈
from langchain_chroma.vectorstores import Chroma
from langchain.storage import InMemoryStore # 메모리 기반 저장소 사용
from langchain_core.documents import Document # 문서 데이터 표현 모듈
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings # 각각 클립 별로 텍스트와 이미지를 임베딩

# 요약본 색인화하고 원본 데이터 반환
def create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts_2k_token,
    table_summaries,
    tables,
    image_summaries,
    images
):
  
  # 저장소 초기화(임시 저장소)
  store = InMemoryStore()
  id_key = 'doc_id'

  # 다중 벡터 검색기 생성: 텍스트, 표, 이미지 요약본 색인화
  retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key
  )

  # 벡터 저장소와 문서 저장소에 문서를 추가하는 헬퍼 함수
  # 요약본을 벡터 저장소와 문서 저장소에 추가한다. 각 문서는 고유한 doc_id를 부여받고, 요약본과 원본 데이터가 함께 저장된다.
  def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
      Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)
    ]

    retriever.vetorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents))) # mset: 여러개의 키-값 쌍을 한 번에 설정하는 메서드

  # 텍스트, 테이블, 이미지 별 구분
  if text_summaries:
    add_documents(retriever, text_summaries, texts)
  if table_summaries:
    add_documents(retriever, table_summaries, tables)
  if image_summaries:
    add_documents(retriever, image_summaries, images)

  return retriever

# 임베딩 모델을 설정하고 벡터 저장소를 초기화한다.
# OpenCLIPEmbeddings를 사용하여 이미지와 텍스트 데이터를 임베딩한다.
# Chroma 벡터 저장소는 이러한 임베딩을 저장하고 검색하는 데 사용된다.
embedding = OpenCLIPEmbeddings()

# 벡터 저장소 생성
vectorstore = Chroma(
  collection_name='mm_rag_finace', # 벡터 저장소 이름
  embedding_function=embedding
)

# 다중 벡터 검색기 생성
retriever_multi_vector_img = create_multi_vector_retriever(
  vectorstore,
  text_summaries,
  texts,
  table_summaries,
  tables,
  image_summaries,
  img_base64_list
)

# ==================================================== #
# 이 스크립트는 다중 모달 RAG(Retrieval-Augmented Generation) 체인을 생성하여 텍스트, 표, 이미지 데이터를 처리하고, 한국어로 투자 조언을 제공한다.

# 모듈 및 함수
# 임포트

# io, re, IPython.display, langchain_core.runnables, langchain.prompts, PIL.Image: 다양한 모듈을 사용하여 이미지 처리, 데이터 처리, 디스플레이, 체인 생성 등을 수행.
# 함수 정의

# plt_img_base64(img_base64): Base64 인코딩된 문자열을 이미지로 표시합니다.
# looks_like_base64(sb): 문자열이 Base64 형식인지 확인합니다.
# is_image_data(b64data): Base64 데이터가 이미지인지 확인합니다.
# resize_base64_image(base64_string, size=(128, 128)): Base64 형식의 이미지를 리사이즈합니다.
# split_image_text_types(docs): Base64로 인코딩된 이미지와 텍스트를 분리합니다.
# img_prompt_func(data_dict): 주어진 데이터 딕셔너리로부터 메시지를 생성합니다.
# multi_modal_rag_chain(retriever): 다중 모달 RAG 체인을 생성합니다.
# korean_convert_rag(): 영어 텍스트를 한국어로 변환하는 RAG 체인을 생성합니다.
# 코드 실행

# 임포트 및 초기 설정

# 필요한 모듈 및 함수를 임포트.
# 이미지 처리 및 변환

# Base64 인코딩된 이미지를 처리하고 표시.
# 문자열이 Base64 형식인지, Base64 데이터가 이미지인지 확인.
# 이미지를 리사이즈.
# Base64로 인코딩된 이미지와 텍스트를 분리.
# 메시지 생성

# 주어진 데이터 딕셔너리로부터 메시지를 생성.
# 투자 분석을 위한 프롬프트 메시지를 구성.
# RAG 체인 생성

# 다중 모달 RAG 체인을 생성.
# 영어 텍스트를 한국어로 변환하는 RAG 체인을 생성.
# 다중 모달 RAG 체인과 한국어 변환 RAG 체인을 결합.

# ================================================ #