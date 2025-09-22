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
    texts,
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
  texts_2k_token,
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

# 입출력 작업을 지원하는 모듈. 파일이나 메모리에서 데이터를 읽고 쓰는 작업을 처리
# 특히 BytesIO는 메모리 상에서 바이트 기반 데이터를 읽고 쓰는 데 사용되며 이미지 데이터를 처리할 때 매우 유용하다.
import io
import re
from IPython.display import HTML, display
# RunnableLambda: 함수를 실행 가능한 객체로 변환해서 다음 체인에 추가할 수 있도록 함
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image

# 이미지를 처리하기 위한 함수 정의
# 이미지가 Base64 형식으로 인코딩 되어 있는지 확인하고, 필요에 따라 이미지를 리사이즈한 후 다시 Base64로 반환
# 이 함수는 Base64로 인코딩된 이미지 문자열을 HTML 태그로 변환하여 시각적으로 표시
def plt_img_base64(img_base64):
  image_html = f'<img src="data:image/jpeg;base64, {img_base64}" />'

  # HTML 랜더링하여 이미지 표시
  display(HTML(image_html))

# 입력된 문자열이 base64인지 확인
def looks_like_base64(sb):
  return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

# 입력된 base64 문자열이 이미지 데이터가 맞는지 확인
def is_image_data(b64data):
  image_signatures = {
    b"\xff\xd8\xff": "jpg",
    b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
    b"\x47\x49\x46\x38": "gif",
    b"\x52\x49\x46\x46": "webp",
  }

  try:
    header = base64.b64decode(b64data)[:8] # 처음 8바이트까지 헤더로 저장
    for sig in image_signatures.items():
      if header.startswith(sig):
        return True
      return False
  except Exception:
    return False

# 이미지를 주어진 크기로 리사이즈하고, 다시 Base64로 변환하여 반환
# 이미지 크기를 줄여 저장 공간을 줄이거나 네트워크 전송 시간을 단축하는 데 유용
def resize_base64_image(base64_string, size=(128, 128)):
  img_data = base64.b64decode(base64_string)
  img = Image.open(io.BytesIO(img_data))

  # 이미지 크기 조정
  resized_img = img.resize(size, Image.LANCZOS) # LANCZOS - 이미지 크기 조정 시 사용되는 알고리즘

  # 크기 조정된 이미지를 bytes 버퍼에 저장
  buffered = io.BytesIO()
  resized_img.save(buffered, format=img.format)

  # 크기 조정된 이미지를 base64로 인코딩
  return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 주어진 docs에서 base64로 인코딩된 이미지와 텍스트 데이터를 분리하여 처리
# 이미지 데이터는 크기를 조정한 후 리스트에 저장하고 텍스트 데이터는 그대로 텍스트 리스트에 저장
# 이를 통해 이미지와 텍스트 데이터를 따로 다룰 수 있다.
def split_image_text_types(docs):
  b64_images = []
  texts = []
  for doc in docs:
    # 문서 유형이 Document일 경우 page_content 추출
    if isinstance(doc, Document):
      doc = doc.page_content
    if looks_like_base64(doc) and is_image_data(doc): # 입력된 문자열이 base64이고, 이 문자열이 이미지 데이터일 때
      doc = resize_base64_image(doc, size=(1300, 600))
      b64_images.append(doc)
  return {"images": b64_images, "texts": texts}

# 투자 조언을 제공하는 멀티 모달 분석을 위한 프롬프트 메시지를 생성
# 이미지와 텍스트 데이터를 포함한 data_dict를 받아 이를 바탕으로 프롬프트를 구성하여 투자 분석을 요청하는 메시지를 반환
# 여기서는 이미지와 텍스트가 함께 제공될 수 있으며 해당 데이터를 프롬프트로 변환하여 LLM이 처리할 수 있도록 한다.
def img_prompt_func(data_dict):
  # 주어진 맥락을 하나의 문자열로 결합
  formatted_texts = "\n".join(data_dict["context"]["texts"])
  messages = []

  # 이미지가 포함된 경우 메시지에 추가
  if data_dict["context"]["texts"]:
    for image in data_dict["context"]["texts"]:
      image_message = {
        "type": "image_url",
        "image_url": {"url": f'data:image/jpeg;base64, {image}'}
      }
      messages.append(image_message)

   # 분석할 텍스트 추가
  text_message = {
    "type": "text",
    "text": (
      "You are financial analyst tasking with providing investment advice.\n"
      "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
      "Use this information to provide investment advice related to the user question. \n"
      f"User-provided question: {data_dict['question']}\n\n"
      "Text and / or tables:\n"
      f"{formatted_texts}\n\n"
      "Please provide the final answer in Korean(hangul)."
    ),
  }
  messages.append(text_message)
  return [HumanMessage(content=messages)]

# 멀티모달 RAG 체인 구성
# 텍스트와 이미지가 섞인 입력을 처리해서 질문에 답변할 수 있는 파이프라인 구성

# context: 데이터를 처리하는 핵심 요소. retriever는 검색된 데이터를 가져오며, 이 데이터는 텍스트와 이미지로 이뤄져 있을 수 있다.
# 이 데이터를 RunnableLambda(split_image_text_types) 함수를 사용하여 텍스트와 이미지로 분리한다.

# question: 사용자가 입력한 줄문을 직접 전달한다. 여기서 RunnablePassthrough() 함수를 사용하여 입력된 질문을 그대로 전달한다.

# img_prompt_func: 텍스트와 이미지를 분석하기 위한 프롬프트를 생성하는 역할을 한다.
# 텍스트와 이미지가 결합된 데이터를 LLM이 이해할 수 있도록 적절한 형식의 프롬프트로 변환한다.
# 프롬프트는 사용자의 질문, 이미지, 그리고 추가 텍스트 정보를 포함한다.

# multi_modal_rag_chain(): 텍스트, 이밎, 표 등의 데이터를 분석하기 위해 멀티 모달 RAG 체인을 생성한다.
# RAG 체인은 다양한 형식의 데이터를 기반으로 추가적인 정보 검색 및 응답 생성을 치리하는데 사용된다.
def multi_modal_rag_chain(retriever):
  # 다중 모드 LLM 생성
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)

  # RAG 파이프라인 구성
  chain = (
    {
      "context": retriever | RunnableLambda(split_image_text_types),
      "question": RunnablePassthrough()
    }
    | RunnableLambda(img_prompt_func)
    | model
    | StrOutputParser()
  )

  return chain

def korean_convert_rag():
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

  # 프롬프트 템플릿 설정
  prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to Korean."),
    ("human", "Translate the following English text to Korean: {english_text}")
  ])

  # 파이프라인 구성
  chain = (
    {"english_text": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
  )

  return chain

# RAG 종합 체인 생성
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

# 영어로 답변이 나온다면 번역
korean_convert_rag = korean_convert_rag()
final_multimodal_rag = chain_multimodal_rag | korean_convert_rag

# 조회 결과 확인 및 RAG 체인 생성
query = "주가 변동률과 가장 관련 있는 자료를 찾아주세요."
docs = retriever_multi_vector_img(query)

# 변환된 결과 개수 
print(len(docs))

print("=" * 50)

# 첫번째 문서를 이미지로 표시
plt_img_base64(docs[0])