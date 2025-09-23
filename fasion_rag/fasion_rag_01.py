# 패션 관련 이미지와 텍스트를 활용해 사용자에게 스타일링 조언을 제공하는 시스템
# 이미지 데이터셋을 로드하고 ChromaDB를 통해 벡터 기반의 이미지 검색을 수행하며 멀티모달 기능이 있는 모델을 활용해
# 텍스트와 이미지를 결합한 종합적인 패션 스타일링 추천 답변 생성

import os
import base64
import chromadb
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 현재 디렉토리 경로
current_directory = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 저장 함수
def setup_dataset():
  try:
    dataset = load_dataset('detection-datasets/fashionpedia')
  except:
    # 대안으로 fasion-mnist
    dataset = load_dataset('fashion_mnist')

  # 데이터셋 저장할 폴더
  dataset_folder = os.path.join(current_directory, 'fasion_dataset')
  
  # 폴더가 없으면 생성
  os.makedirs(dataset_folder, exist_ok=True)
  return dataset, dataset_folder

# 이미지들을 로컬 폴더에 저장하는 코드
def save_images(dataset, dataset_folder, num_images=500):
  # 파라미터로 주어진 이미지 개수만큼 저장
  for i in range(num_images):
    try:
      image = dataset['train'][i]['image']
      image.save(os.path.join(dataset_folder, f'image_{i + 1}.png'))
    except Exception as e:
      print("error: ", e)
      continue
  print(f'Save {num_images} images to {dataset_folder}')

# 이미지 벡터를 저장하고, 사용자가 입력한 질문에 맞는 이미지를 검색하기 위해 ChromaDB 사용
# 데이터베이스를 설정하고 이미지를 추가
# setup_chroma_db() 함수는 ChromaDB 벡터 데이터베이스를 설정하고, 이미지를 처리할 로더 및 임베딩 함수를 초기화
# add_images_to_db() 함수는 로컬 폴더에서 PNG 파일을 읽어와 ChromaDB에 ID와 경로를 지정
def setup_chroma_db():
  # 벡터 데이터베이스 저장 경로 설정
  vdb_path = os.path.join(current_directory, 'img_vdb')
  # chroma 클라이언트 초기화
  chroma_client = chromadb.PersistentClient(path=vdb_path) # PersistentClient - 데이터베이스 파일 영구적 저장
  # 이미지 로더 및 open clip 임베딩 설정
  image_loader = ImageLoader()
  clip = OpenCLIPEmbeddingFunction()

  # 이미지 데이터베이스 생성
  image_vdb = chroma_client.get_or_create_collection(
    name="image",
    embedding_function=clip,
    data_loader=image_loader
  )

  return image_vdb

# 기존에 존재하는 이미지가 있으면 IDs를 가져옴
def get_existing_ids(image_vdb, dataset_folder):
  existing_ids = []
  # dataset_folder 내의 이미지 파일 수 계산
  num_images = len([name for name in os.listdir(dataset_folder)])
  print(f'데이터 폴더 전체 이미지 수: {num_images}')

  records = image_vdb.query(
    query_texts=[""], # 빈 문자열을 쿼리로 사용하여 모든 이미지를 검색
    n_results=num_images,
    include=['ids']
  )
  for record in records['ids']:
    existing_ids.extend(record) # 기존 ids를 추가
    print(f'{len(record)}개의 기존 IDs가 존재 합니다.')
  return existing_ids

# 이미지를 데이터베이스에 추가
def add_images_to_db(image_vdb, dataset_folder):
  existing_ids = get_existing_ids(image_vdb, dataset_folder)
  ids = []
  uris = []

  # 폴더에서 이미지를 읽어와서 데이터베이스에 추가
  for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
      img_id = str(i)
      if img_id not in existing_ids:
        file_path = os.path.join(dataset_folder, filename)
        ids.append(str(i))
        uris.append(file_path)

  if ids:
    image_vdb.add(ids=ids, uris=uris)
    print('이미지가 데이터베이스에 추가되었습니다.')
  else:
    print('추가할 새로운 이미지가 없습니다.')

# 사용자가 입력한 질문을 바탕으로 데이터베이스에서 가장 관련성 높은 이미지를 검색
# 그 결과를 출력하는 기능 구현
# query_db() 함수는 쿼리된 질문에 대해 가장 유사한 이미지 2개를 검색
# print_results() 함수는 검색된 이미지의 ID, 경로, 유사도를 출력

# 데이터베이스 쿼리를 실행하는 함수
def query_db(image_vdb, query, results=2):
  return image_vdb.query(
    query_texts=[query],
    n_results=results,
    include=['uris', 'distances']
  )

# 결과를 출력하는 함수
# def print_results(results):
#   for idx, uri in enumerate(results['uris'][0]):
#     print(f'ID: {results['ids'][0][idx]}')
#     print(f'Distances: {results['distances'][0][idx]}')
#     print(f'Path: {uri}')
#     print("\n")

# OpenAI의 모델을 사용하여 사용자의 질문을 영어로 번역하고, 다시 한국어로 변환하는 기능을 추가
# translate() 함수는 사용자가 입력한 질문을 영어로 번역하거나 영어로 답변된 내용을 한국어로 변환
# 결과를 한국어로 번역하는 함수
def translate(text, target_lang):
  translation_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
  translation_prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are a translator. Translate the following text to {target_lang}."),
    ("user", "{text}")
  ])

  # 체인 구성
  translation_chain = translation_prompt | translation_model | StrOutputParser()

  return translation_chain.invoke({'text': text})

# GPT 모델을 사용하여 이미지를 기반으로 패션 및 스타일링 조언을 제공하는 비전 체인 설정
# 사용자가 입력한 질문에 대한 답변을 생성할 때 이미지를 텍스트와 함께 분석하고
# 사용자의 요청에 맞는 패션 조언을 제공
# 시각적 정보를 처리하는 체인 설정
def setup_vision_chain():
  gpt_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
  parser = StrOutputParser()
  image_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful fashion and styling assistant. Answer the user's question using the given image context with direct references to parts of the images provided. Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure."),
    ("user", [
      {"type": "text", "text": "What are some ideas for styling {user_query}"},
      {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
      {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
    ]),
  ])

  return image_prompt | gpt_model | parser

# BAase64 인코딩
# 이미지를 LLM의 컨텍스트로 입력하려면 base64로 인코딩하는 과정이 필요하다
# 이는 텍스트 기반 시스템에서 안전하게 사용할 수 있는 64개의 ASCII 문자만을 사용하여 이진 데이터를 인코딩한다.
# 이 방식은 전송 및 저장 과정에서 데이터 손상을 방지한다.
# 모델이나 이를 호스팅하는 시스템이 이미지 데이터를 텍스트 형식으로 처리할 수 있게 되면 다음과 같은 이점이 있다.
# - 호환성 유지: 다양한 플랫폼이나 프로그래밍 언어에서 데이터를 일관되게 처리할 수 있도록 보장한다.
# - 데이터 전송 안정성: 텍스트 기반 프로토콩을 통해 이미지를 안전하게 전송할 수 있다.
# - 데이터 손상 방지: 이진 데이터를 텍스트로 변환하여 전송 및 저장 과정에서 발생할 수 있는 손상을 예방한다.

# 또한 base64 인코딩을 통해 이미지를 텍스트 문자열로 변환하면, 텍스트 전용 시스템에서도 데이터 전송 및 처리가 원활해진다.
# 추가로 LLM에 이미지를 전달하려면 JPEG, PNG, GIF, WEBP 등의 지원 형식으로 20mb 미만이 되도록 압축하고
# 가급적 1024 * 1024 이하로 리사이징한 뒤 base64로 인코딩해야 한다.
# base64는 원본보다 약 33% 길어지므로, 흔히 300KB 짜리 웹용 JPEG라도 변환 후에는 약 10만 토큰이나 차지해
# GPT-4o의 128k 토큰 한도에 거의 가깝게 된다.
# 이미지를 여러 장 넣거나 프롬프트 텍스트가 길면 곧바로 한도를 초과하므로 해상도와 품질, 개수를 꼭 조절해야 한다.
# 제한을 넘기거나 인코딩이 손상되면 400 오류가 발생한다.
def format_prompt_inputs(data, user_query):
  inputs = {}

  # 사용자 질문을 딕셔너리에 추가
  inputs['user_query'] = user_query

  # 'uris' 리스트에서 첫 두 이미지 경로 가져오기
  image_path_1 = data['uris'][0][0] # 첫 번째 이미지
  image_path_2 = data['uris'][0][1] # 두 번째 이미지

  # 첫번째 이미지 인코딩
  with open(image_path_1, 'rb') as image_file:
    image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')
  
  # 두번째 이미지 인코딩
  with open(image_path_2, 'rb') as image_file:
    image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

  # inputs 딕셔너리를 체인에 전달하면, LLM은 이미지와 텍스트를 동시에 고려하여 사용자에게 응답을 제공한다.
  return inputs

# 이미지를 base64로 로드 하는 함수
def load_image_as_base64(image_path):
  with open(image_path, 'rb') as img_file:
    return base64.b64encode(img_file.read()).decode('utf-8')

# ==================main=================== #

def main():
  st.set_page_config(page_title='FashionRAG', layout="wide")
  st.title('패션 및 스타일링 어시스턴트')

  # 데이터셋 폴더 및 이미지가 있는지 확인
  dataset_folder = os.path.join(current_directory, 'fasion_dataset')
  
  if not os.path.exists(dataset_folder) or not any(fname.endswith('.png') for fname in os.listdir(dataset_folder)):
    with st.spinner("데이터셋 설정 및 이미지 저장 중 ..."):
      dataset, dataset_folder = setup_dataset()
      save_images(dataset, dataset_folder)
    st.success("데이터셋 설정 및 이미지 저장")
  else:
    st.info('데이터셋 폴더와 이미지가 존재합니다. 설정을 건너뜁니다.')

  # 이미지를 효율적으로 검색하기 위해 벡터 데이터베이스가 설정되어 있는지 확인
  # 설정되어 있지 않다면 setup_chroma_db() 함수를 호출하여 벡터 데이터베이스를 생성하고
  # add_images_to_db()를 통해 이미지를 추가한다.
  vdb_path = os.path.join(current_directory, 'img_vdb')
  if not os.path.exists(vdb_path) or not os.listdir(vdb_path):
    with st.spinner("데이터셋 설정 및 이미지 저장 중 ..."):
      image_vdb = setup_chroma_db()
      add_images_to_db(image_vdb, dataset_folder)
    st.success("벡터 데이터베이스 설정 및 이미지 추가 완료")
  else:
    st.info('벡터 데이터베이스가 존재합니다. 설정을 건너 뜁니다.')
    image_vdb = setup_chroma_db()

  # 이미지를 기반으로 패션 스타일링 조언을 제공하기 위해 vision_chain을 설정
  # setup_vision_chain() 함수를 호출하여 LLM 모델을 사용한 시각 정보 처리 체인을 생성
  vision_chain = setup_vision_chain()
  st.header('스타일링 조언을 받아보세요.')

  # 사용자로부터 지속적으로 질문을 받기 위해 무한 루프 사용

  # while True:
  #   print('RAG 서비스가 준비됐습니다.')
  #   print('오늘 어떤 스타일에 대한 조언을 받고 싶은가요? (종료하려면 "quit" 입력)')

  query_ko = st.text_input("스타일링에 대한 질문을 입력하세요: ")

  #   if query_ko.lower() == 'quit':
  #     break
  if query_ko:
    with st.spinner('번역 및 쿼리 진행 중 ...'):
      # 영어로 번역
      query_en = translate(query_ko, 'English')
      results = query_db(image_vdb, query_en, results=2)
      # 프롬프트 입력
      prompt_input = format_prompt_inputs(results, query_en)
      # 영어로 답변 생성
      response_en = vision_chain.invoke(prompt_input)
      # 한국어로 번력
      response_ko = translate(response_en, "Korean")
    
    st.subheader('검색된 이미지: ')
    
    for idx, uri in enumerate(results['uris'][0]):
      img_base64 = load_image_as_base64(uri)
      img_data_url = f'data:image/jpeg;base64, {img_base64}'
      st.image(img_data_url, caption=f'이미지 {idx + 1}', width=300)
    st.subheader('스타일링 조언: ')
    st.markdown(response_ko)

    # print('\n검색된 이미지')
    # print_results(results)

    # print("=" * 100)

    # print("\nfashion rag 응답: ")
    # print(response_ko)
    

# 프로그램 실행
if __name__ == "__main__":
  main()