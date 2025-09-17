# FAISS
# FAISS는 Facebook AI에서 만든 라이브러리로 대규모 벡터 데이터에서 유사한 항목을 빠르게 찾아주는 역할을 한다.
# 이는 similarity search 알고리즘을 사용하며 자연어 처리, 추천 시스템, 이미지 검색 등에서 활용된다.

# FAISS에서 사용하는 알고리즘은 정확도를 약간 희생하더라도 속도를 극대화 하는 
# 근사 최근접 이웃(Approximate Nearest Neighbor, ANN)을 기반으로 하면 CPU에서도 대용량 벡터 처리에 최적화 돼 있다.

# 필요 모듈
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 텍스트 청크 분할 -> 텍스트 데이터 로드 -> 문서 객체 변환

# 현재 위치 저장
current_dir = os.path.dirname(os.path.abspath(__file__))

# 벡터 훈련 데이터 저장 경로
restaurant_faiss = os.path.join(current_dir, 'restaurant-faiss')

# 텍스트 파일 저장
loader = TextLoader(os.path.join(current_dir, 'data', 'restaurants.txt'), encoding="utf-8")

# 파일 내용을 document 객체로 로드
documents = loader.load()

# 텍스트 청크 분할
# 텍스트를 300자 단위로 나누고, 연속된 청크 사이에 50자의 겹침을 두어 텍스트를 분할

# 청크 분할 예시
# - 기술 메뉴얼처럼 장문이 많은 자료: chunk=800, overlap=80
# - FAQ 키워드 검색처럼 빠른 응답을을 요하는 자료: chunk=200, overlap=20
# - 메모리가 작은 모바일 기기: chunk=128, overlap=12
# - 대용량 GPU 서버에서 최고 품질 한국어 RAG를 목표할 때: chunk=1024, overlap=150
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# 로드된 문서를 지정된 크기와 겹침에 따라 더 작은 청크로 분할
docs = text_splitter.split_documents(documents)

# 임베딩
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 생성된 벡터를 사용하여 FAISS 인덱스 생성
db = FAISS.from_documents(docs, embeddings)

# 생성된 인덱스를 디렉토리(restaurant-faiss)에 저장
db.save_local(restaurant_faiss)
print('저장 완료!')

# index.faiss: 라이브러리가 생성한 인덱스 파일로 벡터 데이터가 저장
# index.pkl: 파이썬의 pickle 형식으로 저장된 메다데이터 파일로 인덱스와 관련된 추가 정보를 포함