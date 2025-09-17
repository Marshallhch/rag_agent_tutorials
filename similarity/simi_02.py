# 쿼리 기반 유사 문서 검색 
# 로컬에 저장된 인덱스 호출

# 필요 모듈 설치
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))

async def main():
  # 임베딩 초괴화
  embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

  # 지정됨 임베딩을 사용하여 로컬에 저장된 FAISS 인덱스 호출
  # allow_dangerous_deserialization=True 파라미터를 사용하여 디스크에서 읽어온 데이터를 안전하게 역직렬화
  load_db = FAISS.load_local(os.path.join(current_dir, 'restaurant-faiss'), embeddings, allow_dangerous_deserialization=True)

  # 질의
  query = '음식점의 룸 서비스는 어떻게 운영되나요?'

  # k 값은 유사한 문서 개수
  # result = load_db.similarity_search(query, k=2)
  # print(result)

  # 쿼리를 벡터로 변환
  embedding_vector_query = embeddings.embed_query(query)
  # print(embedding_vector_query)

  # asimilarity_search_by_vector 메서드를 통해 벡터 기반의 비동기 검색 수행
  # 검색된 문서 중 가장 유사한 첫 번째 문서 출력
  docs = await load_db.asimilarity_search_by_vector(embedding_vector_query)
  print(docs[0])

# 함수 실행
if __name__ == '__main__':
  asyncio.run(main())