# ChatPDF 응용한 streamlit 활용 앱

# 필요 모듈 설치
import os  # 운영 체제와 상호작용하기 위한 모듈 (예: 파일 경로, 환경 변수 등)
import tempfile  # 임시 파일과 디렉토리를 만들기 위한 모듈
from langchain import text_splitter
import streamlit as st  # Streamlit 앱을 만들기 위한 웹 프레임워크
# from streamlit_extras.buy_me_a_coffee import button  # "Buy Me a Coffee" 버튼을 추가하기 위한 streamlit_extras 모듈의 기능
from langchain_community.docstore.in_memory import InMemoryDocstore  # LangChain 커뮤니티에서 제공하는 메모리 기반 문서 저장소
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_community.vectorstores import FAISS
import faiss

# 스트림릿 지침 설정
st.title('ChatPDF with Multiquery, Hybrid RAG Fustion')
st.write('----------------')
st.write('PDF 파일을 업로드하고 내용을 기반으로 질문해 주세요.')

# Openai key 입력
openai_key = st.text_input('OpenAI API 키를 입력해 주세요', type='password')

st.write(openai_key)

# GPT 모델 선택
model_choice = st.selectbox(
  '사용한 LLM 모델을 선택하세요.',
  ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
)

# 파일 업로드
uploaded_file = st.file_uploader('PDF 파일을 업로드해 주세요.', type=['pdf'])

# PDF 문서 변환(임시 디렉토리 저장)
def pdf_to_document(uploaded_file):
  temp_dir = tempfile.TemporaryDirectory() # 임시 디렉토리 생성
  temp_filepath = os.path.join(temp_dir.name, uploaded_file.name) # 임시 디렉토리 파일 저장
  with open(temp_filepath, 'wb') as f:
    f.write(uploaded_file.getvalue())
  loader = PyPDFLoader(temp_filepath)
  pages = loader.load_and_split()
  return pages

# 문서 포멧
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

# 파일 업로드 확인
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)

  # 문서를 청크로 분할
  # from_tiktoken_encoder: 일반적인 텍스트 분할은 문자 또는 단어 단위로 이뤄진다. 하지만 GPT 모델과 같은 언어 모델은
  # 입력을 처리할 때 토큰 단위로 처리한다
  # tiktoken은 OpenAI에서 제공하는 토크나이저로 GPT 모델이 사용하는 토큰화 방식과 동일하다
  # 따라서 from_tiktoken_encoder를 사용하면 모델의 토큰화 방식에 맞춰 텍스트를 분할할 수 있어
  # 각 청크가 모델의 토큰 제한을 초과하지 않도록 효과적으로 관리할 수 있다.

  # Split
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50
  )

  splits = text_splitter.split_documents(pages)

  # 임베딩 및 벡터 스토어 설정
  embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

  # FAISS 생성
  # L2 유클리드 거리 사용: 임베딩 벡터 간의 유클리드 거리를 계산하여 비슷한 문서를 찾는 검색 인덱스 생성
  index = faiss.IndexFlatL2(len(embeddings_model.embed_query('hello world'))) # 임시 문자열 기준 유클리드 거리 측정

  # 벡터 스토어
  vectorstore = FAISS(
    embedding_function=embeddings_model,
    index=index,
    docstore = InMemoryDocstore(), # 문서 저장 시 메모리 기반 저장소 생성
    index_to_docstore_id={}
  )

  # 문서 청크를 벡터 스토어에 추가
  vectorstore.add_documents(documents=splits, ids=range(len(splits)))

  # FAISS 리트리버 생성
  # - 역할: 벡터 스토어에서 쿼리에 가장 유사한 문서를 검색
  # - 방식: vectorstore.as_retriever 메서드를 사용해 벡터 스토어 검색 기능을 제공하는 리트리버로 변환
  # - 예시: 사용자가 질문을 입력하면 faiss_retriever는 벡터 스토어에서 쿼리와 가장 유사한 상위 4개의 문서를 검색. 이후 MMR 알고리즘을 적용하여 4개의 문서 중에서 중복을 최소화하고 관련성 높은 1개의 문서를 최종적으로 선택

  # BM25 리트리버 생성
  # - 역할: 전통적인 BM25 알고리즘을 사용하여 텍스트 기반의 문서 검색을 수행
  # - 방식: BM25Retriever.from_documents(splits)를 사용하여 분할된 문서 청크로부터 BM25 리트리버 생성

  # 앙상블 리트리버 생성
  # - 역할: 두 개의 리트리버(BM25, FAISS)를 결합하여 하이브리드 검색을 수행
  # - 방식: EnsembleRetriever를 사용하여 여러 리트리버의 결과를 통합

  # - 매개변수
  # -- retrievers=[bm25_retriever, faiss_retriever]: 결합할 리트리버들의 리스트를 지정
  # -- weights=[0.2, 0.8]: 각 리트리버의 가중치를 설정하여 검색 결과에 대한 기여도 조절 - 0.2는 BM25, 0.8은 FAISS

  # - 하이브리드 검색의 이점
  # -- BM25 리트리버는 키워드 매칭에 강하며, 전통적인 정보 검색 기법으로 쿼리와 문서 간의 용어 일치를 기반으로 한다.
  # -- FAISS 리트리버는 임베딩 벡터를 사용하여 의미적 유사성을 기반으로 검색한다.
  # -- 두 리트리버를 결합하면 키워드 매칭과 의미적 유사성을 모두 고려하여 더 정확하고 풍부한 검색 결과를 얻을 수 있다.

  # FAISS 리트리버
  faiss_retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={"k": 1, "fetch_k": 4}
  )

  # bm25 리트리버
  bm25_retriever = BM25Retriever.from_documents(splits)
  bm25_retriever.k=2

  # 앙상블 리트리버
  ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.2, 0.8]
  )

  # RAG-Fusion을 위한 쿼리 설정
  template = """
  당신은 AI 언어 모델 조수 입니다. 당신의 임무는 주어진 사용자 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다.
  사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는데 도움을 주는 것이 목표입니다.
  각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
  """
  prompt_perspectives = ChatPromptTemplate.from_template(template)

  generate_queries = (
    prompt_perspectives
    | ChatOpenAI(model_name=model_choice, temperature=0, openai_api_key=openai_key)
    | StrOutputParser()
    | (lambda x: x.split("\n")) # 전달받은 템플릿 문자열을 줄바꿈 기준으로 분리
  )

  # RRF 적용 함수
  def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
    """
    여러개의 순위가 매겨진 문서 리스트를 받아, RRF 공식을 사용해 문서의 최종 순위 계산
    Args:
      results: 순위가 매겨진 문서 리스트
      k: 공식에 사용되는 상수
      top_n: 반환할 우선순위가 높은 문서의 개수
    Returns:
      reranked_results: 알고리즘에 따라 재정렬된 문서 리스트
    """
    fused_scores = {}

    for docs in results:
      for rank, doc in enumerate(docs):
        doc_str = dumps(doc)
        if doc_str not in fused_scores:
          fused_scores[doc_str] = 0
        fused_scores[doc_str] += 1 / (k + rank)

    reranked_results = [
      (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results[:top_n]

  # RAG_Fusion
  retrieval_chain_rag_fusion = generate_queries | ensemble_retriever.map() | reciprocal_rank_fusion

  # Final RAG 체인 설정
  template = """
  다음 맥락을 바탕으로 질문에 답변하세요:
  {context}
  질문: {question}
  """
  prompt = ChatPromptTemplate.from_template(template)
  llm = ChatOpenAI(model_name=model_choice, temperature=0, openai_api_key=openai_key)

  final_rag_chain = (
    {'context': retrieval_chain_rag_fusion, 'question': RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
  )

  # 답변 생성 및 화면 출력
  st.header('PDF에 질문하세요.')
  question = st.text_input('질문을 입력해 주세요.')

  if st.button('질문하기'):
    with st.spinner('답변 생성 중 ...'):
      result = final_rag_chain.invoke(question)
      st.write(result)