import os
import base64
from langchain_core.messages import HumanMessage # 사용자 메시지 관리 모듈
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 주어진 이미지 파일을 base64 형식으로 인코딩
# 이미지 데이터를 텍스트 형식으로 변환하여 전송하거나 저장할 때 사용
def encode_image(image_path):
  with open(image_path, 'rb') as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# OpenAI를 호출하여 이미지를 요약한다.
# 이미지는 Base64로 인코딩된 형태로 전달되며, GPT-4 Vison 모델을 사용하여 이미지 요약본을 생성한다.
# 프롬프트는 이미지 설명을 포함하고 있으며 이를 통해 모델이 이미지 요약 작업을 수행할 수 있다.
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

  # print(msg)
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

# 현재 디렉토리 경로
current_directory = os.path.dirname(os.path.abspath(__file__))
figures_directory = os.path.join(current_directory, 'figures')

# print(figures_directory)

# 이미지 요약 생성
img_base64_list, image_summaries = generate_img_summaries(figures_directory)
# print(f"이미지 개수: {len(img_base64_list)}")
# print(f"이미지 요약 개수: {len(image_summaries)}")

# content 정보만 출력
print("=== 이미지 요약 내용 ===")
for i, summary in enumerate(image_summaries):
  print(f"\n[이미지 {i + 1}]")
  print(summary)