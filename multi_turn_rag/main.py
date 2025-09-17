# Response API
# 개발자가 자신의 애플리케이션에 사용자 대신 독립적으로 작업을 수행하는 AI 에이전트를 적용할 수 있는 서비스.
# OpenAI LLM을 쉽게 연동할 수 있고, Response API를 한 번만 호출해도 복잡한 단계의 작업을 자동화 할 수 있다.
# 또한 내장 개발 도구(Tools)를 활용해 웹 검색과 파일 검색을 연동해 더 정확하고 빠른 답변을 얻는다.
# 함수 호출을 통해 직접 구현한 함수를 연동하여 고도화된 AI 애플리케이션을 만들 수 있다.

# Response API vs Langchain
# 둘 모두 대화형 AI 애플리케이션을 만들 수 있지만 제공하는 개발 도구와 구현 자유도가 다르다

# - Response API: 
#   - OpenAI LLM을 기반으로 대화형 애플리케이션을 쉽게 구축할 수 있도록 설계됐다.
#   - 코드를 직접 작성하지 않고도 웹 기반 인터페이스인 플레이그라운드를 통해 애플리케이션에 필요한 코드를 생성할 수 있다.
#   - 내장된 파일 검색 도구를 사용하면 문서 기반 질문에 답하는 문서 검색기(Retriever)를 연동할 수 있다.

# - Langchain
#   - 더 복잡한 어플리케이션의 개발, 배포, 유지 보수를 위한 프레임워크다.
#   - 랭체인에서는 OpenAI, Meta, Anthropic 등의 LLM을 포함해 1000개 이상의 서드파티 개발 도구를 연동할 수 있다.
#   - 다양한 도구를 지원하기 때문에 구현 난이도가 높지만, 그만큼 자유롭게 내가 원하는 어플리케이션을 구현할 수 있다.

# 플레이그라운드 벡터 파일 저장
# 1. https://platform.openai.com/chat/edit?models=gpt-5 접속
# 2. System Message: 당신은 소설 운수 좋은 날을 집필한 현진건 작가 입니다. <- 작성
# 3. 상단 Add 버튼 클릭 -> File Search 선택 -> unsu.pdf 파일 업로드 -> Attach 버튼 클릭
# 4. 벡터 스토어 코드 복사: https://platform.openai.com/docs/guides/tools-file-search
# 5. id: vs_68ca0b90faf48191a45f839ab6816ae6

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

# 질의 응답 생성
response = client.responses.create(
  model='gpt-4o-mini',
  instructions='당신은 소설 운수 좋은 날을 집필한 현진건 작가 입니다.',
  input='아내가 먹고싶어하는 음식은 뭐야?',
  tools=[{
    "type": "file_search",
    "vector_store_ids": ["vs_68ca0b90faf48191a45f839ab6816ae6"]
  }]
)
print(response.output_text)

# Response API의 previous_response_id 파라미터에 직전 응답의 ID 값을 전달하면 지난 대화의 맥락을 기반으로 답변을 생성할 수 있다.
second_response = client.responses.create(
  previous_response_id=response.id,
  model='gpt-4o-mini',
  instructions='당신은 소설 운수 좋은 날을 집필한 현진건 작가 입니다.',
  input='방금 한 말을 단문으로 작성해줘',
  tools=[{
    "type": "file_search",
    "vector_store_ids": ["vs_68ca0b90faf48191a45f839ab6816ae6"]
  }]
)
print(second_response.output_text)