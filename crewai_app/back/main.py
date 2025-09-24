from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import asyncio
import json
import logging # 오류 및 정보를 자세히 기록하는데 사용

from crew import create_crew

# 기본적인 정보 수준 이상의 로그 기록
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 직렬화 함수 정의
# serialize_object 함수는 파인썬 객체럴 JSON으로 직렬화하는데 사용
# 특히 FastAPI가 복잡한 사용자 정의 객체를 처리하는 경우, 직렬화하지 못할 수 있기 때문에 이를 해결하기 위한 함수이다.
# 튜플일 경우 리스트 형태로 반환되고, 객체는 딕셔너리 형태로 반환되고, 객체의 속성이 있으면 그 속성을 반환하고, 그렇지 않으면 문자열로 반환한다.\

def serialize_object(obj):
  if isinstance(obj, tuple):
    return list(obj)
  elif isinstance(obj, list):
    return [serialize_object(item) for item in obj]
  elif isinstance(obj, dict):
    return {key: serialize_object(value) for key, value in obj.items()}
  elif hasattr(obj, '__dict__'):
    result = {}
    for key, value in obj.__dict__.items():
      if not key.startswith('_'):
        result[key] = serialize_object(value)
    return result
  elif hasattr(obj, 'raw'): # TaskOutput 객체는 raw 속성을 반환
    return str(obj.raw)
  else:
    return str(obj)
  
# JSON 데이터를 직렬화하여 문자열로 변환한다.
# serialize_object 함수를 사용하여 복잡한 객체를 처리하여 ensure_ascii=False를 사용하여 UTF-8 인코딩을 유지한다.
# indent=4를 적용하여 가독성을 높인다.

def custom_json_dumps(data):
  return json.dumps(data, default=serialize_object, ensure_ascii=False, indent=4)

# FastAPI 인스턴스 생성
app = FastAPI(
  title="CrewAI Contents  Generator",
  version="1.0.0",
  description="토픽을 기반으로 CrewAI를 사용하여 블로그 콘텐츠를 생성하는 API 서비스"
)

# CORS 처리
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], # 모든 도메인 요청 허용
  allow_credentials=True, # 인증 정보 허용
  allow_methods=["*"], # 모든 HTTP 메서드 허용
  allow_headers=["*"] # 모든 헤더 정보 허용
)

# TopicInput 모델 정의
# pydantic의 BaseModel을 상속하여 입력 데이터를 정의한다.
# 클라이언트가 API에 POST 요청을 보낼 때 JSON 데이터를 검증하는 역할을 한다.
# 여기서는 topic 필드만 포함되며, 이를 기반으로 CrewAI에서 블로그 콘텐츠를 생성한다
class TopicInput(BaseModel):
  topic: str

@app.get("/")
async def root():
  return {"message": "Hello World"}

# 엔드포인트
@app.post("/crewai")
async def crewai_endpoint(input: TopicInput):
  try:
    crew = create_crew()
    loop = asyncio.get_event_loop() # 비동기 이벤트 루프 호출
    # 동기적으로 동작하는 crew.kickoff 함수를 비동기적으로 실행하기 위해 run_in_executor 함수를 사용한다.
    # 여기서 첫 번째 인자를 None으로 지정하면 기본적으로 스레드 풀(Thread Pool)을 사용한다. 이는 새로운 스레드를 생성하여 crew.kickoff를 백그라운드에서 실행한다. 이후 kickoff는 크루를 시작하고 주제에 맞는 콘텐츠를 생성한다.
    result = await loop.run_in_executor(None, crew.kickoff, {"topic": input.topic})
    serialized_json_string = custom_json_dumps(result)
    serialized_result = json.loads(serialized_json_string)

    return JSONResponse(content=serialized_result)
  except Exception as e:
    logger.error(f"CrewAI endpoint error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
  
# 메인 벡엔드 실행
if __name__ == "__main__":
  import uvicorn
  uvicorn.run("main:app", host="0.0.0.0", port=8000)