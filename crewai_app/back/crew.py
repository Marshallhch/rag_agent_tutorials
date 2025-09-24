# Process: 작업 순서를 순차적으로 수행하도록 지정하는 모듈
from crewai import Crew, Process
from agents import planner, writer, translator_kor, editor
from tasks import plan, write, edit, translate

def create_crew():
  crew = Crew(
    agents = [planner, writer, editor, translator_kor],
    tasks = [plan, write, edit, translate],
    process = Process.sequential,
    verbose = True
  )
  return crew