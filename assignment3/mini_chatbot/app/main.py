import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

# --- utils ---
from utils.llm_client import get_llm_response
from utils.logger import setup_logging, log_response

# FastAPI 인스턴스 생성
app = FastAPI(title="Mini Chatbot API")

# 로거 초기화 (앱 시작 시 1회 실행)
@app.on_event("startup")
def on_startup():
    """
    FastAPI 앱이 시작될 때, 로그 파일(CSV)이 준비되었는지 확인하고
    없으면 헤더와 함께 생성합니다.
    """
    print("FastAPI application startup...")
    setup_logging()

# Pydantic 요청 모델 정의
class ChatRequest(BaseModel):
    message: str
    prompt_version: str

# Pydantic 응답 모델 정의
class ChatResponse(BaseModel):
    reply: str
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chatbot 엔드포인트:
    1. LLM을 호출하여 응답을 받습니다.
    2. 응답 내용을 백그라운드에서 로그로 기록합니다.
    3. 사용자에게 최종 응답을 반환합니다.
    """
    
    # 1. LLM 클라이언트 호출
    # llm_data = {"reply": ..., "model": ..., "latency_ms": ..., "total_tokens": ...}
    llm_data = get_llm_response(request.message, request.prompt_version)
    
    # 2. 로그 데이터 생성
    #    FIELDNAMES 순서: ["timestamp", "prompt", "prompt_version", "model", "latency_ms", "total_tokens"]
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": request.message,  # 요청받은 message를 'prompt' 필드에 저장
        "prompt_version": request.prompt_version,
        "model": llm_data["model"],
        "latency_ms": llm_data["latency_ms"],
        "total_tokens": llm_data["total_tokens"]
    }
    
    # 3. 로그 기록
    #    CSV 파일 I/O(쓰기)가 API 응답 시간을 지연시키지 않도록
    #    FastAPI의 BackgroundTasks를 사용합니다.
    background_tasks.add_task(log_response, log_data)
    
    # 4. 사용자에게 응답 반환
    return {
        "reply": llm_data["reply"],
        "model": llm_data["model"]
    }