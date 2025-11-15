import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경 변수(API 키)를 로드합니다.
load_dotenv()

# OpenAI 클라이언트 초기화
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 실패: {e}")
    client = None

# 프롬프트 버전별 시스템 메시지 정의
SYSTEM_PROMPTS = {
    "v1": "당신은 친절하고 간결하게 답변하는 AI 챗봇입니다.",
    "v2": "당신은 AI 전문가입니다. 질문에 대해 상세하고, 구조화된 답변을 제공해야 합니다."
}

# 사용할 LLM 모델 정의
MODEL_NAME = "gpt-4o-mini"

def get_llm_response(message: str, prompt_version: str) -> dict:
    """
    OpenAI API를 호출하여 응답을 받고,
    지연 시간 및 토큰 수를 포함한 딕셔너리를 반환합니다.
    """
    
    # 프롬프트 버전 선택
    system_prompt = SYSTEM_PROMPTS.get(prompt_version, SYSTEM_PROMPTS["v1"])
    
    start_time = time.time()
    
    try:
        if not client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다.")

        # OpenAI API 호출
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )
        
        # 응답 데이터 추출
        reply = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        model_used = response.model # API가 실제로 응답한 모델 이름
    
    except Exception as e:
        # API 호출 실패 시 에러 메시지 반환
        print(f"Error calling OpenAI API: {e}")
        reply = f"API 호출 중 오류가 발생했습니다: {e}"
        total_tokens = 0
        model_used = "error-model"

    # Latency(지연시간) 계산 (ms)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # 요구된 딕셔너리 형식으로 반환
    return {
        "reply": reply,
        "model": model_used,
        "latency_ms": latency_ms,
        "total_tokens": total_tokens
    }