# main.py (Langfuse v2.25.1 호환 최종 코드)

import os
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import OpenAI

# .env 파일 로드를 시도합니다.
load_dotenv()

print("--- [Phase 2] V2 프롬프트 테스트 시작 ---")

# 1. .env에서 직접 키를 읽어옵니다.
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 키가 제대로 로드되었는지 확인합니다.
if not secret_key or not public_key or not openai_api_key:
    raise ValueError("CRITICAL: .env 파일에서 API 키를 찾을 수 없습니다. 파일 위치와 내용을 다시 확인해주세요.")
else:
    print("✅ .env 파일에서 API 키를 성공적으로 읽었습니다.")

# 2. Langfuse()를 초기화할 때, 키를 직접 인자로 전달합니다.
client = Langfuse(
    secret_key=secret_key,
    public_key=public_key
)
openai_client = OpenAI(api_key=openai_api_key)
print("✅ Langfuse 및 OpenAI 클라이언트 초기화 성공.")

def get_langfuse_prompt(prompt_name):
    """Langfuse UI에 저장된 프롬프트를 이름과 라벨로 가져옵니다."""
    try:
        # get_prompt()는 Langfuse v2.x의 기능입니다.
        prompt = client.get_prompt(prompt_name)
        print(f"✅ Langfuse에서 '{prompt_name}' 프롬프트 (Version: {prompt.version}, 최신)를 성공적으로 가져왔습니다.")
        return prompt
    except Exception as e:
        print(f"🚨 Langfuse에서 프롬프트를 가져오는 데 실패했습니다: {e}")
        return None


def summarize_meeting_generator(prompt_text, transcript, prompt_version):
    """주어진 프롬프트와 스크립트로 회의록을 생성하고 Langfuse에 Trace를 남깁니다."""
    
    trace = client.trace(
        name="summarize-meeting-generator",
        input={"transcript": transcript},
        metadata={"prompt_version": prompt_version}
    )
    
    generation = trace.generation(
        name="minutes-generation",
        model="gpt-4o-mini",
        input=[{"role": "system", "content": prompt_text},
               {"role": "user", "content": transcript}],
        metadata={"temperature": 0.7}
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": transcript}
            ],
            temperature=0.7,
        )
        output = response.choices[0].message.content
        generation.end(output=output, usage=response.usage)
        return output
    except Exception as e:
        print(f"오류 발생: {e}")
        generation.end(output={"error": str(e)}, level="ERROR")
        return None

# --- 테스트 실행 영역 ---
if __name__ == "__main__":
    langfuse_prompt = get_langfuse_prompt("summarize-meeting-generator")

    transcripts = [
        "김팀장: 3분기 마케팅 예산이 계획보다 15% 초과됐습니다. 원인은 신규 채널 광고 집행 때문입니다. 다음 주 월요일까지 이대리님이 비용 절감안 보고서 좀 작성해주세요.",
        "박부장: 신제품 출시일은 11월 15일로 확정하고, 전 부서에 공지하도록 하죠. 마케팅팀은 바로 홍보 계획 수립에 착수해주세요.",
        "Alice: The main issue is performance on older devices. The dev team needs to focus on optimization for the next two weeks. Bob, please assign tasks.",
        "최상무: A B2B 서비스의 연간 구독료를 5% 인상하는 안건입니다. 반대 의견 있으신가요? ... 좋습니다. 그럼 내년 1월 1일부터 인상된 가격으로 계약 갱신하도록 재무팀에 전달하세요.",
        "정실장: 고객센터 인력 충원이 시급합니다. 이번 달 말까지 최소 2명의 상담원을 추가 채용하는 것을 목표로 인사팀과 협력해 주세요. 이 팀장님이 담당입니다."
    ]

    print(f"\n총 {len(transcripts)}건의 회의록 요약을 시작합니다...")

    for i, script in enumerate(transcripts):
        print(f"\n===== [실행 {i+1}] =====")
        result = summarize_meeting_generator(
            prompt_text=langfuse_prompt.prompt, 
            transcript=script,
            prompt_version=langfuse_prompt.version
        )
        print(result)

    # 모든 작업이 끝난 후, client 객체를 사용해 전체 추적을 종료/전송합니다.
    client.shutdown()

    print("\n\n✅ [Phase 1 완료] 모든 실행이 완료되었습니다. Langfuse 프로젝트에서 5개의 Trace를 확인하세요.")