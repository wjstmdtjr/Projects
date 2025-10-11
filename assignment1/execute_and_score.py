import os
import json
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import OpenAI

# --- 초기 설정 ---
load_dotenv()
print("--- [최종 평가] 생성 및 채점 통합 스크립트 시작 ---")

secret_key = os.getenv("LANGFUSE_SECRET_KEY")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not secret_key or not public_key or not openai_api_key:
    raise ValueError("CRITICAL: .env 파일에서 API 키를 찾을 수 없습니다.")

client = Langfuse(secret_key=secret_key, public_key=public_key)
openai_client = OpenAI(api_key=openai_api_key)
print("✅ 클라이언트 초기화 성공.")

# --- 평가자(Scorer) 함수 정의 ---

def score_json_validity(output_str):
    try:
        json.loads(output_str); return 1
    except: return 0

def score_factual_consistency(input_transcript, output_str):
    # ... (이전과 동일한 LLM-as-a-Judge 함수)
    judge_prompt_template = """
    Evaluate if the output is factually consistent with the original transcript. Score 1 if consistent, 0 if not. Output only a single number: 0 or 1.
    [original_transcript]: {input_transcript}
    [output]: {output_str}
    """
    judge_prompt = judge_prompt_template.format(input_transcript=input_transcript, output_str=output_str)
    try:
        response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": judge_prompt}], temperature=0)
        return int(response.choices[0].message.content.strip())
    except Exception: return None

# [수정됨] Action Item 유사성 판단 함수 (더 유연한 기준 적용)
def are_tasks_semantically_similar(task1, task2):
    """
    두 작업(task) 설명이 의미적으로 유사한지 판단합니다.
    언어가 다르거나(한/영), 표현 방식이 달라도 핵심 내용이 같으면 유사하다고 봅니다.
    """
    judge_prompt = f"""
    Your role is to determine if two action items are semantically similar. Answer only with "yes" or "no".
    Consider them similar even if they are in different languages (Korean/English) or use slightly different phrasing, as long as they describe the same core task.

    Here are some examples:
    - Task A: "마케팅팀과 다음 주에 미팅 잡기"
      Task B: "Schedule a follow-up meeting with the marketing team for next week"
      -> yes
    
    - Task A: "send the client the final proposal"
      Task B: "클라이언트에게 최종 제안서 전달"
      -> yes

    - Task A: "1분기 실적 보고서 초안 작성"
      Task B: "Draft the Q1 performance report"
      -> yes

    - Task A: "서버 아키텍처 다이어그램 업데이트"
      Task B: "프로젝트 예산 검토"
      -> no

    Now, evaluate the following tasks:
    Task A: "{task1}"
    Task B: "{task2}"
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": judge_prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception:
        # API 오류 발생 시, 안전하게 False를 반환하여 점수에 영향을 주지 않도록 함
        print(f"Warning: Semantic similarity check failed for tasks: '{task1}' vs '{task2}'")
        return False

def score_action_item_recall(expected_output_dict, output_str):
    try:
        model_output_dict = json.loads(output_str)
        expected_items = expected_output_dict.get("action_items", [])
        model_items = model_output_dict.get("action_items", [])
        
        if not isinstance(expected_items, list) or not isinstance(model_items, list): return 0.0
        if not expected_items: return 1.0

        found_count = 0
        for expected in expected_items:
            # any()를 사용하여 모델이 생성한 아이템 중 하나라도 의미가 같으면 True로 판단
            # 수정된 are_tasks_semantically_similar 함수가 여기서 사용됩니다.
            if any(are_tasks_semantically_similar(expected.get("task"), model_item.get("task")) for model_item in model_items):
                found_count += 1
        
        return found_count / len(expected_items)
    except (json.JSONDecodeError, TypeError):
        return 0.0

# --- 통합 실행 및 채점 로직 ---
if __name__ == "__main__":
    DATASET_NAME = "summarize-meeting-eval-set"
    PROMPT_NAME = "summarize-meeting-generator"

    # 1. 데이터셋과 프롬프트들을 가져옵니다.
    dataset = client.get_dataset(DATASET_NAME)
    prompt_v1 = client.get_prompt(PROMPT_NAME, version=5)
    prompt_v2 = client.get_prompt(PROMPT_NAME, version=6)
    prompts_to_evaluate = [prompt_v1, prompt_v2]
    print(f"✅ 데이터셋 '{DATASET_NAME}'과 프롬프트 V1, V2 로드 완료.")

    # 2. 각 프롬프트에 대해 데이터셋의 모든 아이템을 실행하고 즉시 채점합니다.
    for prompt in prompts_to_evaluate:
        print(f"\n--- 프롬프트 Version {prompt.version} 실행 및 채점 시작 ---")
        is_json_prompt = prompt.version == 6

        for i, item in enumerate(dataset.items):
            # 2a. Trace 생성
            trace = client.trace(name="final-evaluation-run", metadata={"prompt_version": prompt.version})
            
            # 2b. LLM 호출하여 결과 생성
            generation = trace.generation(name="generation", input=item.input)
            response_format_config = {"type": "json_object"} if is_json_prompt else {"type": "text"}
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt.prompt}, {"role": "user", "content": str(item.input)}],
                temperature=0.7, response_format=response_format_config
            )
            model_output = response.choices[0].message.content
            generation.end(output=model_output, usage=response.usage)
            
            # 2c. 생성된 결과 바로 채점
            consistency_score = score_factual_consistency(str(item.input), model_output)
            if consistency_score is not None:
                trace.score(name="factual-consistency", value=consistency_score)

            if is_json_prompt:
                validity_score = score_json_validity(model_output)
                trace.score(name="json-validity", value=validity_score)
                
                recall_score = score_action_item_recall(item.expected_output, model_output)
                trace.score(name="action-item-recall", value=recall_score)

            print(f"   ... 아이템 {i+1}/10 처리 및 채점 완료")

    client.shutdown()
    print("\n\n✅ [통합 평가 완료] 모든 결과 생성 및 점수 기록이 Langfuse에 완료되었습니다.")