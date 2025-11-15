import pandas as pd
import sys

LOG_FILE = "logs/llm_responses.csv"

def analyze_logs():
    """
    logs/llm_responses.csv 파일을 읽어 통계를 계산하고 출력합니다.
    """
    try:
        # CSV 파일 로드
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print(f"오류: 로그 파일 '{LOG_FILE}'을(를) 찾을 수 없습니다.")
        print("Phase 5 (데이터 생성)를 먼저 실행했는지 확인하세요.")
        sys.exit(1)

    if df.empty:
        print("오류: 로그 파일이 비어있습니다. API를 호출하여 데이터를 생성하세요.")
        sys.exit(1)

    print("--- 📊 9·10주차 통합 과제 로그 분석 ---")

    # 1. 프롬프트 버전별 평균 지연시간
    latency_by_version = df.groupby('prompt_version')['latency_ms'].mean()
    print("\n[1. 프롬프트 버전(v1/v2)별 평균 지연시간 (ms)]")

    print(latency_by_version.to_markdown(floatfmt=".2f"))

    # 2. 프롬프트 버전별 평균 토큰 수
    tokens_by_version = df.groupby('prompt_version')['total_tokens'].mean()
    print("\n[2. 프롬프트 버전(v1/v2)별 평균 토큰 수]")
    print(tokens_by_version.to_markdown(floatfmt=".2f"))

    # 3. 모델별 평균 지연시간
    try:
        latency_by_model = df.groupby('model')['latency_ms'].mean()
        print("\n[3. 사용 모델별 평균 지연시간 (ms)]")
        print(latency_by_model.to_markdown(floatfmt=".2f"))
    except KeyError:
        print("\n[3. 'model' 필드를 찾을 수 없거나 데이터가 불충분합니다.]")

    print("\n--- 분석 완료 ---")
    print("위 표를 복사하여 summary.md 파일에 붙여넣으세요.")

if __name__ == "__main__":
    analyze_logs()