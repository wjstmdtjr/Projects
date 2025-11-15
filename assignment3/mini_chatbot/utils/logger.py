import csv
import os
from datetime import datetime

# 로그 파일 및 헤더 정의
LOG_FILE = "logs/llm_responses.csv"

# CSV 헤더 필드 정의
# log_data 딕셔너리의 키와 순서가 일치해야 합니다.
FIELDNAMES = [
    "timestamp", 
    "prompt",         # 사용자 원본 질문 (message)
    "prompt_version", 
    "model", 
    "latency_ms", 
    "total_tokens"
]

def setup_logging():
    """
    로거 초기화 함수:
    로그 파일이 존재하지 않으면, 파일을 새로 만들고 헤더를 씁니다.
    """
    
    # 1. logs 디렉토리가 있는지 확인하고 없으면 생성
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # 2. 로그 파일이 이미 있는지 확인
    file_exists = os.path.isfile(LOG_FILE)
    
    if not file_exists:
        # 파일이 없으면, 'write' 모드(w)로 새로 만들고 헤더를 씁니다.
        try:
            with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
                writer.writeheader()
            print(f"로그 파일 생성 완료: {LOG_FILE}")
        except IOError as e:
            print(f"로그 파일 생성 오류: {e}")

def log_response(log_data: dict):
    """
    로그 기록 함수:
    로그 데이터를 CSV 파일에 'append' 모드(a)로 한 줄 추가합니다.
    """
    
    # 1. 로그 파일이 존재하는지 다시 한번 확인 (setup_logging이 먼저 실행되었어야 함)
    if not os.path.isfile(LOG_FILE):
        print("로그 파일이 없습니다. setup_logging을 먼저 실행하세요.")
        # 만약의 경우를 대비해 다시 한번 헤더를 쓸 수도 있습니다.
        setup_logging()

    # 2. 'append' 모드(a)로 파일 열기 (기존 내용에 추가)
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            
            # log_data 딕셔너리에서 FIELDNAMES에 정의된 키만 추출하여 기록
            # (혹시 log_data에 'reply' 같이 헤더에 없는 키가 있어도 오류 방지)
            filtered_data = {key: log_data.get(key) for key in FIELDNAMES}
            
            writer.writerow(filtered_data)
            
    except IOError as e:
        print(f"로그 기록 오류: {e}")
    except KeyError as e:
        print(f"로그 데이터에 누락된 키가 있습니다: {e}. (데이터: {log_data})")