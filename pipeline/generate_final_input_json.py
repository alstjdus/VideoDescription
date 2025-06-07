import json
from datetime import datetime

def parse_time(time_str):
    """00:00:00,000 형식을 datetime 객체로 변환"""
    return datetime.strptime(time_str, "%H:%M:%S,%f")

def is_overlap(start1, end1, start2, end2):
    """두 시간 구간이 겹치는지 확인"""
    return max(start1, start2) < min(end1, end2)

# JSON 파일 불러오기
with open('yolo_results.json', 'r', encoding='utf-8') as f:
    frames = json.load(f)

with open('video_script_large.json', 'r', encoding='utf-8') as f:
    subtitles = json.load(f)

# 자막 타임스탬프 파싱
for sub in subtitles:
    sub_start, sub_end = sub["timestamp"].split(" --> ")
    sub["start"] = parse_time(sub_start.strip())
    sub["end"] = parse_time(sub_end.strip())

# 프레임별 자막 병합
for frame in frames:
    frame_start, frame_end = frame["timestamp"].split(" --> ")
    start_time = parse_time(frame_start.strip())
    end_time = parse_time(frame_end.strip())

    # 겹치는 자막 수집
    matched_dialogues = []
    for sub in subtitles:
        if is_overlap(start_time, end_time, sub["start"], sub["end"]):
            matched_dialogues.append(sub["dialogue"])
    
    # 프레임에 대사 추가
    frame["dialogue"] = matched_dialogues

# 결과 저장
with open('final_input.json', 'w', encoding='utf-8') as f:
    json.dump(frames, f, ensure_ascii=False, indent=2)

print("통합 JSON 생성 완료: final_input.json")
