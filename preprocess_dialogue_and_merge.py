# srt 대본에서 대사만 추출
import re
import json

def extract_dialogue_with_timestamp(txt_path, output_path=None):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    timestamp = None
    buffer = []

    for line in lines:
        line = line.strip()

        # 숫자만 있는 줄(씬 넘버 등) 제외
        if re.fullmatch(r"\d+", line):
            continue

        # 타임스탬프 탐지
        if "-->" in line:
            timestamp = line
            continue

        # 줄이 비었으면, 지금까지 모은 대사 저장 후 초기화
        if not line and buffer:
            dialogue = " ".join(buffer).strip()
            # dialogue가 비거나 의미 없는 경우 제외
            if dialogue and dialogue not in ("-", "--", "- -"):
                results.append({
                    "timestamp": timestamp,
                    "dialogue": dialogue
                })
            buffer = []
            timestamp = None
            continue

        # 줄이 있고 타임스탬프가 있으면 처리
        if line and timestamp:
            # 대사 안의 {} 설명문 제거 (완전히 제거)
            line = re.sub(r"\{.*?\}", "", line)
            # 대사 안의 [] 설명문 제거 (완전히 제거)
            line = re.sub(r"\[.*?\]", "", line)
            # 제거 후 공백 정리
            line = line.strip()
            if line:
                buffer.append(line)

    # 마지막 대사 처리
    if buffer and timestamp:
        dialogue = " ".join(buffer).strip()
        if dialogue and dialogue not in ("-", "--", "- -"):
            results.append({
                "timestamp": timestamp,
                "dialogue": dialogue
            })

    # 저장
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results
  
# 사용 예시
dialogue_list = extract_dialogue_with_timestamp(
    txt_path="/content/drive/MyDrive/data/srt/drama1.S01E02.episode1.txt",  # 실제 경로 반영
    output_path="/content/drive/MyDrive/data/frame/drama1/episode1/dialogue_timestamp.json"  # 저장 경로
)

print(f"{len(dialogue_list)}개의 대사-타임스탬프 쌍을 추출했습니다.")

import json
import os
from datetime import datetime

# === 경로 설정 ===
script_path = "/content/drive/MyDrive/data/frame/drama1/episode1/script/episode1.json"        # 캡션 포함 json
dialogue_txt_path = "/content/drive/MyDrive/data/frame/drama1/episode1/dialogue_timestamp.json"  # 대사 json
deepface_path = "/content/drive/MyDrive/scene_analysis/deepface/drama1_episode1_deepface_result.json"
yolo_path = "/content/drive/MyDrive/scene_analysis/yolo/filtered_0.9/drama1_episode1_conf09.json"
ocr_path = "/content/drive/MyDrive/scene_analysis/ocr/drama1_episode1_easyocr_text_array.json"
output_path = "/content/drive/MyDrive/data_preprocessing/drama1.jsonl"
keyframe_path = "/content/drive/MyDrive/data/frame/drama1/episode1/keyframe/"

# 시간 문자열을 비교 가능한 형식으로 변환 함수 (시:분:초,밀리초)
def timestamp_to_millis(ts):
    # "00:00:05,964 --> 00:00:07,966"
    start, end = ts.split(" --> ")
    def to_ms(t):
        h,m,s_ms = t.split(":")
        s, ms = s_ms.split(",")
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)
    return to_ms(start), to_ms(end)

# 1) JSON 파일 로드
with open(script_path, "r", encoding="utf-8") as f:
    script_data = json.load(f)

with open(dialogue_txt_path, "r", encoding="utf-8") as f:
    dialogue_data = json.load(f)

with open(deepface_path, "r", encoding="utf-8") as f:
    deepface_data = json.load(f)

with open(yolo_path, "r", encoding="utf-8") as f:
    yolo_data = json.load(f)

with open(ocr_path, "r", encoding="utf-8") as f:
    ocr_data = json.load(f)


image_to_timestamp = {item["image"]: item["timestamp"] for item in script_data}


# 3) yolo에서 confidence 0.9 이상만 필터링 + person 객체만 별도 저장
yolo_dict = {entry["timestamp"]: entry.get("predictions", []) for entry in yolo_data}

# deepface
deepface_raw = {d["image"]: d["caption"] for d in deepface_data}

# OCR dict (timestamp → texts)
ocr_dict = {entry["timestamp"]: entry.get("texts", []) for entry in ocr_data}


# 4) dialogue 누적 대사 텍스트 생성
# 대사 리스트를 타임스탬프 순서대로 누적 합산
dialogue_data_sorted = sorted(dialogue_data, key=lambda x: timestamp_to_millis(x["timestamp"])[0])
# 타임스탬프를 정수 시작값으로 매핑
dialogue_ranges = [(timestamp_to_millis(d["timestamp"]), d["dialogue"]) for d in dialogue_data_sorted]

def get_dialogue_at_timestamp(frame_start_ms):
    for (start_ms, end_ms), text in dialogue_ranges:
        if start_ms <= frame_start_ms <= end_ms:
            return text
    return ""

# 5) 결과 조합 및 jsonl 저장
with open(output_path, "w", encoding="utf-8") as fout:
    for item in script_data:
        ts = item["timestamp"]
        img = item["image"]
        caption = item.get("caption", "")

        # YOLO 결과
        preds = yolo_dict.get(ts, [])

        # person 클래스가 있으면 deepface 결과 포함
        has_person = any(p["class"] == "person" for p in preds)

        deepface_caption = deepface_raw.get(img, "") if has_person else ""

        # 누적 대사
        frame_start_ms, _ = timestamp_to_millis(ts)
        dialogue_single = get_dialogue_at_timestamp(frame_start_ms)

        ocr_texts = ocr_dict.get(ts, [])

        # 최종 dict 구성
        output_dict = {
            "image": img,
            "timestamp": ts,
            "yolo": preds,
            "deepface": deepface_caption if deepface_caption else "",
            "ocr": ocr_texts,
            "dialogue": dialogue_single,
            "caption": caption
        }

        fout.write(json.dumps(output_dict, ensure_ascii=False) + "\n")

print(f"SFT 학습용 JSONL 파일이 생성되었습니다: {output_path}")
