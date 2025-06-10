from ultralytics import YOLO
import json
import cv2

from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent  # 프로젝트 루트

video_path = Path(sys.argv[1])  # 전달받은 비디오 경로 (ex. uploads/testvideo-uuid1234.mp4)

input_json_path = BASE_DIR / "deepface_ocr.json"
output_json_path = BASE_DIR / "yolo_results.json"

#input_path_no_bbox  = save_path
#output_path_no_bbox = ./results_no_bbox.json"
#input_path          = output_path_no_bbox
#output_path         = "./results_conf09.json"

# print(ultralytics.__version__)
# print(torch.__version__)
# print(cv2.__version__)

# YOLO 모델 로드
model = YOLO("yolov8n.pt")

# 타임스탬프 파싱 함수 ("HH:MM:SS,mmm" → 초 float 변환)
def parse_time(ts: str) -> float:
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

# JSON 로드
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 비디오 로드
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

# 각 항목에 대해 YOLO 분석 수행
for entry in data:
    # 타임스탬프 파싱
    start_ts, end_ts = entry["timestamp"].split(" --> ")
    start_sec = parse_time(start_ts)
    end_sec   = parse_time(end_ts)

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps)

    all_classes = set()

    for fi in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        result = model.predict(frame, imgsz=640, conf=0.9)[0]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            all_classes.add(cls_name)

    # 결과 저장
    entry["yolo"] = [{"class": c} for c in sorted(all_classes)]

# 결과 저장
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"YOLO 결과 저장 완료 → {output_json_path}")
