#pip install ultralytics yt-dlp opencv-python torch numpy

import ultralytics
from ultralytics import YOLO

import torch
import json
from datetime import datetime
import cv2
import numpy as np

video_path          = r"C:\Users\shp10\Downloads\drama1_episode1.mp4"
script_path         = r"C:\Users\shp10\Downloads\drama1_episode1_script.json"
save_path           = r"C:\Users\shp10\Downloads\drama1_results.json"
input_path_no_bbox  = save_path
output_path_no_bbox = r"C:\Users\shp10\Downloads\drama1_results_no_bbox.json"
input_path          = output_path_no_bbox
output_path         = r"C:\Users\shp10\Downloads\drama1_results_conf09.json"

print(ultralytics.__version__)
print(torch.__version__)
print(cv2.__version__)

# YOLOv8 모델 불러오기 (nano 모델은 가장 가볍고 빠름)
model = YOLO("yolov8n.pt")

# 분석할 비디오 경로 (Drive에 마운트된 경로)
print(f"Using local video → {video_path}")

def parse_time(ts: str) -> float:
    # "HH:MM:SS,mmm" → seconds
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

with open(script_path, "r", encoding="utf-8") as f:
    script = json.load(f)

timestamps = []
for entry in script:
    start_ts, end_ts = entry["timestamp"].split(" --> ")
    timestamps.append({
        "timestamp": entry["timestamp"],
        "start": parse_time(start_ts),
        "end":   parse_time(end_ts),
        "caption": entry["caption"]
    })

with open("timestamps.json", "w", encoding="utf-8") as f:
    json.dump(timestamps, f, ensure_ascii=False, indent=2)

print(f"생성된 window 개수: {len(timestamps)}")



# 비디오 열기 + FPS
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# 타임스탬프 로드
with open("timestamps.json", "r", encoding="utf-8") as f:
    windows = json.load(f)

# 전역 캐시: {frame_index: [ {class, confidence, bbox}, ... ] }
frame_cache = {}

out = []

for win in windows:
    # 1) 프레임 인덱스 계산, 최소 1 프레임 보장
    start_f = int(win["start"] * fps)
    end_f   = int(win["end"]   * fps)
    if end_f < start_f:
        end_f = start_f

    all_preds = []

    # 2) 윈도우 내 프레임 순회
    for fi in range(start_f, end_f + 1):
        # 이미 처리해 뒀으면 캐시에서 꺼내고
        if fi in frame_cache:
            preds = frame_cache[fi]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            res = model.predict(frame, imgsz=640, conf=0.25)[0]
            preds = []
            for x1, y1, x2, y2, conf, cls in res.boxes.data.cpu().numpy().tolist():
                preds.append({
                    "class":      res.names[int(cls)],
                    "confidence": float(conf),
                    "bbox":       [float(x1), float(y1), float(x2), float(y2)]
                })
            # 캐시에 저장
            frame_cache[fi] = preds

        # 이 프레임에서 검출된 모든 박스 누적
        all_preds.extend(preds)

    # 3) 검출이 하나도 없으면 빈 predictions
    if not all_preds:
        out.append({
            "timestamp":   win["timestamp"],
            "caption":     win["caption"],
            "predictions": []
        })
        continue

    # 4) NMS 입력 형식으로 변환
    boxes, scores = [], []
    for p in all_preds:
        x1, y1, x2, y2 = p["bbox"]
        boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h]
        scores.append(p["confidence"])

    # 5) OpenCV NMS 수행
    indices = cv2.dnn.NMSBoxes(
        boxes, scores,
        score_threshold=0.25,
        nms_threshold=0.45
    )

    # 6) 인덱스 정리 후 최종 preds 구성
    keep = set()
    if len(indices) > 0:
        for idx in indices:
            # idx가 [i], (i,), 혹은 scalar int 경우 모두 대응
            if isinstance(idx, (list, tuple, np.ndarray)):
                keep.add(int(idx[0]))
            else:
                keep.add(int(idx))

    final_preds = [all_preds[i] for i in keep]

    out.append({
        "timestamp":   win["timestamp"],
        "caption":     win["caption"],
        "predictions": final_preds
    })

# 7) 결과 저장
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {len(out)} 윈도우 결과 → {save_path}")



with open(input_path_no_bbox, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2) bbox 키 제거
for entry in data:
    # predictions 리스트가 비어있지 않다면
    if "predictions" in entry and entry["predictions"]:
        entry["predictions"] = [
            {k: v for k, v in pred.items() if k != "bbox"}
            for pred in entry["predictions"]
        ]

# 3) 결과 저장
with open(output_path_no_bbox, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {output_path_no_bbox}")



# 1) JSON 불러오기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2) confidence > 0.9 필터링 & 빈 윈도우 제거, bbox도 제거
filtered = []
for entry in data:
    # threshold 이상인 예측만 골라내기
    preds = [p for p in entry.get("predictions", []) if p.get("confidence", 0) > 0.9]
    if not preds:
        continue

    # bbox, confidence 모두 제거하고 class만 남기기
    clean_preds = [{"class": p["class"]} for p in preds]

    filtered.append({
        "timestamp":   entry["timestamp"],
        "caption":     entry["caption"],
        "predictions": clean_preds
    })

# 3) 결과 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {len(filtered)} 윈도우 → {output_path}")
