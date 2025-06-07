import os
import cv2
import json
import re
import easyocr

# EasyOCR 초기화
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# 경로 설정
base_path = "/content/drive/MyDrive/캡디/Data/frame/drama1/episode1"
image_folder = os.path.join(base_path, "keyframes")  # 키프레임 폴더
timestamp_json_path = os.path.join(base_path, "script", "episode1.json")  # 대본 json 경로

output_base = "/content/drive/MyDrive/캡디/scene_analysis/OCR"
text_only_path = os.path.join(output_base, "drama1_episode1_easyocr_text_array.json")
with_bbox_path = os.path.join(output_base, "drama1_episode1_easyocr_with_bbox.json")

# OCR 설정값
UPSCALE_FACTOR = 2
CONFIDENCE_THRESHOLD = 0.30

# 타임스탬프 불러오기
with open(timestamp_json_path, "r", encoding="utf-8") as f:
    timestamp_data = json.load(f)
timestamp_map = {entry["image"]: entry["timestamp"] for entry in timestamp_data}

# 이미지 파일 목록 필터링 및 정렬
image_files = [
    f for f in os.listdir(image_folder)
    if re.match(r'^frame_\d+\.png$', f)
]
image_files.sort()

# 결과 저장 리스트
results_text_only = []
results_with_bbox = []

# OCR 수행
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"\n[처리 중] {image_file} → {image_path}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 불러올 수 없습니다.")

        # 업스케일
        img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR))

        # OCR 실행
        ocr_result = reader.readtext(img, detail=1)

        # 콘솔 출력: 전체 출력
        print(f"[OCR 결과] {image_file}")
        for bbox, text, conf in ocr_result:
            print(f"  - '{text}' (정확도: {conf:.2f})")

        # 타임스탬프 매핑
        timestamp = timestamp_map.get(image_file, "00:00:00,000 --> 00:00:01,000")

        # 정확도 필터링
        filtered = [
            (bbox, text, conf) for bbox, text, conf in ocr_result
            if conf >= CONFIDENCE_THRESHOLD
        ]

        # 텍스트 전용 저장
        results_text_only.append({
            "timestamp": timestamp,
            "texts": [text for _, text, _ in filtered]
        })

        # bbox 포함 저장 (텍스트별 timestamp 포함)
        results_with_bbox.append({
            "timestamp": timestamp,
            "texts": [
                {
                    "text": text,
                    "bbox": [[int(x), int(y)] for x, y in bbox],
                    "timestamp": timestamp
                }
                for bbox, text, _ in filtered
            ]
        })

    except Exception as e:
        print(f"[오류] {image_file} → OCR 실패: {e}")
        timestamp = timestamp_map.get(image_file, "00:00:00,000 --> 00:00:01,000")

        results_text_only.append({
            "timestamp": timestamp,
            "texts": []
        })
        results_with_bbox.append({
            "timestamp": timestamp,
            "texts": []
        })

# JSON 저장
with open(text_only_path, "w", encoding="utf-8") as f:
    json.dump(results_text_only, f, ensure_ascii=False, indent=2)

with open(with_bbox_path, "w", encoding="utf-8") as f:
    json.dump(results_with_bbox, f, ensure_ascii=False, indent=2)

print(f"\n✅ OCR 결과 저장 완료:\n- 텍스트 배열: {text_only_path}\n- 위치 포함: {with_bbox_path}")
