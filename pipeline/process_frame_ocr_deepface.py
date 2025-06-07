import os
import json
import re
import cv2
import easyocr
from deepface import DeepFace

# ---------------- 설정 ----------------
image_folder = "./keyframes_pyscenedetect/testvideo/"
timestamp_json_path = "./keyframes_pyscenedetect/testvideo/keyframes_timestamp.json"
output_path = "deepface_ocr.json"

# OCR 설정
UPSCALE_FACTOR = 2
CONFIDENCE_THRESHOLD = 0.30

# EasyOCR 초기화
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# ---------------- 타임스탬프 불러오기 ----------------
with open(timestamp_json_path, "r", encoding="utf-8") as f:
    timestamp_data = json.load(f)
timestamp_map = {entry["frame_file"]: entry["timestamp"] for entry in timestamp_data}

# ---------------- 이미지 파일 목록 정렬 ----------------
image_files = [
    f for f in os.listdir(image_folder)
    if re.match(r'^frame_\d{4}\.png$', f)
]
image_files.sort()

# ---------------- 결과 저장 리스트 ----------------
combined_results = []

# ---------------- 처리 루프 ----------------
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    timestamp = timestamp_map.get(image_file, "00:00:00,000 --> 00:00:01,000")
    texts = []
    emotion = ""

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 불러올 수 없습니다.")

        # ----------- OCR -----------
        resized_img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR))
        ocr_result = reader.readtext(resized_img, detail=1)
        texts = [text for _, text, conf in ocr_result if conf >= CONFIDENCE_THRESHOLD]

        # ----------- DeepFace 감정 분석 -----------
        df_result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        emotion = df_result[0]['dominant_emotion']

    except Exception as e:
        print(f"[오류] {image_file} 처리 중 오류 발생: {e}")

    # ----------- 결과 통합 저장 -----------
    combined_results.append({
        "image": image_file,
        "timestamp": timestamp,
        "deepface": emotion,
        "ocr": texts,
    })

# ---------------- JSON 저장 ----------------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_results, f, ensure_ascii=False, indent=2)

print(f"\n 모든 분석 결과가 저장되었습니다: {output_path}")
