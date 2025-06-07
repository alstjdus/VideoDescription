import os
import json
import re
import cv2
from deepface import DeepFace

image_folder = "/content/drive/MyDrive/캡디/Data/frame/drama1/키프레임/"

# 정규표현식으로 'frame_숫자.png'만 필터링
image_files = [
    f for f in os.listdir(image_folder)
    if re.match(r'^frame_\d+\.png$', f)
]

results_list = []

for image_file in sorted(image_files):
    image_path = os.path.join(image_folder, image_file)
    print(f"처리 중: {image_path}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("이미지를 불러올 수 없습니다.")

        result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        print(f"결과: {result}")
        dominant_emotion = result[0]['dominant_emotion']
        print(f"{image_file} → 감정: {dominant_emotion}")
    except Exception as e:
        dominant_emotion = ""
        print(f"{image_file} → 분석 실패: {e}")

    results_list.append({
        "image": image_file,
        "caption": dominant_emotion
    })

# 결과 저장
output_path = "/content/drive/MyDrive/캡디/장면분석/deepface/수리남_deepface_result.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_list, f, ensure_ascii=False, indent=2)

print(f"분석 결과가 '{output_path}' 파일로 저장되었습니다.")
