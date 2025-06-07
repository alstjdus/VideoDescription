import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# === 경로 설정 ===
truth_path = "path/to/ground_truth.jsonl"        # 정답 자막이 포함된 JSONL 파일 (image, caption 필드)
candidates_path = "path/to/caption_candidates.json"  # 자막 후보 리스트가 들어 있는 JSON 파일 (image, captions 필드)
output_path = "path/to/output_sft_data.jsonl"    # 생성될 SFT 학습용 jsonl 출력 경로

# === 모델 로드 (Ko 모델 or 다국어 BERT)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# === 정답 캡션 로드 (이미지 이름 기준)
truth_map = {}
with open(truth_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        truth_map[entry["image"]] = entry["caption"]

# === 후보 캡션 데이터 로드
with open(candidates_path, "r", encoding="utf-8") as f:
    candidate_data = json.load(f)

# === 유사도 기반 SFT jsonl 생성
with open(output_path, "w", encoding="utf-8") as out_f:
    for entry in tqdm(candidate_data):
        image = entry["image"]
        candidates = entry["captions"]
        gt_caption = truth_map.get(image, None)
        if not gt_caption or not candidates:
            continue  # 정답 없으면 스킵

        # 유사도 계산
        all_texts = [gt_caption] + candidates
        embeddings = model.encode(all_texts, convert_to_tensor=True)
        sim_scores = util.cos_sim(embeddings[0], embeddings[1:])[0]
        best_idx = int(sim_scores.argmax().item())  # 0,1,2 → 1~3으로 변환

        # 메타 정보 텍스트화
        ocr = ", ".join(entry.get("ocr", [])) or "없음"
        dialogue = entry.get("dialogue", "") or "없음"
        yolo = ", ".join(obj.get("label", "") for obj in entry.get("yolo", [])) or "없음"
        deepface = entry.get("deepface", "") or "없음"

        prompt = (
            f"장면 정보:\n"
            f"- 대사: {dialogue}\n"
            f"- OCR: {ocr}\n"
            f"- YOLO 감지: {yolo}\n"
            f"- 얼굴 정보: {deepface}\n\n"
            f"후보 자막:\n" +
            "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(candidates)]) +
            "\n\n가장 적절한 자막 번호는?"
        )

        response = str(best_idx + 1)  # 1~3

        json.dump({"prompt": prompt, "response": response}, out_f, ensure_ascii=False)
        out_f.write("\n")

print(f"✅ 학습용 SFT 데이터 생성 완료 → {output_path}")
