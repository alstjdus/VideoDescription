import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from pathlib import Path

# === 설정 ===
BASE_DIR = Path(__file__).parent.parent

CANDIDATE_PATH = BASE_DIR / 'caption_candidates.json'
OUTPUT_PATH = BASE_DIR / 'final_caption.json'
MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "michiboke/phi2-lora-caption-selector"
SIM_THRESHOLD = 0.87  # 중복 판단 기준

# === 모델 로딩 (phi-2 + LoRA) ===
print("🔧 Loading phi-2 + LoRA model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model.eval()

# === SBERT 로딩 ===
print("🔧 Loading SBERT model...")
sbert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# === 데이터 로딩 ===
with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

selected_captions = []

# === 1단계: phi-2로 자막 선택 ===
print("🎯 Selecting best captions using phi-2...")
for entry in tqdm(data):
    prompt = "장면 정보:\n"
    prompt += f"- 대사: {entry.get('dialogue', '') or '없음'}\n"
    prompt += f"- OCR: {' '.join(entry.get('ocr', [])) or '없음'}\n"
    yolo_objects = ', '.join([obj.get("class", "") for obj in entry.get("yolo", []) if isinstance(obj, dict)]) or "없음"
    prompt += f"- 객체: {yolo_objects}\n"
    prompt += f"- 얼굴 정보: {entry.get('deepface', '') or '없음'}\n\n"

    for i, caption in enumerate(entry["captions"], 1):
        prompt += f"{i}. {caption}\n"
    prompt += "\n가장 적절한 자막 번호는?"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    number_str = ''.join(filter(str.isdigit, response))

    try:
        best_idx = int(number_str) - 1
        best_caption = entry["captions"][best_idx]
    except:
        best_caption = entry["captions"][0]  # fallback

    selected_captions.append({
        "image": entry["image"],
        "timestamp": entry.get("timestamp", ""),
        "caption": best_caption.strip()
    })

# === 2단계: 중복 제거 (SBERT 기반) ===
print("🧹 Removing duplicate captions (semantic similarity)...")
deduplicated = []
prev_embedding = None

for item in selected_captions:
    emb = sbert.encode(item["caption"], convert_to_tensor=True)
    if prev_embedding is not None:
        sim = util.cos_sim(prev_embedding, emb).item()
        if sim >= SIM_THRESHOLD:
            print(f"⚠️ 중복 제거됨: {item['caption']} (유사도: {sim:.3f})")
            continue
    deduplicated.append(item)
    prev_embedding = emb

# === 결과 저장 ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(deduplicated, f, ensure_ascii=False, indent=2)

print(f"✅ 자막 선택 + 중복 제거 완료! → {OUTPUT_PATH}")
