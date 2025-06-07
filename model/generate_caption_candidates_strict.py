import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# === 설정 ===
# 사용자가 직접 설정해야 할 경로
IMAGE_DIR = "path/to/keyframes"  # 키프레임 이미지가 저장된 디렉토리
JSONL_PATH = "path/to/captions.jsonl"  # OCR/YOLO/DeepFace 정보가 담긴 jsonl 파일

# 사용할 HyperCLOVA X 모델 이름
MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CANDIDATES = 3  # 후보 자막 수

# === 프롬프트 구성 ===
# === 자막 생성 가이드라인 및 사용자 요청 ===
guideline = (
    "당신은 영상 해설 자막 생성기입니다. 다음 원칙을 따르세요:\n"
    "- 자막은 하나의 문장으로 작성합니다 (30자 이내).\n"
    "- 문장은 현재 시제, 평서형, 정중하지 않은 단정형(~다) 말투로 마무리합니다.\n"
    "- 인물의 행동, 표정, 주변 상황을 간결하게 묘사합니다.\n"
    "- 설명은 감정이 아닌 관찰 위주로 작성합니다.\n"
    "- 의미 없는 배경 정보나 부사, 수식어는 생략합니다.\n"
    "- 대화나 텍스트가 명확하면 요약해 포함할 수 있습니다."
)

user_request = "이미지에 대한 자막을 1문장, 30자 이내로 생성해 주세요. '~다' 형태의 단정한 말투를 사용하세요."

# === 모델 로드 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
model.eval()

# === JSONL 부가 정보 불러오기 ===
metadata = {}
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        metadata[entry["image"]] = entry

results = []

# === 이미지별 캡션 생성 ===
for filename in tqdm(sorted(os.listdir(IMAGE_DIR))):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    meta = metadata.get(filename, {})

    # === JSONL 정보 자연어로 정리 ===
    ocr = " ".join(meta.get("ocr", [])) or "없음"
    dialogue = meta.get("dialogue", "") or "없음"
    yolo_data = meta.get("yolo", [])
    yolo = ", ".join(obj.get("label", "") for obj in yolo_data if isinstance(obj, dict)) or "없음"
    deepface = meta.get("deepface", "") or "없음"

    context_text = (
        f"[장면 추가 정보]\n"
        f"- 대사: {dialogue}\n"
        f"- 인식된 글자(OCR): {ocr}\n"
        f"- 감지된 객체(YOLO): {yolo}\n"
        f"- 얼굴 정보(DeepFace): {deepface}\n"
    )

    full_prompt = f"{guideline}\n{context_text}\n{user_request}"

    # === ChatML 형식 입력 구성 ===
    chat = [
        {"role": "system", "content": {"type": "text", "text": "너는 영상 해설 자막 생성기야."}},
        {"role": "user", "content": {"type": "text", "text": full_prompt}},
        {"role": "user", "content": {
            "type": "image",
            "image": image_path,
            "filename": filename
        }},
    ]

    # === 이미지 불러오기 및 전처리 ===
    chat, all_images, is_video_list = processor.load_images_videos(chat)
    image_inputs = processor(all_images, is_video_list=is_video_list)
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt", tokenize=True, add_generation_prompt=True)

    # === 자막 후보 N개 생성 ===
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids.to(device),
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.1,
            num_return_sequences=NUM_CANDIDATES,
            **image_inputs
        )

    captions = tokenizer.batch_decode(output, skip_special_tokens=True)
    captions = [c.strip() for c in captions]

    results.append({
        "image": filename,
        "captions": captions,
        "timestamp": meta.get("timestamp", ""),
        "ocr": meta.get("ocr", []),
        "dialogue": meta.get("dialogue", ""),
        "yolo": meta.get("yolo", []),
        "deepface": meta.get("deepface", "")
    })

# === 결과 저장 ===
with open("/home/compu/jyw/capstone/test/후보2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ 자막 후보 3개씩 생성 완료! -> caption_candidates.json")
