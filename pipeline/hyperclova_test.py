import os
import json
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import torch
from tqdm import tqdm

# 모델 및 디바이스 설정
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델, 프로세서, 토크나이저 불러오기
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 프레임 디렉토리 및 메타데이터 경로
frame_dir = "./keyframes_pyscenedetect/testvideo"
metadata_all_path = "final_input.json"
output_json_path = "captions.json"
results = {}

# 메타데이터 전체 로드 및 딕셔너리화 (image filename → metadata)
with open(metadata_all_path, "r", encoding="utf-8") as f:
    all_metadata = json.load(f)
metadata_dict = {meta["image"]: meta for meta in all_metadata}

# 영상 해설 가이드라인
guideline = (
    "당신은 영상 해설용 자막 생성기입니다. 아래 원칙을 따르세요: "
    "해설은 간결하고 플롯 중심으로 단정적 서술어로 표현합니다. 중요하지 않은 시각 요소는 생략합니다. "
    "장면의 시각적 특징을 사실적으로 설명하고, 인물의 표정과 행동은 구체적으로 묘사합니다. "
    "장소, 날씨, 배경 설명은 플롯에 중요할 때만 포함하며, 해설은 현재 시제와 평서형 어체로 작성합니다."
)

# 프레임 이미지들 반복 처리
for filename in tqdm(sorted(os.listdir(frame_dir))):
    if not filename.endswith(".png"):
        continue

    frame_path = os.path.join(frame_dir, filename)
    frame_id = os.path.splitext(filename)[0]

    # 해당 프레임의 메타데이터 가져오기
    metadata = metadata_dict.get(filename)
    if metadata is None:
        print(f"메타데이터 없음: {filename}")
        continue

    # 프롬프트 구성
    metadata_text = (
        f"{metadata.get('timestamp', '없음')}\n"
        f"장면 내 인물의 감정: {metadata.get('deepface', '없음')}\n"
        f"장면의 대사: {' '.join(metadata.get('dialogue', [])) or '없음'}\n"
        f"감지된 객체: {', '.join([obj['class'] for obj in metadata.get('yolo', [])]) or '없음'}\n"
        f"장면 내 텍스트 인식: {', '.join(metadata.get('ocr', [])) or '없음'}"
    )

    user_text = (
        "이 이미지에 대한 영상 해설 자막을 30자 이내로 생성해 주세요. "
        "다음은 해당 프레임의 메타데이터입니다:\n" + metadata_text
    )

    # VLM 입력 구성
    vlm_chat = [
        {"role": "system", "content": {"type": "text", "text": guideline}},
        {"role": "user", "content": {"type": "text", "text": user_text}},
        {
            "role": "user",
            "content": {
                "type": "image",
                "image": frame_path
            }
        }
    ]

    # 전처리 및 토크나이징
    try:
        new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
        preprocessed = preprocessor(all_images, is_video_list=is_video_list)
        input_ids = tokenizer.apply_chat_template(
            new_vlm_chat,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True
        ).to(device)

        # 모델 추론
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.6,
            temperature=0.7,
            repetition_penalty=1.0,
            **preprocessed
        )

        # 결과 디코딩
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 결과 저장
        results[frame_id] = {
            "caption": generated_text,
            "timestamp": metadata.get("timestamp", "")
        }

    except Exception as e:
        print(f"{frame_id} 처리 실패: {e}")
        continue

# 전체 결과 저장
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n 결과 저장 위치: {output_json_path}")
