# 🎥 VideoDescription

**AI-powered system that generates and refines audio descriptions from drama videos to recreate accessible content for the visually impaired.**  
**시각장애인을 위한 접근 가능한 콘텐츠 제작을 위한 AI 기반 드라마 영상 해설 자동 생성 시스템입니다.**

---

## 🧩 Table of Contents

- [Features](#-features--주요-기능)
- [Installation](#-installation--설치-방법)
- [Usage](#-usage--사용-방법)
- [Architecture](#-architecture--전체-구조)
- [Contribution](#-contribution--기여-방법)
- [Demo](#-demo)
- [License](#-license)

---

## ✨ Features / 주요 기능

- 🎬 **MP4 영상 입력**
- 🧠 **Key Frame 추출**: 영상 내 주요 장면을 자동으로 식별
- 👁️ **장면 분석**:
  - **YOLOv8**: 장면 내 객체 탐지
  - **EasyOCR**: 텍스트 인식 (간판, 자막 등)
  - **DeepFace**: 인물 감지 및 얼굴 분석
- 🗣 **Whisper**: 대사 오디오를 텍스트로 변환
- 💬 **HyperCLOVA-X**:
  - 추출된 정보 기반으로 key frame마다 3~4개의 캡션 생성
- 🧠 **GPT 기반 캡션 정제**:
  - HyperCLOVA-X가 생성한 설명 후보 중, 직접 구축한 드라마 해설 데이터셋으로 fine-tuning한 phi-2 모델을 활용하여 가장 자연스럽고 상황에 맞는 문장을 자동으로 선택
  - 유사하거나 불필요한 문장은 제거
  - TTS 음성 타이밍에 맞춰 최적화
- 🔊 **TTS**: 최종 정제된 캡션을 음성으로 변환
- 🧵 **영상 통합**: 생성된 TTS 음성을 원본 영상에 삽입하여 해설 포함 영상 출력

---

## Model 폴더

---

### 📜 Caption Generation Script (generate_caption_candidates_basic.py, generate_caption_candidates_strict.py)

이 스크립트는 HyperCLOVA X Vision 모델을 사용하여, 주어진 키프레임 이미지와 부가 정보를 기반으로 **자막 후보** 3개씩을 생성합니다.
- generate_caption_candidates_basic.py:
최소한의 프롬프트 규칙을 적용하여 비교적 자유롭게 자막 후보를 생성

- generate_caption_candidates_strict.py:
자막 작성 원칙을 엄격하게 적용하여 더 정제된 자막 후보를 생성, 현재 시제, '~다' 말투, 관찰 기반 서술 등 명시적 지침 포함, 부가 정보(대사, OCR, YOLO 객체, 얼굴 표정 등)를 활용해 더 정교한 프롬프트 구성

---

### 🧠 SFT 데이터 생성: `generate_sft_data_by_similarity.py`

이 스크립트는 생성된 자막 후보들 중에서 **정답 자막(ground truth caption)**과 가장 유사한 문장을 선택하여, **SFT 학습용 JSONL 형식**으로 변환합니다.

✅ 주요 기능
- `sentence-transformers` 기반 KoSBERT를 사용하여 cosine similarity 계산
- 3개의 후보 자막 중에서 정답 자막과 가장 유사한 것을 자동 선택
- prompt/response 형식으로 SFT용 jsonl 출력

---

### 🛠️ Fine-tuning Script: train_phi2_lora.py

train_phi2_lora.py는 phi-2 모델을 LoRA(저자원 어댑터) 기법으로 미세조정(fine-tuning)하기 위한 스크립트입니다.
본 프로젝트의 드라마 해설 데이터셋(SFT 데이터)을 사용해, 사전학습된 대형 언어 모델을 효율적으로 적응시켜 자연스러운 캡션 생성 성능을 향상시키는 역할을 합니다.

✅ 주요 기능
- Microsoft의 phi-2 사전학습 모델 불러오기
- PEFT 라이브러리를 이용한 LoRA adapter 설정 (저비용 파인튜닝)
- 여러 JSONL 포맷의 SFT 학습 데이터를 하나로 병합하여 처리
- prompt와 response를 연결한 텍스트 토크나이징
- Huggingface Trainer 기반 학습 파이프라인 구성
- 학습 중 체크포인트 저장 및 FP16 mixed precision 지원

---

### 🔧 설정
스크립트 상단에서 다음 경로를 사용자 환경에 맞게 수정해야 합니다:

```python
IMAGE_DIR = "path/to/keyframes"
JSONL_PATH = "path/to/captions.jsonl"
truth_path = "path/to/ground_truth.jsonl"  # 정답 자막이 포함된 JSONL 파일 (image, caption 필드)
candidates_path = "path/to/caption_candidates.json"  # 후보 자막들이 포함된 JSON 파일 (image, captions 필드)
output_path = "path/to/output_sft_data.jsonl"  # 생성된 SFT 학습 데이터가 저장될 경로
OUTPUT_DIR = "path/to/save/phi2_lora_adapter"  # LoRA 학습 결과 저장 디렉토리

jsonl_files = [
    "path/to/sft/drama1.jsonl",
    "path/to/sft/drama2.jsonl",
    # ... 학습에 사용할 SFT JSONL 파일들
]
```

---

## 💻 Installation / 설치 방법

> Python 3.8 이상이 필요합니다.

1. 프로젝트 클론:
```bash
git clone https://github.com/your-username/VideoDescription.git
cd VideoDescription
