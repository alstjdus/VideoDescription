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

## model 폴더

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

### Caption Selector using phi-2 + LoRA and SBERT: caption_selector.py
- 여러 후보 자막 후보(candidates) 중에서 Microsoft phi-2 모델에 LoRA adapter를 적용한 모델로 가장 적절한 자막을 선택합니다.
- 선택된 자막에 대해 SBERT 임베딩 기반 의미 중복 제거를 수행하여 최종 자막을 추출합니다.

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
CANDIDATE_PATH = "path/to/candidate.json"  # 사용자가 직접 설정할 경로
OUTPUT_PATH = "path/to/final_caption.json"  # 결과 저장할 경로
jsonl_files = [
    "path/to/sft/drama1.jsonl",
    "path/to/sft/drama2.jsonl",
    # ... 학습에 사용할 SFT JSONL 파일들
]
```

---

## pipeline 폴더

---

### extract_dialogue_whisper.py
Whisper 모델을 사용하여 동영상 파일에서 대사를 추출하고, 이를 JSON 형식으로 저장합니다.

- 사용 모델: `openai/whisper-large-v2`
- 입력: mp4 형식의 동영상 파일 (`testvideo.mp4`)
- 출력: `video_script_large.json` (자막 JSON 파일)
- 주요 구성:
  - ffmpeg를 통해 오디오 추출
  - Whisper로 대사 인식
  - `timestamp`, `dialogue` 구조의 JSON으로 저장

---

### 🧠 process_frame_ocr_deepface.py
영상에서 추출된 키프레임 이미지에 대해 문자인식(OCR)과 얼굴 감정 분석을 동시에 수행하는 모듈입니다.

- 입력:
  - 키프레임 이미지 폴더 (`./keyframes_pyscenedetect/testvideo/`)
  - 타임스탬프 정보가 포함된 JSON (`keyframes_timestamp.json`)
- 출력:
  - `deepface_ocr.json`: 각 이미지별 OCR 텍스트 및 감정 분석 결과 포함

- 주요 라이브러리:
  - EasyOCR (텍스트 인식)
  - DeepFace (감정 인식)

---

### 🧠 YOLO 기반 객체 인식: `run_yolo_on_video.py`
이 스크립트는 YOLOv8(nano 모델)을 활용하여 동영상의 각 구간(자막 기반 타임스탬프)에 등장하는 객체를 탐지합니다. 결과는 기존 자막 JSON (`deepface_ocr.json`)에 `"yolo"` 필드로 추가됩니다.

- 입력:
  - `testvideo.mp4`: 객체 인식을 수행할 영상 파일
  - `deepface_ocr.json`: 기존 자막/딥페이스/OCR 등의 정보가 포함된 JSON
- 출력:
  - `yolo_results.json`: YOLO 객체 인식 결과가 포함된 JSON

- 주요 사용 라이브러리:
  - `ultralytics` (YOLOv8)

---

### 🎞️ Scene Detection 및 Keyframe 추출: `keyframe_extractor.py`
이 스크립트는 영상 내 장면 전환(Scene Change)을 감지하고, 각 장면의 시작 시점의 프레임을 이미지로 저장합니다. 또한 각 키프레임의 타임스탬프 범위를 포함하는 메타데이터를 JSON 파일로 저장합니다.

- 입력:
  - `testvideo.mp4`: 분석할 영상 파일
- 출력:
  - 키프레임 이미지: `keyframes_pyscenedetect/testvideo/frame_0000.png` 등
  - 타임스탬프 JSON: `keyframes_pyscenedetect/testvideo/keyframes_timestamp.json`

- 주요 사용 라이브러리:
  - `PySceneDetect`

---

### 🔗 YOLO + 자막 병합 스크립트: `generate_final_input_json.py`
이 스크립트는 YOLOv8 객체 인식 결과(`yolo_results.json`)와 Whisper 음성 자막 결과(`video_script_large.json`)를 타임스탬프 기준으로 병합합니다. 결과로 생성되는 `final_input.json`은 각 객체 탐지 시간대에 대응하는 자막(대사)을 포함하며, 멀티모달 비디오 설명 시스템의 입력으로 사용됩니다.

- 입력:
  - `yolo_results.json`: YOLO 객체 탐지 결과
  - `video_script_large.json`: Whisper로부터 추출된 자막 JSON
- 출력:
  - `final_input.json`: 자막이 병합된 최종 JSON

---

### Video Description Pipeline: 'video_description_with_tts.py'
이 파이프라인은 영상 파일과 자막, 대사 데이터를 병합하여 Melo TTS를 사용해 해설 음성을 생성하고, 원본 영상에 해설 음성을 합성하여 최종 영상을 출력합니다.

---

## 💻 Installation / 설치 방법

> Python 3.8 이상이 필요합니다.

1. 프로젝트 클론:
```bash
git clone https://github.com/your-username/VideoDescription.git
cd VideoDescription
