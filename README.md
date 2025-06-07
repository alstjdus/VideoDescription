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
## 📜 Caption Generation Script (generate_caption_candidates_basic.py, generate_caption_candidates_strict.py)

이 스크립트는 HyperCLOVA X Vision 모델을 사용하여, 주어진 키프레임 이미지와 부가 정보를 기반으로 **30자 이내 단정형 자막 후보** 3개씩을 생성합니다.

---

### 🔧 설정
스크립트 상단에서 다음 경로를 사용자 환경에 맞게 수정해야 합니다:

```python
IMAGE_DIR = "path/to/keyframes"
JSONL_PATH = "path/to/captions.jsonl"


---

## 💻 Installation / 설치 방법

> Python 3.8 이상이 필요합니다.

1. 프로젝트 클론:
```bash
git clone https://github.com/your-username/VideoDescription.git
cd VideoDescription
