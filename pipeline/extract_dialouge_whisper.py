# pip install -U openai-whisper
# pip install ffmpeg-python
import whisper
import json
import shutil
import sys
import os

from pathlib import Path


BASE_DIR = Path(__file__).parent  # pipeline 폴더 경로

# ffmpeg 경로 설정
ffmpeg_path = BASE_DIR / "ffmpeg.exe"
ffmpeg_dir = str(ffmpeg_path.parent)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# 입력 비디오 경로
video_path = Path(sys.argv[1])
video_name = video_path.stem

input_path = str(video_path)  # whisper에 str로 전달
output_path = BASE_DIR.parent / "video_script_large.json"  # 프로젝트 루트 기준 저장

# 모델 로드
model = whisper.load_model("large-v2")

# 음성 인식
result = model.transcribe(str(input_path), task="transcribe", language="ko", fp16=False)
segments = result.get("segments", [])

# 타임스탬프 포맷 함수
def format_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# JSON 구조화
output = []
for seg in segments:
    ts = f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}"
    output.append({"timestamp": ts, "dialogue": seg["text"].strip()})

# 파일로 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Transcription saved to {output_path}")
