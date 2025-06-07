# pip install -U openai-whisper
# pip install ffmpeg-python
import whisper
import json
import shutil
import sys
import os

# ffmpeg 설치 확인
# if shutil.which("ffmpeg") is None:
#     print("ffmpeg가 설치되어 있지 않습니다. https://ffmpeg.org/download.html 에서 설치 후, 환경 변수 PATH에 추가하세요.")
#     sys.exit(1)

# 경로 설정
input_path  = "testvideo.mp4"
output_path = "video_script_large.json"

# 프로젝트 내 ffmpeg.exe 경로
ffmpeg_path = os.path.abspath("ffmpeg.exe")
ffmpeg_dir = os.path.dirname(ffmpeg_path)

# 현재 PATH에 ffmpeg 디렉토리 추가 (임시)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

# 모델 로드
model = whisper.load_model("large-v2")

# 음성 인식
result = model.transcribe(input_path, task="transcribe", language="ko", fp16=False)
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
