import cv2
import json
from pathlib import Path
from scenedetect import detect, ContentDetector
import sys

# 사용 예시
LOCAL_VIDEO_PATH = Path(sys.argv[1])
BASE_OUTPUT_DIR = Path('./keyframes_pyscenedetect') / LOCAL_VIDEO_PATH.stem

def format_timecode(timecode):
    return str(timecode).replace('.', ',')

def detect_scenes_and_save_keyframes_with_metadata(video_path: str, output_dir: Path, threshold=30.0):
    output_dir.mkdir(parents=True, exist_ok=True)  # 확실히 디렉터리 생성

    scene_list = detect(
        video_path=str(video_path),  # str로 변환 필요
        detector=ContentDetector(threshold=threshold)
    )

    print(f"🎬 {video_path.name}: {len(scene_list)}개 장면 감지")

    cap = cv2.VideoCapture(str(video_path))
    metadata = []

    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"frame_{i:04d}.png"
            out_path = output_dir / frame_filename
            cv2.imwrite(str(out_path), frame)

            timestamp_range = f"{format_timecode(start_time)} --> {format_timecode(end_time)}"
            metadata.append({
                "frame_file": frame_filename,
                "timestamp": timestamp_range
            })

    cap.release()

    json_path = output_dir / "keyframes_timestamp.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Keyframe 및 타임스탬프 JSON 저장 완료: {json_path}\n")


video_name = LOCAL_VIDEO_PATH.stem
output_dir = BASE_OUTPUT_DIR

detect_scenes_and_save_keyframes_with_metadata(LOCAL_VIDEO_PATH, output_dir)
