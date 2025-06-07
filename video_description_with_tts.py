# moviepy 이 버전으로 install 해야함: pip install moviepy==1.0.3
import json
from datetime import datetime

import os
import json
import tempfile

from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment
import imageio_ffmpeg as ffmpeg_builder
from melo.api import TTS


# 자막 타임스탬프 문자열을 datetime 객체로 변환하는 함수
def parse_timestamp_range(ts_range):
    start_str, end_str = ts_range.split(" --> ")
    fmt = "%H:%M:%S,%f"
    start = datetime.strptime(start_str.strip(), fmt)
    end = datetime.strptime(end_str.strip(), fmt)
    return start, end

# 파일 불러오기
with open("video_script_large.json", "r", encoding="utf-8") as f:
    dialogues = json.load(f)

with open("captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

# 캡션 타임스탬프 기준 정렬
caption_list = []
for key, value in captions.items():
    start, end = parse_timestamp_range(value["timestamp"])
    caption_list.append({
        "timestamp": value["timestamp"],
        "caption": value["caption"],
        "start": start,
        "end": end
    })

# 각 캡션 타임스탬프에 해당하는 dialogue 찾아서 병합
merged = []
for caption in caption_list:
    matched_dialogue = ""
    for dlg in dialogues:
        dlg_start, dlg_end = parse_timestamp_range(dlg["timestamp"])
        # 대사의 시작 시각이 해설 캡션 구간 안에 포함되면 매치
        if caption["start"] <= dlg_start < caption["end"]:
            matched_dialogue = dlg["dialogue"]
            break
    merged.append({
        "timestamp": caption["timestamp"],
        "caption": caption["caption"],
        "dialogue": matched_dialogue
    })

# 결과 저장
with open("captions_dialogue.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("병합 완료: captions_dialogue.json")



# ffmpeg 경로 설정
AudioSegment.converter = ffmpeg_builder.get_ffmpeg_exe()

def time_to_seconds(time_str):
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + float(s + '.' + ms)

def generate_tts_audio(text, speed=1.2):
    """
    Melo TTS로 텍스트 음성을 생성하고 mp3 파일 경로를 반환
    """
    # 1) temp 파일 경로 설정
    tmp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    tmp_wav_file.close()
    tmp_mp3_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp_mp3_file.close()

    # 2) 모델 로드 및 설정
    model = TTS(language="KR", device="auto")
    speaker_id = model.hps.data.spk2id["KR"]

    # 3) TTS 수행 (wav 생성)
    model.tts_to_file(text, speaker_id, tmp_wav_file.name, speed=speed)

    # 4) wav → mp3 변환
    sound = AudioSegment.from_wav(tmp_wav_file.name)
    sound.export(tmp_mp3_file.name, format="mp3", bitrate="192k")

    # 5) wav 임시 파일 삭제
    try:
        os.remove(tmp_wav_file.name)
    except:
        pass

    return tmp_mp3_file.name

def main():
    json_path = "captions_dialogue.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_path = "testvideo.mp4"
    video = VideoFileClip(video_path)
    original_audio = video.audio

    tts_audio_clips = []
    processed_start_times = set()

    min_interval = 3.0
    last_tts_time = -min_interval

    for entry in data:
        dialogue = entry.get("dialogue", "").strip()
        caption = entry.get("caption", "").strip()
        timestamp = entry.get("timestamp", "").strip()

        if dialogue == "" and caption != "":
            start_str, _ = [t.strip() for t in timestamp.split("-->")]
            start_sec = time_to_seconds(start_str)

            if start_sec - last_tts_time < min_interval:
                continue
            if start_sec in processed_start_times:
                continue
            processed_start_times.add(start_sec)

            # Melo TTS로 mp3 생성
            tts_mp3_path = generate_tts_audio(caption, speed=1.5)
            tts_clip = AudioFileClip(tts_mp3_path).set_start(start_sec)
            tts_audio_clips.append(tts_clip)
            last_tts_time = start_sec

    all_audio_clips = [original_audio] + tts_audio_clips
    composite_audio = CompositeAudioClip(all_audio_clips).set_duration(video.duration)

    final_video = video.set_audio(composite_audio)

    output_path = "melotts_output.mp4"
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        threads=4,
        verbose=True,
        logger=None
    )

    # 불필요한 mp3 파일 정리
    for tmp in os.listdir(tempfile.gettempdir()):
        if tmp.endswith(".mp3") and tmp.startswith("tmp"):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), tmp))
            except:
                pass

    print(f"완료: '{output_path}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
