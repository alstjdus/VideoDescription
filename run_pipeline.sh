#!/bin/bash

echo "[START] Input video path: $1"
echo "[START] Output file name: $2"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./run_pipeline.sh [video file path] [output file name]"
    exit 1
fi

VIDEO="$1"
OUTNAME="$2"

# STEP 1: Analysis
echo "[STEP 1] Creating and setting up analysis environment..."
python3.10 -m venv venv_analysis
source venv_analysis/bin/activate
pip install --upgrade pip
pip install -r requirements/analysis.txt

echo "[STEP 1] Running analysis scripts..."
python pipeline/keyframe.py "$VIDEO"
python pipeline/process_frame_ocr_deepface.py "$VIDEO"
python pipeline/process_frame_yolo.py "$VIDEO"
pip install -U openai-whisper
python pipeline/extract_dialouge_whisper.py "$VIDEO"
python pipeline/json_merged.py
deactivate

# STEP 2: Caption
echo "[STEP 2] Creating and setting up caption environment..."
python3.11 -m venv venv_caption
source venv_caption/bin/activate
pip install --upgrade pip
pip install -r requirements/caption.txt

echo "[STEP 2] Generating captions..."
python pipeline/hyperclovax.py "$VIDEO"
python pipeline/caption_selector.py
deactivate

# STEP 3: TTS
echo "[STEP 3] Creating and setting up TTS environment..."
python3.10 -m venv venv_tts
source venv_tts/bin/activate
pip install --upgrade pip
pip install -r requirements/tts.txt
python -m unidic download

echo "[STEP 3] Converting text to speech..."
python pipeline/video_description_with_tts.py "$VIDEO"
deactivate

# FINAL: Copy result
echo "[FINAL] Copying result file..."
mkdir -p results
cp video_output.mp4 "results/$OUTNAME"

echo "[DONE] Pipeline execution completed successfully."
