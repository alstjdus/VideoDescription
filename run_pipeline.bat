@echo off
echo [START] Input video path: %~1
echo [START] Output file name: %~2

if "%~1"=="" (
    echo Usage: run_pipeline.bat [video file path] [output file name]
    exit /b
)

if "%~2"=="" (
    echo Output file name argument is required.
    exit /b
)

set VIDEO=%~1
set OUTNAME=%~2

REM STEP 1: Analysis
echo [STEP 1] Creating and setting up analysis environment...
py -3.10 -m venv venv_analysis
venv_analysis\Scripts\python.exe -m pip install --upgrade pip
venv_analysis\Scripts\pip.exe install -r requirements/analysis.txt

echo [STEP 1] Running analysis scripts...
venv_analysis\Scripts\python.exe pipeline/keyframe.py %VIDEO%
venv_analysis\Scripts\python.exe pipeline/process_frame_ocr_deepface.py %VIDEO%
venv_analysis\Scripts\python.exe pipeline/process_frame_yolo.py %VIDEO%
venv_analysis\Scripts\pip.exe install -U openai-whisper
venv_analysis\Scripts\python.exe pipeline/extract_dialouge_whisper.py %VIDEO%
venv_analysis\Scripts\python.exe pipeline/json_merged.py

REM STEP 2: Caption
echo [STEP 2] Creating and setting up caption environment...
py -3.11 -m venv venv_caption
venv_caption\Scripts\python.exe -m pip install --upgrade pip
venv_caption\Scripts\pip.exe install -r requirements/caption.txt

echo [STEP 2] Generating captions...
venv_caption\Scripts\python.exe pipeline/hyperclovax.py %VIDEO%
venv_caption\Scripts\python.exe pipeline/caption_selector.py

REM STEP 3: TTS
echo [STEP 3] Creating and setting up TTS environment...
py -3.10 -m venv venv_tts
venv_tts\Scripts\python.exe -m pip install --upgrade pip
venv_tts\Scripts\pip.exe install -r requirements/tts.txt
venv_tts\Scripts\python.exe -m unidic download

echo [STEP 3] Converting text to speech...
venv_tts\Scripts\python.exe pipeline/video_description_with_tts.py %VIDEO%

REM FINAL: Copy result
echo [FINAL] Copying result file...
copy video_output.mp4 results\%OUTNAME%

echo [DONE] Pipeline execution completed successfully.