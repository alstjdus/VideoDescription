from flask import Flask, render_template, request, send_file, jsonify
import subprocess
import uuid
import os

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
BATCH_SCRIPT = os.path.join(BASE_DIR, 'run_pipeline.bat')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    unique_id = str(uuid.uuid4())
    input_filename = unique_id + '.mp4'
    output_filename = unique_id + '_output.mp4'
    video_path = os.path.join(UPLOAD_FOLDER, input_filename)

    request.files['video'].save(video_path)
    print(f"[INFO] 업로드된 파일 저장 완료: {video_path}")

    try:
        # run_pipeline.bat에 입력 파일 경로만 전달
        result = subprocess.run(
            ['cmd.exe', '/c', BATCH_SCRIPT, video_path, output_filename],
            shell=True,
            check=True,
            cwd=BASE_DIR,
            stdout=None,  # 실시간 터미널 출력
            stderr=None
        )

        print('stdout:', result.stdout)
        print('stderr:', result.stderr)

        # 결과 파일이 있다고 가정하는 경로 지정 (예: TTS 결과가 이 이름으로 저장된다고 가정)
        output_filename = 'video_output.mp4'
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        if not os.path.exists(output_path):
            return jsonify({'error': '분석 결과 없음'}), 500

        return send_file(output_path, as_attachment=False, mimetype='video/mp4')

    except subprocess.CalledProcessError as e:
        return jsonify({'error': '파이프라인 실행 실패', 'detail': e.stderr or str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)