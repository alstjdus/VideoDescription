from flask import Flask, render_template, request, send_file, jsonify
import subprocess
import uuid
import os

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
SCRIPT_PATH = os.path.join(BASE_DIR, 'run_pipeline.sh')  # Linux용 .sh 파일로 수정

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
        # run_pipeline.sh에 실행 권한 부여
        subprocess.run(['chmod', '+x', 'run_pipeline.sh'], cwd=BASE_DIR, check=True)

        # 파이프라인 실행 (bash 사용)
        result = subprocess.run(
            ['bash', 'run_pipeline.sh', video_path, output_filename],
            check=True,
            cwd=BASE_DIR
        )

        print('[INFO] 파이프라인 실행 완료')

        # 결과 파일 경로
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        if not os.path.exists(output_path):
            return jsonify({'error': '분석 결과 없음'}), 500

        return send_file(output_path, as_attachment=False, mimetype='video/mp4')

    except subprocess.CalledProcessError as e:
        return jsonify({'error': '파이프라인 실행 실패', 'detail': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render가 할당하는 포트 사용
    app.run(host='0.0.0.0', port=port)
