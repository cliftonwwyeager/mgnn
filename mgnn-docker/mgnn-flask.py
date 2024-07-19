import os
import redis
import hashlib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from mgnn import analyze_file, train_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/path/to/upload/folder'
app.config['ALLOWED_EXTENSIONS'] = {'exe', 'dll', 'bin'}

# Initialize Redis
r = redis.Redis(host='redis', port=6379, db=0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        file_hash = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
        r.lpush('file_queue', filepath)
        return jsonify({'message': 'File uploaded successfully', 'file_hash': file_hash}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/process', methods=['GET'])
def process_files():
    while r.llen('file_queue') > 0:
        filepath = r.rpop('file_queue').decode('utf-8')
        result = analyze_file(filepath, app.config['UPLOAD_FOLDER'])  # Pass the upload directory to analyze_file
        file_hash = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
        r.hset('results', file_hash, str(result))
    return jsonify({'message': 'Processing complete'}), 200

@app.route('/api/results/<file_hash>', methods=['GET'])
def get_results(file_hash):
    result = r.hget('results', file_hash)
    if result:
        return jsonify({'file_hash': file_hash, 'result': result.decode('utf-8')}), 200
    else:
        return jsonify({'error': 'No results found for this file'}), 404

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    file_hash = data['file_hash']
    feedback = data['feedback']
    r.hset('feedback', file_hash, feedback)
    return jsonify({'message': 'Feedback received'}), 200

@app.route('/api/train', methods=['POST'])
def train():
    train_model(app.config['UPLOAD_FOLDER'])
    return jsonify({'message': 'Training complete'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)