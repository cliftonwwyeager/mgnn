from flask import Flask, request, jsonify, render_template
import os
import hashlib
import redis
from werkzeug.utils import secure_filename
from mgnn1 import load_malicious_hashes_from_csv, convert_files_to_hashes, binary_to_image, build_model, train_model

app = Flask(__name__)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
redis_password = os.getenv('REDIS_PASSWORD', None)
redis_client = redis.StrictRedis(host=redis_host, port=int(redis_port), password=redis_password, db=0)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/path/to/save', filename)
        file.save(file_path)
        hash_value = calculate_hash(file_path)
        image = binary_to_image(hash_value, '/path/to/data', '/path/to/full.csv')
        model = build_model((256, 256, 1))
        prediction = model.predict(image)
        redis_client.set(hash_value, prediction.tolist())
        return jsonify({'prediction': prediction.tolist()})

@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    if request.method == 'GET':
        return render_template('confirm.html')
    data = request.json
    hash_value = data.get('hash')
    if not hash_value:
        return jsonify({'error': 'No hash provided'})
    prediction = redis_client.get(hash_value)
    if not prediction:
        return jsonify({'error': 'No prediction found for the provided hash'})
    redis_client.set(f"confirmed:{hash_value}", prediction)
    return jsonify({'status': 'Confirmation recorded'})

def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()

if __name__ == '__main__':
    app.run(debug=True)
