from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import json
import logging
import hashlib
import time
import requests
import subprocess
import numpy as np
import pickle
import redis
from datetime import datetime
from mgnn import build_model, binary_to_image
app = Flask(__name__)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
output_dir = '/home/user/mgnn'
os.makedirs(output_dir, exist_ok=True)
REDIS_KEY_MALWARE_COUNT = "stats:malware_count"
REDIS_KEY_INGEST_COUNT = "stats:ingest_count"
REDIS_KEY_ELASTIC_EXPORTS = "stats:elastic_exports"
REDIS_KEY_CORTEX_EXPORTS = "stats:cortex_exports"
REDIS_KEY_SPLUNK_EXPORTS = "stats:splunk_exports"
REDIS_KEY_SENTINEL_EXPORTS = "stats:sentinel_exports"
REDIS_KEY_MODEL_ACCURACY = "stats:model_accuracy"
REDIS_KEY_RUNTIME_LOGS = "logs:runtime"

def initialize_redis_counters():
    for key in [REDIS_KEY_MALWARE_COUNT, REDIS_KEY_INGEST_COUNT, REDIS_KEY_ELASTIC_EXPORTS, 
                REDIS_KEY_CORTEX_EXPORTS, REDIS_KEY_SPLUNK_EXPORTS, REDIS_KEY_SENTINEL_EXPORTS, REDIS_KEY_MODEL_ACCURACY]:
        if not redis_client.exists(key):
            redis_client.set(key, "0.0" if key == REDIS_KEY_MODEL_ACCURACY else 0)
initialize_redis_counters()

def increment_redis_counter(key, amount=1):
    if not redis_client.exists(key):
        redis_client.set(key, 0)
    redis_client.incrby(key, amount)

def append_runtime_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = json.dumps({"time": timestamp, "message": message})
    redis_client.rpush(REDIS_KEY_RUNTIME_LOGS, entry)
    redis_client.ltrim(REDIS_KEY_RUNTIME_LOGS, -1000, -1)

def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()

def process_file(file_path):
    img = binary_to_image(file_path, image_dim=256)
    X = np.expand_dims(img, axis=(0, -1))
    return X

def load_model():
    model_path = os.path.join(output_dir, 'best_model.h5')
    model = None
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            append_runtime_log(f"Model loaded from {model_path}.")
        except Exception as e:
            append_runtime_log(f"Error loading model: {e}")
            model = None
    if model is None:
        try:
            model = build_model(input_shape=(256, 256, 1), num_classes=2)
            append_runtime_log("No trained model found; using a fresh untrained model.")
        except Exception as e:
            append_runtime_log(f"Failed to initialize model: {e}")
            model = None
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return render_template('status.html')

@app.route('/get_stats')
def get_stats():
    malware_count = int(redis_client.get(REDIS_KEY_MALWARE_COUNT) or 0)
    ingest_count = int(redis_client.get(REDIS_KEY_INGEST_COUNT) or 0)
    elastic_exports = int(redis_client.get(REDIS_KEY_ELASTIC_EXPORTS) or 0)
    cortex_exports = int(redis_client.get(REDIS_KEY_CORTEX_EXPORTS) or 0)
    splunk_exports = int(redis_client.get(REDIS_KEY_SPLUNK_EXPORTS) or 0)
    sentinel_exports = int(redis_client.get(REDIS_KEY_SENTINEL_EXPORTS) or 0)
    model_accuracy = float(redis_client.get(REDIS_KEY_MODEL_ACCURACY) or 0.0)
    return jsonify({
        "malware_count": malware_count,
        "ingest_count": ingest_count,
        "elastic_exports": elastic_exports,
        "cortex_exports": cortex_exports,
        "splunk_exports": splunk_exports,
        "sentinel_exports": sentinel_exports,
        "model_accuracy": model_accuracy
    })

@app.route('/get_runtime_logs')
def get_runtime_logs():
    logs = redis_client.lrange(REDIS_KEY_RUNTIME_LOGS, 0, -1)
    decoded_logs = []
    for entry in logs:
        try:
            decoded_logs.append(json.loads(entry))
        except Exception:
            decoded_logs.append(entry if isinstance(entry, str) else entry.decode('utf-8'))
    return jsonify(decoded_logs)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(output_dir, filename)
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {e}'}), 500
    increment_redis_counter(REDIS_KEY_INGEST_COUNT)
    hash_value = calculate_hash(file_path)
    try:
        X = process_file(file_path)
    except Exception as e:
        os.remove(file_path)
        logging.error(f"Error processing file {file_path}: {e}")
        return jsonify({'error': 'An error occurred while processing the file.'}), 400
    model = load_model()
    if model is None:
        os.remove(file_path)
        return jsonify({'error': 'Model not available'}), 500
    predictions = model.predict(X)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    if predicted_class == 1:
        increment_redis_counter(REDIS_KEY_MALWARE_COUNT)
    redis_client.set(hash_value, json.dumps({
        'predicted_class': predicted_class,
        'confidence': confidence
    }))
    append_runtime_log(f"File '{filename}' processed (SHA256: {hash_value}). "
                       f"Prediction: {predicted_class} (Conf={confidence:.4f})")
    try:
        os.remove(file_path)
    except Exception:
        pass
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

@app.route('/process', methods=['POST'])
def process_training():
    subprocess.call(["python", "mgnn-train.py"])
    try:
        model_acc = redis_client.get(REDIS_KEY_MODEL_ACCURACY)
        append_runtime_log(f"Training completed. New model accuracy: {model_acc}%")
    except Exception as e:
        append_runtime_log(f"Training process completed with error: {e}")
    return jsonify({"status": "Model training complete"})

@app.route('/confirm', methods=['POST'])
def confirm():
    data = request.get_json(force=True)
    hash_value = data.get('hash')
    is_correct = data.get('is_correct')
    true_class = data.get('true_class', None)
    if not hash_value or is_correct is None:
        return jsonify({'error': 'Invalid input'}), 400
    prediction_json = redis_client.get(hash_value)
    if not prediction_json:
        return jsonify({'error': 'No prediction found for the provided hash'}), 404
    prediction = json.loads(prediction_json)
    record_confirmation(hash_value, prediction, bool(is_correct))
    if is_correct:
        record = {'hash': hash_value, 'predicted_class': prediction['predicted_class']}
        redis_client.sadd('accurate_predictions', json.dumps(record))
        if true_class is not None:
            record_update = {
                'hash': hash_value,
                'predicted_class': prediction['predicted_class'],
                'true_class': true_class
            }
            redis_client.sadd('accurate_predictions', json.dumps(record_update))
    append_runtime_log(f"User confirmation received for {hash_value}: is_correct={is_correct}")
    return jsonify({'status': 'Confirmation recorded'})

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/get_results')
def get_results():
    csv_file = os.path.join(output_dir, 'confirmations.csv')
    if not os.path.exists(csv_file):
        return jsonify({'error': 'No results found'}), 404
    results = []
    try:
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(row)
    except Exception as e:
        return jsonify({'error': f'Failed to read results: {e}'}), 500
    return jsonify(results)

def record_confirmation(hash_value, prediction, is_correct):
    csv_file = os.path.join(output_dir, 'confirmations.csv')
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = ['hash', 'predicted_class', 'confidence', 'is_correct', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'hash': hash_value,
                'predicted_class': prediction.get('predicted_class'),
                'confidence': prediction.get('confidence'),
                'is_correct': bool(is_correct),
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        append_runtime_log(f"Error recording confirmation: {e}")

def export_to_siem(siem_name, endpoint_url, data_payload, counter_key):
    headers = {"Content-Type": "application/json"}
    if siem_name == "Elastic":
        api_key = os.getenv("ELASTIC_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
    elif siem_name == "Cortex XSIAM":
        token = os.getenv("CORTEX_XSIAM_API_KEY", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif siem_name == "Splunk":
        token = os.getenv("SPLUNK_HEC_TOKEN", "")
        if token:
            headers["Authorization"] = f"Splunk {token}"
    elif siem_name == "Sentinel":
        shared_key = os.getenv("SENTINEL_SHARED_KEY", "")
        if shared_key:
            headers["Authorization"] = f"SharedKey {shared_key}"
    try:
        resp = requests.post(endpoint_url, json=data_payload, headers=headers, timeout=5)
        if 200 <= resp.status_code < 300:
            increment_redis_counter(counter_key, 1)
            append_runtime_log(f"{siem_name} export successful (status={resp.status_code}).")
        else:
            append_runtime_log(f"{siem_name} export failed (status={resp.status_code}).")
    except Exception as e:
        append_runtime_log(f"{siem_name} export exception: {e}")

@app.route('/export/elastic', methods=['POST'])
def export_elastic():
    payload = {"event": "training_complete", "timestamp": datetime.now().isoformat()}
    export_to_siem("Elastic", os.getenv("ELASTIC_ENDPOINT", "http://localhost:9200/mgnn_index/_doc"), payload, REDIS_KEY_ELASTIC_EXPORTS)
    return jsonify({"status": "OK"})

@app.route('/export/cortex', methods=['POST'])
def export_cortex():
    payload = {"event": "training_complete", "timestamp": datetime.now().isoformat()}
    export_to_siem("Cortex XSIAM", os.getenv("CORTEX_ENDPOINT", "http://cortex.example/api/xsiam"), payload, REDIS_KEY_CORTEX_EXPORTS)
    return jsonify({"status": "OK"})

@app.route('/export/splunk', methods=['POST'])
def export_splunk():
    payload = {"event": "training_complete", "timestamp": datetime.now().isoformat()}
    export_to_siem("Splunk", os.getenv("SPLUNK_ENDPOINT", "http://splunk.example:8088/services/collector"), payload, REDIS_KEY_SPLUNK_EXPORTS)
    return jsonify({"status": "OK"})

@app.route('/export/sentinel', methods=['POST'])
def export_sentinel():
    payload = {"event": "training_complete", "timestamp": datetime.now().isoformat()}
    export_to_siem("Sentinel", os.getenv("SENTINEL_ENDPOINT", "http://sentinel.example/api/logs"), payload, REDIS_KEY_SENTINEL_EXPORTS)
    return jsonify({"status": "OK"})

@app.route('/export/influx', methods=['POST'])
def export_influx():
    if request.is_json:
        detection_payload = request.get_json(force=True)
        export_to_siem("Elastic", os.getenv("INFLUXDB_URL", "http://localhost:8086"), detection_payload, REDIS_KEY_ELASTIC_EXPORTS)
    return jsonify({"status": "OK"})
        detection_payload = request.get_json(force=True)
        export_to_siem("Elastic", os.getenv("INFLUXDB_URL", "http://localhost:8086"), detection_payload, REDIS_KEY_ELASTIC_EXPORTS)
    return jsonify({"status": "OK"})
