from flask import Flask, request, jsonify, render_template
import os
import hashlib
import redis
import json
import csv
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import subprocess
app = Flask(__name__)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
redis_password = os.getenv('REDIS_PASSWORD', None)
redis_client = redis.StrictRedis(host=redis_host, port=int(redis_port), password=redis_password, db=0)
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
    for key in [REDIS_KEY_MALWARE_COUNT, REDIS_KEY_INGEST_COUNT, REDIS_KEY_ELASTIC_EXPORTS, REDIS_KEY_CORTEX_EXPORTS, REDIS_KEY_SPLUNK_EXPORTS, REDIS_KEY_SENTINEL_EXPORTS]:
        if not redis_client.exists(key):
            redis_client.set(key, 0)
    if not redis_client.exists(REDIS_KEY_MODEL_ACCURACY):
        redis_client.set(REDIS_KEY_MODEL_ACCURACY, 0.0)
initialize_redis_counters()
def increment_redis_counter(key, amount=1):
    if not redis_client.exists(key):
        redis_client.set(key, 0)
    redis_client.incrby(key, amount)
def append_runtime_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    redis_client.rpush(REDIS_KEY_RUNTIME_LOGS, log_entry)
    redis_client.ltrim(REDIS_KEY_RUNTIME_LOGS, -1000, -1)
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
    return jsonify({"malware_count": malware_count,"ingest_count": ingest_count,"elastic_exports": elastic_exports,"cortex_exports": cortex_exports,"splunk_exports": splunk_exports,"sentinel_exports": sentinel_exports,"model_accuracy": model_accuracy})
@app.route('/get_runtime_logs')
def get_runtime_logs():
    logs = redis_client.lrange(REDIS_KEY_RUNTIME_LOGS, 0, -1)
    return jsonify(logs)
@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    filename = secure_filename(file.filename)
    file_path = os.path.join(output_dir, filename)
    file.save(file_path)
    increment_redis_counter(REDIS_KEY_INGEST_COUNT)
    hash_value = calculate_hash(file_path)
    X = process_file(file_path)
    model = load_model()
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
    if len(X_tensor.shape) == 2 and X_tensor.shape[0] > 1:
        out = outputs[0].unsqueeze(0)
    else:
        out = outputs
    predicted_class = torch.argmax(out, dim=1).item()
    confidence = torch.softmax(out, dim=1).max().item()
    if predicted_class == 1:
        increment_redis_counter(REDIS_KEY_MALWARE_COUNT)
    redis_client.set(hash_value, json.dumps({'predicted_class': predicted_class, 'confidence': confidence}))
    append_runtime_log(f"File uploaded: {filename}, Hash: {hash_value}, Class: {predicted_class}, Confidence: {confidence:.4f}")
    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
@app.route('/process', methods=['POST'])
def process():
    subprocess.call(["python", "mgnn-train.py"])
    return jsonify({'status': 'Model training complete'})
@app.route('/confirm', methods=['GET','POST'])
def confirm():
    if request.method == 'GET':
        return render_template('confirm.html')
    data = request.json
    hash_value = data.get('hash')
    is_correct = data.get('is_correct')
    if not hash_value or is_correct is None:
        return jsonify({'error': 'Invalid input'})
    prediction = redis_client.get(hash_value)
    if not prediction:
        return jsonify({'error': 'No prediction found for the provided hash'})
    prediction = json.loads(prediction)
    record_confirmation(hash_value, prediction, is_correct)
    if is_correct:
        submit_positive_result(hash_value, prediction['predicted_class'])
    redis_client.set(f"confirmed:{hash_value}", json.dumps(prediction))
    append_runtime_log(f"Confirmation for hash {hash_value}, is_correct={is_correct}")
    return jsonify({'status': 'Confirmation recorded'})
@app.route('/results')
def results():
    return render_template('results.html')
@app.route('/get_results')
def get_results():
    csv_file = os.path.join(output_dir, 'confirmations.csv')
    if not os.path.exists(csv_file):
        return jsonify({'error': 'No results found'})
    results = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append(row)
    return jsonify(results)
def submit_positive_result(hash_value, predicted_class):
    record = {'hash': hash_value, 'predicted_class': predicted_class}
    redis_client.sadd('accurate_predictions', json.dumps(record))
def retrieve_confirmations_from_redis():
    accurate_predictions = redis_client.smembers('accurate_predictions')
    return [json.loads(record) for record in accurate_predictions]
def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()
def process_file(file_path):
    data = pd.read_csv(file_path)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data.apply(lambda x: ' '.join(x.astype(str)), axis=1))
    pca = PCA(n_components=100)
    X = pca.fit_transform(tfidf_matrix.toarray())
    return StandardScaler().fit_transform(X)
def load_model():
    best_hidden_dim = redis_client.get("best_hidden_dim")
    best_learning_rate = redis_client.get("best_learning_rate")
    if best_hidden_dim is None:
        best_hidden_dim = 64
    else:
        best_hidden_dim = int(best_hidden_dim)
    if best_learning_rate is None:
        best_learning_rate = 1e-3
    else:
        best_learning_rate = float(best_learning_rate)
    from mgnn import MGNNWithTD
    model = MGNNWithTD(100, best_hidden_dim, 10)
    model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        append_runtime_log("Model loaded with best_model.pth")
    else:
        append_runtime_log("No trained model found; using random weights.")
    return model
def record_confirmation(hash_value, prediction, is_correct):
    csv_file = os.path.join(output_dir, 'confirmations.csv')
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['hash','predicted_class','confidence','is_correct','timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'hash': hash_value,'predicted_class': prediction['predicted_class'],'confidence': prediction['confidence'],'is_correct': is_correct,'timestamp': datetime.now().isoformat()})
def export_to_siem(siem_name, endpoint_url, data_payload, counter_key):
    headers = {"Content-Type": "application/json"}
    if siem_name == "Elastic":
        elastic_api_key = os.getenv("ELASTIC_API_KEY", "someElasticKey")
        headers["Authorization"] = f"ApiKey {elastic_api_key}"
    elif siem_name == "Cortex XSIAM":
        cortex_token = os.getenv("CORTEX_XSIAM_API_KEY", "someCortexKey")
        headers["Authorization"] = f"Bearer {cortex_token}"
    elif siem_name == "Splunk":
        splunk_token = os.getenv("SPLUNK_HEC_TOKEN", "someSplunkToken")
        headers["Authorization"] = f"Splunk {splunk_token}"
    elif siem_name == "Sentinel":
        sentinel_shared_key = os.getenv("SENTINEL_SHARED_KEY", "someSentinelKey")
        headers["Authorization"] = f"SharedKey {sentinel_shared_key}"
    try:
        resp = requests.post(endpoint_url, json=data_payload, headers=headers, timeout=5)
        if 200 <= resp.status_code < 300:
            increment_redis_counter(counter_key, 1)
            append_runtime_log(f"SIEM {siem_name} export success (status={resp.status_code}).")
        else:
            append_runtime_log(f"SIEM {siem_name} export failed (status={resp.status_code}).")
    except Exception as e:
        append_runtime_log(f"SIEM {siem_name} export exception: {str(e)}")
@app.route('/export/elastic', methods=['POST'])
def export_elastic():
    training_results_payload = {"example_key":"example_value","timestamp":datetime.now().isoformat()}
    export_to_siem("Elastic", os.getenv("ELASTIC_ENDPOINT","http://localhost:9200/test_index/_doc"), training_results_payload, REDIS_KEY_ELASTIC_EXPORTS)
    return jsonify({"status": "OK"})
@app.route('/export/cortex', methods=['POST'])
def export_cortex():
    training_results_payload = {"example_key":"example_value","timestamp":datetime.now().isoformat()}
    export_to_siem("Cortex XSIAM", os.getenv("CORTEX_ENDPOINT","http://cortex.example/api/xsiam"), training_results_payload, REDIS_KEY_CORTEX_EXPORTS)
    return jsonify({"status": "OK"})
@app.route('/export/splunk', methods=['POST'])
def export_splunk():
    training_results_payload = {"example_key":"example_value","timestamp":datetime.now().isoformat()}
    export_to_siem("Splunk", os.getenv("SPLUNK_ENDPOINT","http://splunk.example:8088/services/collector"), training_results_payload, REDIS_KEY_SPLUNK_EXPORTS)
    return jsonify({"status": "OK"})
@app.route('/export/sentinel', methods=['POST'])
def export_sentinel():
    training_results_payload = {"example_key":"example_value","timestamp":datetime.now().isoformat()}
    export_to_siem("Sentinel", os.getenv("SENTINEL_ENDPOINT","http://sentinel.example/api/logs"), training_results_payload, REDIS_KEY_SENTINEL_EXPORTS)
    return jsonify({"status": "OK"})
if __name__ == '__main__':
    app.run(debug=True)
