import os
import io
import csv
import json
import random
import logging
import hashlib
import zipfile
import requests
import numpy as np
import redis
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, Multiply, Activation, Reshape, GlobalMaxPooling1D, MultiHeadAttention, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam as KerasAdam, SGD as KerasSGD
from sklearn.model_selection import train_test_split
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
logging.basicConfig(level=logging.INFO)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, db=0)

def download_and_extract_csv(url, extract_to):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(path=os.path.dirname(extract_to))
                for filename in z.namelist():
                    if filename.endswith('.csv'):
                        os.replace(os.path.join(os.path.dirname(extract_to), filename), extract_to)
                        break
    except Exception as e:
        logging.error(f"Error downloading or extracting CSV from {url}: {e}")

def download_csv(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            logging.error(f"Failed to download {url} - HTTP {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading file from {url}: {e}")

def load_malicious_hashes_from_csv(csv_path):
    hashes = []
    if not csv_path or not os.path.exists(csv_path):
        return hashes
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 1:
                    hashes.append(row[1])
    except Exception as e:
        logging.error(f"Error reading CSV {csv_path}: {e}")
    return hashes

def load_signatures_from_txt(txt_path):
    signatures = []
    if not txt_path or not os.path.exists(txt_path):
        return signatures
    try:
        with open(txt_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    signatures.append(line)
    except Exception as e:
        logging.error(f"Error reading signatures from {txt_path}: {e}")
    return signatures

def binary_to_image(file_path, image_dim=256):
    with open(file_path, 'rb') as f:
        byte_sequence = f.read()
    int_sequence = np.frombuffer(byte_sequence, dtype=np.uint8)
    desired_length = image_dim * image_dim
    if len(int_sequence) > desired_length:
        int_sequence = int_sequence[:desired_length]
    else:
        int_sequence = np.pad(int_sequence, (0, desired_length - len(int_sequence)), 'constant')
    image = int_sequence.reshape(image_dim, image_dim).astype(np.uint8)
    return image / 255.0

def create_training_dataset_from_uploads(uploads_dir, signatures, image_dim=256):
    X, y = [], []
    if not os.path.exists(uploads_dir):
        return None, None
    for root, _, files in os.walk(uploads_dir):
        for fname in files:
            file_path = os.path.join(root, fname)
            try:
                img = binary_to_image(file_path, image_dim)
                X.append(img)
                with open(file_path, 'rb') as f:
                    content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                label = 1 if file_hash in signatures else 0
                y.append(label)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
    if X:
        X = np.array(X).reshape(-1, image_dim, image_dim, 1)
        y = tf.keras.utils.to_categorical(y, num_classes=2) if tf is not None else np.array(y)
        return X, y
    else:
        return None, None

def GatedCNNBlock(filters, kernel_size=(3, 3), stride=(1, 1), dropout_rate=0.3):
    def block(x):
        conv = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(dropout_rate)(conv)
        gate = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        gate = BatchNormalization()(gate)
        gate = Activation('sigmoid')(gate)
        gate = Dropout(dropout_rate)(gate)
        return Multiply()([conv, gate])
    return block

def build_model(input_shape=(256, 256, 1), num_filters=32, kernel_size=(3, 3), dropout_rate=0.4, num_classes=2):
    inputs = Input(shape=input_shape)
    x = inputs
    if input_shape[-1] == 1:
        x = Conv2D(3, (3, 3), padding='same')(x)
    x = GatedCNNBlock(num_filters, kernel_size, stride=(2, 2), dropout_rate=dropout_rate)(x)
    x = GatedCNNBlock(num_filters * 2, kernel_size, stride=(2, 2), dropout_rate=dropout_rate)(x)
    x = GatedCNNBlock(num_filters * 4, kernel_size, stride=(2, 2), dropout_rate=dropout_rate)(x)
    x = Reshape((-1, num_filters * 4))(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=num_filters * 1, dropout=dropout_rate)
    attn_output = attn(x, x)
    x = Add()([x, attn_output])
    x = Dropout(dropout_rate)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = KerasAdam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def analyze_file(file_path, model_dir):
    model_path = os.path.join(model_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        return None
    model = load_model(model_path)
    image = binary_to_image(file_path, image_dim=256)
    img_array = np.expand_dims(image, axis=(0, -1))
    predictions = model.predict(img_array)
    pred_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(predictions[0][pred_class])
    return {'file_path': file_path, 'predicted_class': pred_class, 'confidence': confidence}

def export_detections_to_influxdb(detections):
    if not detections:
        return
    influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    db_name = os.getenv("INFLUXDB_DB", "malware")
    api_key = os.getenv("INFLUXDB_API_KEY", "")
    user = os.getenv("INFLUXDB_USER", "")
    password = os.getenv("INFLUXDB_PASSWORD", "")
    write_url = f"{influxdb_url}/write?db={db_name}"
    lines = []
    timestamp = int(time.time() * 1e9)
    for d in detections:
        file_path = d.get("file_path", "unknown")
        file_tag = file_path.replace(" ", "_").replace(",", "_")
        pred = d.get("predicted_class", 0)
        conf = d.get("confidence", 0.0)
        line = f"malware_detections,file_path={file_tag} predicted_class={pred},confidence={conf} {timestamp}"
        lines.append(line)
    data = "\n".join(lines)
    try:
        headers = {}
        auth = None
        if api_key:
            headers["Authorization"] = f"Token {api_key}"
        elif user and password:
            auth = (user, password)
        response = requests.post(write_url, data=data, headers=headers, auth=auth, timeout=5)
        if response.status_code != 204:
            logging.error(f"InfluxDB export error: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"InfluxDB export exception: {e}")

class FileCreatedHandler(FileSystemEventHandler):
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir
    def on_created(self, event):
        if event.is_directory:
            return
        result = analyze_file(event.src_path, self.model_dir)
        if result is not None:
            if result['predicted_class'] == 1:
                try:
                    file_hash = hashlib.sha256(open(event.src_path, 'rb').read()).hexdigest()
                except Exception as e:
                    file_hash = None
                if file_hash:
                    redis_client.sadd('detected_malware', file_hash)
                export_detections_to_siem([result])
                confirmations = retrieve_confirmations_from_redis()
                if confirmations:
                    reinforcement_learning_update(self.model_dir, confirmations)

def export_detections_to_siem(detections):
    if not detections:
        return
    data_payload = {"detections": detections, "timestamp": time.time()}
    cortex_url = os.getenv("CORTEX_XSIAM_URL", "https://xsiam.example.api/ingest")
    cortex_token = os.getenv("CORTEX_XSIAM_API_KEY", "")
    try:
        headers = {"Authorization": f"Bearer {cortex_token}", "Content-Type": "application/json"}
        requests.post(cortex_url, headers=headers, json=data_payload, timeout=5)
    except Exception as e:
        logging.error(f"Cortex XSIAM export error: {e}")
    elastic_url = os.getenv("ELASTIC_URL", "https://elastic.example:9200")
    elastic_api_key = os.getenv("ELASTIC_API_KEY", "")
    try:
        headers = {"Authorization": f"ApiKey {elastic_api_key}", "Content-Type": "application/json"}
        index_name = os.getenv("ELASTIC_INDEX", "malware-detections")
        requests.post(f"{elastic_url}/{index_name}/_doc", headers=headers, json=data_payload, verify=False, timeout=5)
    except Exception as e:
        logging.error(f"Elastic export error: {e}")
    splunk_url = os.getenv("SPLUNK_HEC_URL", "https://splunk.example:8088/services/collector")
    splunk_token = os.getenv("SPLUNK_HEC_TOKEN", "example_splunk_token")
    try:
        headers = {"Authorization": f"Splunk {splunk_token}", "Content-Type": "application/json"}
        event_data = {"event": data_payload, "sourcetype": "malware_detection"}
        requests.post(splunk_url, headers=headers, json=event_data, verify=False, timeout=5)
    except Exception as e:
        logging.error(f"Splunk export error: {e}")
    sentinel_workspace = os.getenv("SENTINEL_WORKSPACE_ID", "example_workspace_id")
    sentinel_key = os.getenv("SENTINEL_SHARED_KEY", "example_shared_key")
    sentinel_log_type = os.getenv("SENTINEL_LOG_TYPE", "MalwareDetections")
    try:
        sentinel_url = f"https://{sentinel_workspace}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
        headers = {"Content-Type": "application/json", "Log-Type": sentinel_log_type}
        requests.post(sentinel_url, headers=headers, json=data_payload, timeout=5)
    except Exception as e:
        logging.error(f"Sentinel export error: {e}")
    export_detections_to_influxdb(detections)

def retrieve_confirmations_from_redis():
    confirmed = redis_client.smembers('accurate_predictions')
    confirmation_data = []
    for record in confirmed:
        try:
            confirmation_data.append(json.loads(record))
        except json.JSONDecodeError:
            continue
    return confirmation_data

def reinforcement_learning_update(model_dir, confirmation_data):
    model_path = os.path.join(model_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        return
    model = load_model(model_path)
    optimizer = KerasAdam(learning_rate=1e-4)
    images = []
    labels = []
    for data in confirmation_data:
        hash_value = data.get('hash')
        true_class = data.get('true_class', data.get('predicted_class', 0))
        bin_file = os.path.join(model_dir, 'uploads', f"{hash_value}.bin")
        if not os.path.exists(bin_file):
            continue
        img = binary_to_image(bin_file, image_dim=256)
        img = np.expand_dims(img, axis=(0, -1))
        images.append(img[0])
        labels.append(true_class)
    if not images:
        return
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    for epoch in range(5):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, preds))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.save(os.path.join(model_dir, 'best_model_updated.h5'))
    logging.info(f"Reinforcement learning update applied to {len(images)} samples.")

def main(data_dir, directories_to_monitor):
    output_dir = os.path.join("mgnn", "mgnn-docker")
    os.makedirs(output_dir, exist_ok=True)
    abuse_csv = os.path.join(data_dir, 'abuse.csv')
    download_and_extract_csv('https://bazaar.abuse.ch/export/csv/full/', abuse_csv)
    malshare_txt = os.path.join(data_dir, 'malshare.txt')
    download_csv('https://malshare.com/daily/malshare.txt', malshare_txt)
    openmalware_txt = os.path.join(data_dir, 'openmalware.txt')
    download_csv('https://openmalware.com/daily/openmalware.txt', openmalware_txt)
    abuse_hashes = load_malicious_hashes_from_csv(abuse_csv)
    malshare_hashes = load_signatures_from_txt(malshare_txt)
    openmalware_hashes = load_signatures_from_txt(openmalware_txt)
    all_signatures = list(set(abuse_hashes + malshare_hashes + openmalware_hashes))
    logging.info(f"Total combined signatures: {len(all_signatures)}")
    uploads_dir = os.path.join(data_dir, 'uploads')
    real_X, real_y = create_training_dataset_from_uploads(uploads_dir, all_signatures, image_dim=256)
    synthetic_count = 100
    synthetic_X = np.random.rand(synthetic_count, 256, 256, 1).astype(np.float32)
    synthetic_y = np.random.randint(0, 2, size=(synthetic_count,))
    synthetic_y = tf.keras.utils.to_categorical(synthetic_y, num_classes=2) if tf is not None else synthetic_y
    if real_X is not None and real_y is not None and len(real_X) > 0:
        X_data = np.concatenate([real_X, synthetic_X], axis=0)
        y_data = np.concatenate([real_y, synthetic_y], axis=0)
        logging.info(f"Combined training data: {X_data.shape[0]} samples (Real: {real_X.shape[0]}, Synthetic: {synthetic_count})")
    else:
        X_data, y_data = synthetic_X, synthetic_y
        logging.info("No real training data found; using synthetic data only.")
    indices = np.arange(X_data.shape[0]); np.random.shuffle(indices)
    X_data, y_data = X_data[indices], y_data[indices]
    x_train, x_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    model = build_model(input_shape=(256, 256, 1), num_filters=32, kernel_size=(3, 3), dropout_rate=0.4, num_classes=2)
    model.compile(optimizer=KerasSGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=10, batch_size=32)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logging.info(f"Final model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    model.save(os.path.join(output_dir, 'best_model.h5'))
    confirmations = retrieve_confirmations_from_redis()
    if confirmations:
        reinforcement_learning_update(output_dir, confirmations)
    observers = []
    for folder in directories_to_monitor:
        os.makedirs(folder, exist_ok=True)
        event_handler = FileCreatedHandler(output_dir)
        observer = Observer()
        observer.schedule(event_handler, folder, recursive=True)
        observer.start()
        observers.append(observer)
    logging.info(f"Monitoring directories: {directories_to_monitor}")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        for obs in observers:
            obs.stop()
        for obs in observers:
            obs.join()

if __name__ == "__main__":
    data_dir = "/path/to/upload/folder"
    directories_to_monitor = [os.path.join(data_dir, 'uploads')]
    main(data_dir, directories_to_monitor)
    data_dir = "/path/to/upload/folder"
    directories_to_monitor = [os.path.join(data_dir, 'uploads')]
    main(data_dir, directories_to_monitor)
