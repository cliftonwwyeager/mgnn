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
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Conv2D, SeparableConv2D,
    DepthwiseConv2D, Multiply, Activation, Reshape, GlobalMaxPooling1D,
    GlobalAveragePooling2D, MultiHeadAttention, Add, Concatenate, LayerNormalization,
    SpatialDropout2D, Lambda
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam as KerasAdam, SGD as KerasSGD
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import AUC
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
    return list(set(hashes))

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
    return list(set(signatures))

def _byte_category_map(int_sequence: np.ndarray) -> np.ndarray:
    cat = np.empty_like(int_sequence, dtype=np.float32)
    cat[:] = 1.0
    cat[int_sequence == 0] = 0.0
    cat[((int_sequence < 32) | (int_sequence == 127)) & (int_sequence != 0)] = 0.33
    cat[(int_sequence >= 32) & (int_sequence <= 126)] = 0.66
    return cat

def _blockwise_entropy(int_sequence: np.ndarray, block_size: int = 256) -> np.ndarray:
    L = len(int_sequence)
    pad_len = (block_size - (L % block_size)) % block_size
    if pad_len:
        int_sequence = np.pad(int_sequence, (0, pad_len), 'constant')
    blocks = int_sequence.reshape(-1, block_size)
    ent = np.zeros(blocks.shape[0], dtype=np.float32)
    for i, blk in enumerate(blocks):
        counts = np.bincount(blk, minlength=256).astype(np.float32)
        p = counts / (np.sum(counts) + 1e-9)
        p = p[p > 0]
        ent[i] = -np.sum(p * np.log2(p))
    ent = ent / 8.0
    per_byte = np.repeat(ent, block_size)
    return per_byte[:L]

def binary_to_image(file_path, image_dim=256, channels=3):
    with open(file_path, 'rb') as f:
        byte_sequence = f.read()
    int_sequence = np.frombuffer(byte_sequence, dtype=np.uint8)
    desired_length = image_dim * image_dim
    if len(int_sequence) > desired_length:
        int_sequence = int_sequence[:desired_length]
    else:
        int_sequence = np.pad(int_sequence, (0, desired_length - len(int_sequence)), 'constant')

    ch0 = (int_sequence.astype(np.float32) / 255.0)
    ch1 = _blockwise_entropy(int_sequence, block_size=256).astype(np.float32)
    ch2 = _byte_category_map(int_sequence).astype(np.float32)
    stacked = np.stack([ch0, ch1, ch2], axis=-1)  # (desired_length, 3)
    image = stacked.reshape(image_dim, image_dim, channels).astype(np.float32)
    return image

def create_training_dataset_from_uploads(uploads_dir, signatures, image_dim=256):
    X, y = [], []
    if not os.path.exists(uploads_dir):
        return None, None
    seen_hashes = set()
    for root, _, files in os.walk(uploads_dir):
        for fname in files:
            file_path = os.path.join(root, fname)
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                if file_hash in seen_hashes:
                    continue
                seen_hashes.add(file_hash)
                img = binary_to_image(file_path, image_dim=image_dim, channels=3)
                X.append(img)
                label = 1 if file_hash in signatures else 0
                y.append(label)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
    if X:
        X = np.array(X, dtype=np.float32).reshape(-1, image_dim, image_dim, 3)
        y = tf.keras.utils.to_categorical(y, num_classes=2) if tf is not None else np.array(y)
        return X, y
    else:
        return None, None

def SqueezeExcite(ch, ratio=16):
    def block(x):
        s = GlobalAveragePooling2D()(x)
        s = Dense(ch // ratio, activation='relu')(s)
        s = Dense(ch, activation='sigmoid')(s)
        s = tf.keras.layers.Reshape((1,1,ch))(s)
        return Multiply()([x, s])
    return block

def CoordConv():
    def layer(x):
        shape = tf.shape(x)
        h = shape[1]; w = shape[2]
        xx = tf.linspace(-1.0, 1.0, w)
        yy = tf.linspace(-1.0, 1.0, h)
        x_grid = tf.tile(tf.reshape(xx, (1,1,w,1)), (shape[0], h, 1, 1))
        y_grid = tf.tile(tf.reshape(yy, (1,h,1,1)), (shape[0], 1, w, 1))
        return Concatenate(axis=-1)([x, x_grid, y_grid])
    return Lambda(layer)

def ResidualGatedSE(filters, kernel_size=3, stride=1, dropout_rate=0.2):
    def block(x):
        shortcut = x
        x1 = SeparableConv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('swish')(x1)
        x1 = SpatialDropout2D(dropout_rate)(x1)
        gate = SeparableConv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        gate = BatchNormalization()(gate)
        gate = Activation('sigmoid')(gate)
        xg = Multiply()([x1, gate])
        xg = SqueezeExcite(filters)(xg)
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        out = Add()([shortcut, xg])
        out = Activation('swish')(out)
        return out
    return block

def TransformerRefiner(embed_dim, num_heads=4, dropout=0.2):
    def block(x):
        B, H, W, C = x.shape
        seq = Reshape((-1, x.shape[-1]))(x)
        seq_ln = LayerNormalization()(seq)
        mha = MultiHeadAttention(num_heads=num_heads, key_dim=max(8, embed_dim//num_heads), dropout=dropout)
        attn = mha(seq_ln, seq_ln)
        seq = Add()([seq, attn])
        seq_ln2 = LayerNormalization()(seq)
        ffn = Dense(embed_dim*2, activation='swish')(seq_ln2)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(embed_dim)(ffn)
        seq = Add()([seq, ffn])
        return seq
    return block

def build_model(input_shape=(256, 256, 3), base_filters=32, dropout_rate=0.2, num_classes=2):
    inputs = Input(shape=input_shape)
    x = CoordConv()(inputs)
    x = Conv2D(base_filters, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = ResidualGatedSE(base_filters, kernel_size=3, stride=1, dropout_rate=dropout_rate)(x)
    x = ResidualGatedSE(base_filters, kernel_size=5, stride=2, dropout_rate=dropout_rate)(x)
    x = ResidualGatedSE(base_filters*2, kernel_size=3, stride=1, dropout_rate=dropout_rate)(x)
    x = ResidualGatedSE(base_filters*2, kernel_size=3, stride=2, dropout_rate=dropout_rate) (x)
    x = ResidualGatedSE(base_filters*4, kernel_size=3, stride=1, dropout_rate=dropout_rate)(x)
    x = ResidualGatedSE(base_filters*4, kernel_size=3, stride=2, dropout_rate=dropout_rate)(x)
    seq = TransformerRefiner(embed_dim=base_filters*4, num_heads=4, dropout=dropout_rate)(x)
    seq = GlobalMaxPooling1D()(seq)
    head = Dense(256, activation='swish')(seq)
    head = Dropout(dropout_rate)(head)
    head = Dense(128, activation='swish')(head)
    head = Dropout(dropout_rate)(head)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(head)
    model = Model(inputs=inputs, outputs=outputs)
    initial_lr = 3e-4
    lr_schedule = CosineDecayRestarts(initial_learning_rate=initial_lr, first_decay_steps=300, t_mul=2.0, m_mul=0.8, alpha=1e-5)
    optimizer = KerasAdam(learning_rate=lr_schedule)

    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
        return loss

    model.compile(optimizer=optimizer, loss=focal_loss(gamma=2.0, alpha=0.5), metrics=['accuracy', AUC(name='auc')])
    return model

def analyze_file(file_path, model_dir):
    model_path_keras = os.path.join(model_dir, 'best_model.keras')
    model_path_h5 = os.path.join(model_dir, 'best_model.h5')
    model_path = model_path_keras if os.path.exists(model_path_keras) else model_path_h5
    if not os.path.exists(model_path):
        return None
    model = load_model(model_path, compile=False)
    image = binary_to_image(file_path, image_dim=256, channels=3)
    img_array = np.expand_dims(image, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    malware_index = 1
    score = float(predictions[malware_index])
    threshold = float(os.getenv('MALWARE_THRESHOLD', '0.80'))
    pred_class = 1 if score >= threshold else 0
    confidence = score if pred_class == 1 else float(predictions[0])
    return {'file_path': file_path, 'predicted_class': pred_class, 'confidence': float(confidence)}

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
                    with open(event.src_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                except Exception:
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
            if isinstance(record, bytes):
                record = record.decode('utf-8', errors='ignore')
            confirmation_data.append(json.loads(record))
        except json.JSONDecodeError:
            continue
    return confirmation_data

def reinforcement_learning_update(model_dir, confirmation_data):
    model_path_keras = os.path.join(model_dir, 'best_model.keras')
    model_path_h5 = os.path.join(model_dir, 'best_model.h5')
    model_path = model_path_keras if os.path.exists(model_path_keras) else model_path_h5
    if not os.path.exists(model_path):
        return
    model = load_model(model_path, compile=False)
    optimizer = KerasAdam(learning_rate=1e-4)
    images = []
    labels = []
    for data in confirmation_data:
        hash_value = data.get('hash')
        true_class = data.get('true_class', data.get('predicted_class', 0))
        bin_file = os.path.join(model_dir, 'uploads', f"{hash_value}.bin")
        if not os.path.exists(bin_file):
            continue
        img = binary_to_image(bin_file, image_dim=256, channels=3)
        images.append(img)
        labels.append(true_class)
    if not images:
        return
    images = np.array(images, dtype=np.float32)
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    for epoch in range(5):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, preds))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.save(os.path.join(model_dir, 'best_model_updated.keras'))
    model.save(os.path.join(model_dir, 'best_model_updated.h5'))
    logging.info(f"Reinforcement learning update applied to {len(images)} samples.")

def _compute_class_weights(y_one_hot):
    y = np.argmax(y_one_hot, axis=1)
    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    total = positives + negatives
    w0 = total / (2.0 * negatives)
    w1 = total / (2.0 * positives)
    return {0: float(w0), 1: float(w1)}

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
    synthetic_X = np.random.rand(synthetic_count, 256, 256, 3).astype(np.float32)
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
    x_train, x_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.4, random_state=42, stratify=np.argmax(y_data, axis=1))
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1))
    model = build_model(input_shape=(256, 256, 3), base_filters=32, dropout_rate=0.25, num_classes=2)
    class_weights = _compute_class_weights(y_train)
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.keras'), monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.h5'), monitor='val_auc', mode='max', save_best_only=True, verbose=1)
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    loss, accuracy, auc = model.evaluate(x_test, y_test, verbose=0)
    logging.info(f"Final model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    model.save(os.path.join(output_dir, 'last_model.keras'))
    model.save(os.path.join(output_dir, 'last_model.h5'))
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
