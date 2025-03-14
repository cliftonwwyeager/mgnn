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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset

try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalMaxPooling2D, Multiply, Conv2D
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Adam as KerasAdam, SGD as KerasSGD
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
except ImportError:
    tf = None

from sklearn.model_selection import train_test_split
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, db=0)

class MGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MGNNWithTD(MGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.99):
        super(MGNNWithTD, self).__init__(input_dim, hidden_dim, output_dim)
        self.gamma = gamma
    def forward(self, x, target=None, reward=None):
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        if target is not None and reward is not None:
            td_error = reward + self.gamma * target - output
            output = output + td_error
        return output
    def train_with_td(self, optimizer, criterion, scheduler, train_loader, epochs=20):
        self.train()
        for epoch in range(epochs):
            if scheduler:
                scheduler.step()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                target = torch.max(outputs, dim=1)[0].detach()
                reward = (outputs.argmax(dim=1) == labels).float()
                td_outputs = self.forward(inputs, target=target, reward=reward)
                loss = criterion(td_outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            logging.info(f"Epoch {epoch+1}/{epochs} - TD Train Loss: {running_loss/len(train_loader):.4f}")
        logging.info("Finished Training MGNN with TD")

def download_and_extract_csv(url, extract_to):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(path=os.path.dirname(extract_to))
                for filename in z.namelist():
                    if filename.endswith('.csv'):
                        os.rename(os.path.join(os.path.dirname(extract_to), filename), extract_to)
                        break
    except Exception as e:
        logging.error(f"Error downloading or extracting CSV: {str(e)}")

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
        logging.error(f"Error reading CSV: {str(e)}")
    return hashes

def convert_files_to_hashes(directory):
    file_hashes = []
    if not directory or not os.path.exists(directory):
        return file_hashes
    for root, _, files in os.walk(directory):
        for fname in files:
            file_path = os.path.join(root, fname)
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes.append(file_hash)
            except Exception as e:
                logging.error(f"Error hashing file {file_path}: {e}")
    return file_hashes

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

def GatedCNNBlock(filters, kernel_size, stride=(1,1), dropout_rate=0.3):
    def block(x):
        if filters <= 32:
            conv = MobileNetV3Small(input_shape=x.shape[1:], include_top=False, weights='imagenet')(x)
        else:
            conv = EfficientNetB0(input_shape=x.shape[1:], include_top=False, weights='imagenet')(x)
        conv = BatchNormalization()(conv)
        conv = tf.keras.activations.relu(conv)
        conv = Dropout(dropout_rate)(conv)
        gate = Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
        gate = BatchNormalization()(gate)
        gate = tf.keras.activations.sigmoid(gate)
        gate = Dropout(dropout_rate)(gate)
        gated_output = Multiply()([conv, gate])
        return gated_output
    return block

def build_model(input_shape=(256,256,3), num_filters=32, kernel_size=(3,3), dropout_rate=0.3, num_classes=2):
    inputs = Input(shape=input_shape)
    x = GatedCNNBlock(num_filters, kernel_size)(inputs)
    x = GlobalMaxPooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = KerasAdam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_data(directory, img_size=(256,256), batch_size=32, validation_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest',
        validation_split=validation_split
    )
    train_generator = datagen.flow_from_directory(
        directory, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_generator = datagen.flow_from_directory(
        directory, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation'
    )
    return train_generator, val_generator

@tf.function
def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model_tf(data_dir, epochs=10, batch_size=32):
    if tf is None:
        raise ImportError("TensorFlow is not available, cannot train TF model.")
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        train_gen, val_gen = load_and_preprocess_data(data_dir, img_size=(256,256), batch_size=batch_size)
        model = build_model(input_shape=(256,256,3), num_classes=2)
        optimizer = KerasAdam(learning_rate=1e-4)
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    @tf.function
    def distributed_train_step(images, labels):
        per_replica_losses = strategy.run(train_step, args=(model, images, labels, optimizer))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    for epoch in range(epochs):
        for images, labels in train_ds:
            loss = distributed_train_step(images, labels)
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        logging.info(f"[TF] Epoch {epoch+1}/{epochs} - ValLoss: {val_loss:.4f}, ValAcc: {val_accuracy:.4f}")
    model.save(os.path.join(data_dir, 'best_model.h5'))
    logging.info("TensorFlow model training complete.")
    return model

def random_configuration(search_space):
    return {param: random.choice(values) for param, values in search_space.items()}

def evaluate_model(model_builder, x_train, y_train, x_val, y_val, config):
    model = model_builder(
        input_shape=(256,256,1),
        num_filters=config['num_filters'],
        kernel_size=config['kernel_size'],
        dropout_rate=config['dropout_rate'],
        num_classes=2
    )
    model.compile(
        optimizer=KerasSGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=config['epochs'], batch_size=config['batch_size'], verbose=0)
    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc

def run_evolutionary_optimization_generation(model_builder, x_train, y_train, x_val, y_val, search_space, population_size=10):
    population = [random_configuration(search_space) for _ in range(population_size)]
    performances = []
    for config in population:
        performance = evaluate_model(model_builder, x_train, y_train, x_val, y_val, config)
        performances.append(performance)
    best_index = int(np.argmax(performances))
    best_config = population[best_index]
    best_performance = performances[best_index]
    return best_config, best_performance

def evolutionary_optimization_with_feedback(model_builder, x_train, y_train, x_val, y_val, initial_search_space, max_generations=10, performance_threshold=1e-4):
    last_best_performance = 0.0
    best_config = None
    for generation in range(max_generations):
        current_best_config, current_best_perf = run_evolutionary_optimization_generation(
            model_builder, x_train, y_train, x_val, y_val, initial_search_space
        )
        if best_config is None or current_best_perf > last_best_performance:
            best_config = current_best_config
        if abs(current_best_perf - last_best_performance) < performance_threshold:
            logging.info(f"No significant improvement in generation {generation}, stopping search.")
            break
        last_best_performance = current_best_perf
        logging.info(f"Generation {generation}: Best performance = {current_best_perf:.4f}")
    return best_config

def retrieve_confirmations_from_redis():
    accurate_predictions = redis_client.smembers('accurate_predictions')
    confirmation_data = []
    for record in accurate_predictions:
        try:
            confirmation_data.append(json.loads(record))
        except json.JSONDecodeError:
            continue
    return confirmation_data

def reinforcement_learning_update(data_dir, confirmation_data):
    model_path = os.path.join(data_dir, 'best_model.h5')
    if tf is None or not os.path.exists(model_path):
        return
    model = load_model(model_path)
    optimizer = KerasAdam(learning_rate=1e-4)
    images = []
    labels = []
    for data in confirmation_data:
        hash_value = data.get('hash')
        true_class = data.get('true_class', data.get('predicted_class', 0))
        bin_file_path = os.path.join(data_dir, 'uploads', f"{hash_value}.bin")
        if not os.path.exists(bin_file_path):
            continue
        image = binary_to_image(bin_file_path, image_dim=256)
        image = np.expand_dims(image, axis=-1)
        images.append(image)
        labels.append(true_class)
    if not images:
        return
    images = np.array(images)
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    for epoch in range(5):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.save(os.path.join(data_dir, 'best_model_updated.h5'))
    logging.info(f"Reinforcement learning update applied on {len(images)} samples.")

def analyze_file(file_path, data_dir):
    model_path = os.path.join(data_dir, 'best_model.h5')
    if tf is None or not os.path.exists(model_path):
        return None
    model = load_model(model_path)
    image = binary_to_image(file_path, image_dim=256)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    confidence = float(predictions[0][predicted_class])
    return {'file_path': file_path, 'predicted_class': predicted_class, 'confidence': confidence}

def reward_function(tp, fp, fn):
    alpha = 1.0
    beta = 0.5
    gamma = 0.5
    return alpha * tp - beta * fp - gamma * fn

def export_detections_to_siem(detections):
    if not detections:
        return
    data_payload = {"detections": detections, "timestamp": time.time()}
    cortex_url = os.getenv("CORTEX_XSIAM_URL", "https://xsiam.example.api/ingest")
    cortex_token = os.getenv("CORTEX_XSIAM_API_KEY", "example_xsiam_key")
    try:
        headers = {"Authorization": f"Bearer {cortex_token}", "Content-Type": "application/json"}
        requests.post(cortex_url, headers=headers, json=data_payload, timeout=5)
    except Exception as e:
        logging.error(f"Cortex XSIAM export error: {str(e)}")
    elastic_url = os.getenv("ELASTIC_URL", "https://elastic.example:9200")
    elastic_api_key = os.getenv("ELASTIC_API_KEY", "example_elastic_key")
    try:
        headers = {"Authorization": f"ApiKey {elastic_api_key}", "Content-Type": "application/json"}
        index_name = os.getenv("ELASTIC_INDEX", "malware-detections")
        requests.post(f"{elastic_url}/{index_name}/_doc", headers=headers, json=data_payload, verify=False, timeout=5)
    except Exception as e:
        logging.error(f"Elastic export error: {str(e)}")
    splunk_url = os.getenv("SPLUNK_HEC_URL", "https://splunk.example:8088/services/collector")
    splunk_token = os.getenv("SPLUNK_HEC_TOKEN", "example_splunk_token")
    try:
        headers = {"Authorization": f"Splunk {splunk_token}", "Content-Type": "application/json"}
        event_data = {"event": data_payload, "sourcetype": "malware_detection"}
        requests.post(splunk_url, headers=headers, json=event_data, verify=False, timeout=5)
    except Exception as e:
        logging.error(f"Splunk export error: {str(e)}")
    sentinel_workspace_id = os.getenv("SENTINEL_WORKSPACE_ID", "example_workspace_id")
    sentinel_shared_key = os.getenv("SENTINEL_SHARED_KEY", "example_shared_key")
    sentinel_log_name = os.getenv("SENTINEL_LOG_TYPE", "MalwareDetections")
    try:
        sentinel_url = f"https://{sentinel_workspace_id}.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
        headers = {"Content-Type": "application/json", "Log-Type": sentinel_log_name}
        requests.post(sentinel_url, headers=headers, json=data_payload, timeout=5)
    except Exception as e:
        logging.error(f"Sentinel export error: {str(e)}")

class FileCreatedHandler(FileSystemEventHandler):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
    def on_created(self, event):
        if event.is_directory:
            return
        result = analyze_file(event.src_path, self.data_dir)
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
                    reinforcement_learning_update(self.data_dir, confirmations)

def main(data_dir, directories_to_monitor):
    csv_path = os.path.join(data_dir, 'full.csv')
    download_and_extract_csv('https://bazaar.abuse.ch/export/csv/full/', csv_path)
    malicious_hashes = load_malicious_hashes_from_csv(csv_path)
    local_hashes = convert_files_to_hashes(os.path.join(data_dir, 'uploads'))
    search_space = {
        'num_filters': [16, 32, 64],
        'kernel_size': [(3,3), (5,5)],
        'dropout_rate': [0.3, 0.4, 0.5],
        'batch_size': [16, 32],
        'epochs': [5, 10]
    }
    X_data = np.random.rand(100, 256, 256, 1).astype(np.float32)
    y_data = np.random.randint(0, 2, size=(100,))
    y_data = tf.keras.utils.to_categorical(y_data, num_classes=2) if tf is not None else y_data
    x_train, x_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    best_config = evolutionary_optimization_with_feedback(build_model, x_train, y_train, x_val, y_val,
                                                          search_space, max_generations=5)
    logging.info(f"Best hyperparameters from evolutionary search: {best_config}")
    final_model = build_model(input_shape=(256,256,1),
                              num_filters=best_config['num_filters'],
                              kernel_size=best_config['kernel_size'],
                              dropout_rate=best_config['dropout_rate'],
                              num_classes=2)
    final_model.compile(optimizer=KerasSGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                        loss='categorical_crossentropy', metrics=['accuracy'])
    final_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=best_config['epochs'], batch_size=best_config['batch_size'])
    loss, accuracy = final_model.evaluate(x_test, y_test)
    logging.info(f"Final model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    final_model.save(os.path.join(data_dir, 'best_model.h5'))
    confirmations = retrieve_confirmations_from_redis()
    if confirmations:
        reinforcement_learning_update(data_dir, confirmations)
    observers = []
    for folder in directories_to_monitor:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        event_handler = FileCreatedHandler(data_dir)
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
