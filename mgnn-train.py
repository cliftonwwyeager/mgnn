import os
import json
import random
import logging
import hashlib
import numpy as np
import redis
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from sklearn.model_selection import train_test_split
from mgnn import download_and_extract_csv, download_csv, load_malicious_hashes_from_csv, load_signatures_from_txt
from mgnn import create_training_dataset_from_uploads, binary_to_image, build_model
logging.basicConfig(level=logging.INFO)
influxdb_url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
token = os.getenv('INFLUXDB_TOKEN', '')
org = os.getenv('INFLUXDB_ORG', '')
bucket = os.getenv('INFLUXDB_BUCKET', '')
client = influxdb_client.InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_metrics(epoch, loss, accuracy):
    point = influxdb_client.Point("training_metrics").tag("model", "MGNN_TF") \
        .field("epoch", epoch).field("loss", float(loss)).field("accuracy", float(accuracy))
    write_api.write(bucket=bucket, org=org, record=point)

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

def write_to_redis(key, value):
    r.set(key, value)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
val_split = 0.2
batch_size = 64
epochs = 20
output_dir = '/home/user/mgnn'
os.makedirs(output_dir, exist_ok=True)
logging.info("Fetching latest malware signatures from Abuse.ch, Malshare, OpenMalware...")
abuse_csv = os.path.join(output_dir, 'abuse.csv')
malshare_txt = os.path.join(output_dir, 'malshare.txt')
openmalware_txt = os.path.join(output_dir, 'openmalware.txt')
download_and_extract_csv('https://bazaar.abuse.ch/export/csv/full/', abuse_csv)
download_csv('https://malshare.com/daily/malshare.txt', malshare_txt)
download_csv('https://openmalware.com/daily/openmalware.txt', openmalware_txt)
abuse_hashes = load_malicious_hashes_from_csv(abuse_csv)
malshare_hashes = load_signatures_from_txt(malshare_txt)
openmalware_hashes = load_signatures_from_txt(openmalware_txt)
all_signatures = list(set(abuse_hashes + malshare_hashes + openmalware_hashes))
logging.info(f"Total combined malware signatures: {len(all_signatures)}")
uploads_dir = os.path.join(output_dir, 'uploads')
real_X, real_y = create_training_dataset_from_uploads(uploads_dir, all_signatures, image_dim=256)
image_dim = 256
synthetic_mal_count = 100
synthetic_ben_count = 100
synthetic_mal_X = []
synthetic_ben_X = []
for _ in range(synthetic_mal_count):
    arr = np.random.randint(0, 256, size=(image_dim * image_dim,), dtype=np.uint8)
    arr[0], arr[1] = 77, 90
    img = arr.reshape(image_dim, image_dim).astype(np.uint8)
    synthetic_mal_X.append(img / 255.0)
for _ in range(synthetic_ben_count):
    arr = np.random.randint(0, 256, size=(image_dim * image_dim,), dtype=np.uint8)
    if arr[0] == 77 and arr[1] == 90:
        arr[1] = (arr[1] + 1) % 256
    img = arr.reshape(image_dim, image_dim).astype(np.uint8)
    synthetic_ben_X.append(img / 255.0)
synthetic_mal_X = np.array(synthetic_mal_X).reshape(-1, image_dim, image_dim, 1)
synthetic_ben_X = np.array(synthetic_ben_X).reshape(-1, image_dim, image_dim, 1)
synthetic_mal_y = to_categorical(np.ones(synthetic_mal_X.shape[0], dtype=np.int32), num_classes=2)
synthetic_ben_y = to_categorical(np.zeros(synthetic_ben_X.shape[0], dtype=np.int32), num_classes=2)
if real_X is not None and real_y is not None and len(real_X) > 0:
    X_data = np.concatenate([real_X, synthetic_mal_X, synthetic_ben_X], axis=0)
    y_data = np.concatenate([real_y, synthetic_mal_y, synthetic_ben_y], axis=0)
    logging.info(f"Combined training dataset: {X_data.shape[0]} samples "
                 f"(Real: {real_X.shape[0]}, Synthetic: {synthetic_mal_X.shape[0] + synthetic_ben_X.shape[0]})")
else:
    X_data = np.concatenate([synthetic_mal_X, synthetic_ben_X], axis=0)
    y_data = np.concatenate([synthetic_mal_y, synthetic_ben_y], axis=0)
    logging.info("No real training data found. Using synthetic data only.")
indices = np.arange(X_data.shape[0])
np.random.shuffle(indices)
X_data = X_data[indices]
y_data = y_data[indices]
y_classes = np.argmax(y_data, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=val_split, random_state=42, stratify=y_classes
)
model = build_model(input_shape=(256, 256, 1), num_filters=32, kernel_size=(3, 3), dropout_rate=0.4, num_classes=2)
checkpoint_path = os.path.join(output_dir, 'best_model.h5')
checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', mode='max',
                                save_best_only=True, verbose=1)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val), callbacks=[checkpoint_cb])
best_model = tf.keras.models.load_model(checkpoint_path)
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
logging.info(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy * 100:.2f}%")
write_metrics(epochs, val_loss, val_accuracy * 100.0)
write_to_redis("stats:model_accuracy", f"{val_accuracy * 100.0:.2f}")
write_to_redis("training_complete", "true")
logging.info(f"Saved trained model to {checkpoint_path}")
print("Training complete.")
print(f"Validation Accuracy: {val_accuracy * 100.0:.2f}%")
