import os
import io
import csv
import json
import random
import logging
import hashlib
import zipfile
import urllib.request
import numpy as np
import redis
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import optuna
from deap import base, creator, tools, algorithms
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from mgnn import MGNNWithTD

logging.basicConfig(level=logging.INFO)
influxdb_url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
token = os.getenv('INFLUXDB_TOKEN', '')
org = os.getenv('INFLUXDB_ORG', '')
bucket = os.getenv('INFLUXDB_BUCKET', '')
client = influxdb_client.InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_metrics(epoch, loss, accuracy):
    point = influxdb_client.Point("training_metrics") \
        .tag("model", "MGNNWithTD") \
        .field("epoch", epoch) \
        .field("loss", float(loss)) \
        .field("accuracy", float(accuracy))
    write_api.write(bucket=bucket, org=org, record=point)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

def write_to_redis(key, value):
    r.set(key, value)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
input_dim = 100
output_dim = 10 
batch_size = 64
epochs = 20
val_split = 0.2
output_dir = '/home/user/mgnn'
os.makedirs(output_dir, exist_ok=True)
url = 'https://bazaar.abuse.ch/export/csv/full/'
zip_path = os.path.join(output_dir, 'full.zip')
csv_path = os.path.join(output_dir, 'full.csv')
try:
    logging.info("Downloading malware dataset CSV...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
except Exception as e:
    logging.error(f"Failed to download or extract dataset: {e}")
logging.info("Loading dataset...")
if not os.path.exists(csv_path):
    raise FileNotFoundError("Dataset CSV not found. Ensure download was successful.")
data = pd.read_csv(csv_path)
X_raw = data.iloc[:, :-1].astype(str)
y = data.iloc[:, -1].values
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X_raw.apply(lambda x: ' '.join(x), axis=1))
pca = PCA(n_components=input_dim)
X_pca = pca.fit_transform(X_tfidf.toarray())

def generate_obfuscated_samples(X, noise_factor=0.05):
    X_obf = X.copy()
    n, m = X_obf.shape
    X_obf += noise_factor * np.random.randn(n, m)
    for i in range(n):
        cnt = random.randint(1, max(1, int(m * 0.05)))
        idx = np.random.choice(m, cnt, replace=False)
        shift = np.random.uniform(-0.1, 0.1)
        X_obf[i, idx] += shift
    return X_obf

def generate_random_obfuscation(X, num_samples=5000, scaling_factor=0.1):
    _, m = X.shape
    mn, mx = X.min(), X.max()
    rg = mx - mn
    R = mn + rg * np.random.rand(num_samples, m)
    R *= scaling_factor
    flips = np.random.rand(num_samples, m) < 0.05
    R[flips] = -R[flips]
    return R

def generate_pseudo_code(num_samples=5000, input_dim=100, token_variety=50):
    data = np.zeros((num_samples, input_dim))
    for i in range(num_samples):
        data[i] = np.random.randint(0, token_variety, size=(input_dim,)).astype(float)
        data[i] += np.random.uniform(-0.2, 0.2, size=(input_dim,))
    return data

logging.info("Generating synthetic data for augmentation...")
malicious_mask = (y == 1) if 1 in y else np.zeros_like(y, dtype=bool)
X_malicious = X_pca[malicious_mask] if malicious_mask.any() else np.empty((0, input_dim))
X_benign = X_pca[~malicious_mask] if (~malicious_mask).any() else np.empty((0, input_dim))
if X_malicious.size > 0:
    X_obf = generate_obfuscated_samples(X_malicious)
else:
    X_obf = np.empty((0, input_dim))
X_noise = generate_random_obfuscation(X_pca, num_samples=5000)
X_pseudo = generate_pseudo_code(num_samples=5000, input_dim=input_dim)
X_synthetic = np.vstack([X_obf, X_noise, X_pseudo])
y_synthetic = np.concatenate([np.ones(len(X_obf)), np.zeros(len(X_noise) + len(X_pseudo))])

X_combined = np.vstack([X_pca, X_synthetic])
y_combined = np.concatenate([y, y_synthetic])
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y_combined, test_size=val_split, random_state=42, stratify=y_combined
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle_vectorizer = json.dumps(tfidf_vectorizer.vocabulary_)
    f.write(pickle_vectorizer.encode('utf-8'))
with open(os.path.join(output_dir, 'pca_components.npy'), 'wb') as f:
    np.save(f, pca.components_)
    np.save(f, pca.mean_)
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
logging.info("Saved vectorizer, PCA, and scaler for inference.")

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def evaluate(individual):
    hidden_dim = int(individual[0])
    learning_rate = float(individual[1])
    model = MGNNWithTD(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model.train_with_td(optimizer, criterion, scheduler, train_loader, epochs=epochs)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100.0 * correct / total
    write_to_redis(f"val_loss_{hidden_dim}_{learning_rate}", val_loss)
    write_to_redis(f"val_accuracy_{hidden_dim}_{learning_rate}", val_accuracy)
    return (val_loss,)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_hidden_dim", random.randint, 16, 128)
toolbox.register("attr_learning_rate", random.uniform, 1e-5, 1e-1)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_hidden_dim, toolbox.attr_learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=[16, 1e-5], up=[128, 1e-1], eta=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

pop_size = 50
generations = 10
logging.info(f"Starting genetic algorithm hyperparameter search (pop={pop_size}, generations={generations})...")
population = toolbox.population(n=pop_size)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, stats=None, halloffame=None, verbose=True)
best_individual = tools.selBest(population, k=1)[0]
best_hidden_dim, best_learning_rate = int(best_individual[0]), float(best_individual[1])
write_to_redis("best_hidden_dim", best_hidden_dim)
write_to_redis("best_learning_rate", best_learning_rate)
logging.info(f"GA best individual -> hidden_dim: {best_hidden_dim}, learning_rate: {best_learning_rate:.6f}")

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    model = MGNNWithTD(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model.train_with_td(optimizer, criterion, scheduler, train_loader, epochs=epochs)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100.0 * correct / total
    write_to_redis(f"trial_{trial.number}_hidden_dim", hidden_dim)
    write_to_redis(f"trial_{trial.number}_learning_rate", learning_rate)
    write_to_redis(f"trial_{trial.number}_val_loss", val_loss)
    write_to_redis(f"trial_{trial.number}_val_accuracy", val_accuracy)
    return val_loss

logging.info("Starting Optuna hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_trial = study.best_trial
best_hidden_dim = best_trial.params['hidden_dim']
best_learning_rate = best_trial.params['learning_rate']
write_to_redis("optuna_best_hidden_dim", best_hidden_dim)
write_to_redis("optuna_best_learning_rate", best_learning_rate)
write_to_redis("optuna_best_loss", best_trial.value)
logging.info(f"Optuna best trial -> hidden_dim: {best_hidden_dim}, learning_rate: {best_learning_rate:.6f}, loss: {best_trial.value:.4f}")

logging.info("Training final model with best hyperparameters...")
model = MGNNWithTD(input_dim, best_hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
model.train_with_td(optimizer, criterion, scheduler, train_loader, epochs=epochs)
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
val_loss /= len(val_loader.dataset)
val_accuracy = 100.0 * correct / total
logging.info(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

write_metrics(epochs, val_loss, val_accuracy)
write_to_redis("stats:model_accuracy", f"{val_accuracy:.2f}")
write_to_redis("training_complete", "true")
model_path = os.path.join(output_dir, 'best_model.pth')
torch.save(model.state_dict(), model_path)
logging.info(f"Saved trained model to {model_path}")
print("Training complete.")
print(f"Validation Accuracy: {val_accuracy:.2f}%")
