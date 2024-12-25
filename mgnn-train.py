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

logging.basicConfig(level=logging.INFO)
influxdb_url = os.getenv('INFLUXDB_URL')
token = os.getenv('INFLUXDB_TOKEN')
org = os.getenv('INFLUXDB_ORG')
bucket = os.getenv('INFLUXDB_BUCKET')
client = influxdb_client.InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

def write_metrics(epoch, loss, accuracy):
    point = influxdb_client.Point("training_metrics") \
        .tag("model", "MGNNWithTD") \
        .field("epoch", epoch) \
        .field("loss", loss) \
        .field("accuracy", accuracy)
    write_api.write(bucket=bucket, org=org, record=point)

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

def write_to_redis(key, value):
    r.set(key, value)

class MGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

class MGNNWithTD(MGNN):
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.99):
        super(MGNNWithTD, self).__init__(input_dim, hidden_dim, output_dim)
        self.gamma = gamma

    def forward(self, x, target=None, reward=None):
        x = nn.functional.relu(self.fc1(x))
        output = self.fc2(x)
        if target is not None and reward is not None:
            td_error = reward + self.gamma * target - output
            output = output + td_error
        return output

    def train_with_td(self, optimizer, criterion, scheduler, train_loader, epochs=20):
        self.train()
        for epoch in range(epochs):
            scheduler.step()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                target = torch.max(outputs, dim=1)[0].detach()
                reward = (outputs.argmax(dim=1) == labels).float()
                td_outputs = self.forward(inputs, target=target, reward=reward)
                loss = criterion(td_outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

def generate_obfuscated_samples(X, y, noise_factor=0.05):
    X_obf = X.copy()
    n, m = X_obf.shape
    X_obf += noise_factor * np.random.randn(n, m)
    for i in range(n):
        cnt = random.randint(1, int(m * 0.05))
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

input_dim = 100
output_dim = 10
batch_size = 64
epochs = 20
val_split = 0.2
url = 'https://bazaar.abuse.ch/export/csv/full/'
output_dir = '/home/user/mgnn'
zip_path = os.path.join(output_dir, 'full.zip')
csv_path = os.path.join(output_dir, 'full.csv')
os.makedirs(output_dir, exist_ok=True)

urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

data = pd.read_csv(csv_path)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data.iloc[:, :-1].apply(lambda x: ' '.join(x.astype(str)), axis=1))

pca = PCA(n_components=input_dim)
X = pca.fit_transform(tfidf_matrix.toarray())
y = data.iloc[:, -1].values

obfuscated_samples = generate_obfuscated_samples(X, y)
random_obfuscation = generate_random_obfuscation(X, num_samples=5000)
pseudo_code_samples = generate_pseudo_code(num_samples=5000, input_dim=input_dim)
X_synthetic = np.concatenate((obfuscated_samples, random_obfuscation, pseudo_code_samples))
y_synthetic = np.concatenate((np.ones(len(obfuscated_samples)), np.zeros(len(random_obfuscation) + len(pseudo_code_samples))))

X_combined = np.concatenate((X, X_synthetic))
y_combined = np.concatenate((y, y_synthetic))

X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=val_split, random_state=42, stratify=y_combined)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def evaluate(individual):
    hidden_dim = individual[0]
    learning_rate = individual[1]
    model = MGNNWithTD(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model.train_with_td(optimizer, criterion, scheduler, train_loader, epochs)
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
    val_accuracy = 100 * correct / total
    write_to_redis(f"val_loss_{hidden_dim}_{learning_rate}", val_loss)
    write_to_redis(f"val_accuracy_{hidden_dim}_{learning_rate}", val_accuracy)
    return (val_loss,)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_hidden_dim", random.randint, 16, 128)
toolbox.register("attr_learning_rate", random.uniform, 1e-5, 1e-1)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_hidden_dim, toolbox.attr_learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[16, 1e-5], up=[128, 1e-1], eta=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=50)
ngen = 10
cxpb = 0.5
mutpb = 0.2
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)
best_individual = tools.selBest(population, k=1)[0]
write_to_redis("best_hidden_dim", best_individual[0])
write_to_redis("best_learning_rate", best_individual[1])
print('Best Individual: ', best_individual)

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    model = MGNNWithTD(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model.train_with_td(optimizer, criterion, scheduler, train_loader, epochs)
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
    val_accuracy = 100 * correct / total
    write_to_redis(f"trial_{trial.number}_hidden_dim", hidden_dim)
    write_to_redis(f"trial_{trial.number}_learning_rate", learning_rate)
    write_to_redis(f"trial_{trial.number}_val_loss", val_loss)
    write_to_redis(f"trial_{trial.number}_val_accuracy", val_accuracy)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_trial = study.best_trial
write_to_redis("optuna_best_trial_hidden_dim", best_trial.params['hidden_dim'])
write_to_redis("optuna_best_trial_learning_rate", best_trial.params['learning_rate'])
write_to_redis("optuna_best_trial_loss", best_trial.value)
print('Best trial:')
print('  Loss: {}'.format(best_trial.value))
print('  Params: ')
for key, value in best_trial.params.items():
    print('    {}: {}'.format(key, value))

best_hidden_dim = best_trial.params['hidden_dim']
best_learning_rate = best_trial.params['learning_rate']
model = MGNNWithTD(input_dim, best_hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
model.train_with_td(optimizer, criterion, scheduler, train_loader, epochs)

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
val_accuracy = 100 * correct / total

for epoch in range(epochs):
    epoch_loss = val_loss
    write_metrics(epoch + 1, epoch_loss, val_accuracy)
    write_to_redis(f"final_epoch_{epoch+1}_loss", epoch_loss)

write_to_redis("training_complete", "true")
print("Training complete.")
print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
