from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import hashlib
import redis
from werkzeug.utils import secure_filename
import numpy as np
import json
import csv
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mgnn import MGNN
import optuna
import urllib.request
import zipfile
from deap import base, creator, tools, algorithms
import random

app = Flask(__name__)
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
redis_password = os.getenv('REDIS_PASSWORD', None)
redis_client = redis.StrictRedis(host=redis_host, port=int(redis_port), password=redis_password, db=0)
input_dim = 100
output_dim = 10
batch_size = 64
epochs = 20
learning_rate = 0.001
val_split = 0.2
output_dir = '/home/user/mgnn'
os.makedirs(output_dir, exist_ok=True)

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
        file_path = os.path.join(output_dir, filename)
        file.save(file_path)
        hash_value = calculate_hash(file_path)
        X = process_file(file_path)
        model = MGNN(input_dim, best_hidden_dim, output_dim)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(X_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1).max().item()
        redis_client.set(hash_value, json.dumps({'predicted_class': predicted_class, 'confidence': confidence}))
        return jsonify({'predicted_class': predicted_class, 'confidence': confidence})

@app.route('/process', methods=['POST'])
def process():
    train_model()
    return jsonify({'status': 'Model training complete'})

@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    if request.method == 'GET':
        return render_template('confirm.html')
    data = request.json
    hash_value = data.get('hash')
    is_correct = data.get('is_correct')
    filename = data.get('filename')
    
    if not hash_value or is_correct is None or not filename:
        return jsonify({'error': 'Invalid input'})

    prediction = redis_client.get(hash_value)
    if not prediction:
        return jsonify({'error': 'No prediction found for the provided hash'})

    prediction = json.loads(prediction)

    csv_file = os.path.join(output_dir, 'confirmations.csv')
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['filename', 'hash', 'predicted_class', 'confidence', 'is_correct', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'filename': filename,
            'hash': hash_value,
            'predicted_class': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        })

    if is_correct:
        submit_positive_result(hash_value, prediction['predicted_class'])

    redis_client.set(f"confirmed:{hash_value}", json.dumps(prediction))
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
    confirmation_data = [json.loads(record) for record in accurate_predictions]
    return confirmation_data

def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()

def process_file(file_path):
    # Load and preprocess the file
    data = pd.read_csv(file_path)
    X = data.values
    X = StandardScaler().fit_transform(X)
    return X

def load_new_data(file_hash, predicted_class, data_dir):
    file_path = os.path.join(data_dir, f"{file_hash}.csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        X = data.values
        y = np.full((X.shape[0],), predicted_class, dtype=np.int64)
        return X, y
    return None, None

def add_new_data_to_training_set(data_dir, confirmation_data):
    new_X = []
    new_y = []
    for data in confirmation_data:
        file_hash = data['hash']
        predicted_class = data['predicted_class']
        X, y = load_new_data(file_hash, predicted_class, data_dir)
        if X is not None:
            new_X.append(X)
            new_y.append(y)
    if new_X:
        new_X = np.vstack(new_X)
        new_y = np.concatenate(new_y)
        return new_X, new_y
    return None, None

def reinforcement_learning_update(data_dir, confirmation_data, model, existing_X, existing_y):
    new_X, new_y = add_new_data_to_training_set(data_dir, confirmation_data)
    if new_X is not None:
        combined_X = np.vstack([existing_X, new_X])
        combined_y = np.concatenate([existing_y, new_y])
    else:
        combined_X = existing_X
        combined_y = existing_y

    train_dataset = TensorDataset(torch.tensor(combined_X, dtype=torch.float32), torch.tensor(combined_y, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(data_dir, 'best_model_updated.pth'))

def train_model():
    url = 'https://bazaar.abuse.ch/export/csv/full/'
    zip_path = os.path.join(output_dir, 'full.zip')
    csv_path = os.path.join(output_dir, 'full.csv')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42, stratify=y)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def evaluate(individual):
        hidden_dim = individual[0]
        learning_rate = individual[1]

        model = MGNN(input_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

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
    toolbox.register("mutate", tools.mutPolynomialBounded, low=[16, 1e-5], up=[128, 1e-1], eta=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=50)
    ngen = 10
    cxpb = 0.5
    mutpb = 0.2

    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                        stats=None, halloffame=None, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    print('Best Individual: ', best_individual)

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

        model = MGNN(input_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

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
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_trial = study.best_trial
    print('Best trial:')
    print('  Loss: {}'.format(best_trial.value))
    print('  Params: ')
    for key, value in best_trial.params.items():
        print('    {}: {}'.format(key, value))
    best_hidden_dim = best_trial.params['hidden_dim']
    best_learning_rate = best_trial.params['learning_rate']
    model = MGNN(input_dim, best_hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
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
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    print("Training complete.")

if __name__ == '__main__':
    app.run(debug=True)
