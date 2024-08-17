import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
from deap import base, creator, tools, algorithms
import random
import urllib.request
import zipfile
import os
import redis
from mgnn import MGNN
from synthetic_data_generation import generate_obfuscated_samples, generate_random_obfuscation, generate_pseudo_code

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
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
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.astype(int)
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

def write_to_redis(key, value):
    """Helper function to write data to Redis."""
    r.set(key, value)

def evaluate(individual):
    hidden_dim = individual[0]
    learning_rate = individual[1]
    model = MGNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        scheduler.step()
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
    model = MGNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(epochs):
        scheduler.step()
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
model = MGNN(input_dim, best_hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(epochs):
    scheduler.step()
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
    write_to_redis(f"final_epoch_{epoch+1}_loss", epoch_loss)
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
    write_to_redis(f"final_epoch_{epoch+1}_val_loss", val_loss)
    write_to_redis(f"final_epoch_{epoch+1}_val_accuracy", val_accuracy)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

write_to_redis("training_complete", "true")
print("Training complete.")
