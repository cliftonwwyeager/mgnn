import csv
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Multiply, GlobalMaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import os
from sklearn.model_selection import train_test_split
import logging
import random

logging.basicConfig(level=logging.ERROR)

def load_malicious_hashes_from_csv(csv_path='/home/user/full.csv'):
    hashes = []
    if not csv_path or not os.path.exists(csv_path):
        logging.error(f"Invalid path to CSV: {csv_path}")
        return hashes
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 1:
                    hashes.append(row[1])
        return hashes
    except Exception as e:
        logging.error(f"Error reading SHA-256 hashes from {csv_path}: {str(e)}")
        return hashes

def convert_files_to_hashes(directory):
    file_hashes = []
    if not directory or not os.path.exists(directory):
        logging.error(f"Invalid directory path: {directory}")
        return file_hashes
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes.append(file_hash)
    return file_hashes

def binary_to_image(hash_value, data_dir, csv_path, image_dim=256):
    if len(hash_value) != 64:
        logging.error("Provided input data is not a valid SHA-256 hash.")
        return None
    malicious_hashes = load_malicious_hashes_from_csv(csv_path)
    benign_hashes = convert_files_to_hashes(data_dir)
    if hash_value not in malicious_hashes and hash_value not in benign_hashes:
        logging.error("Hash {hash_value} is not from an allowed source.")
        return None
    byte_sequence = None
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                content = f.read()
                if hashlib.sha256(content).hexdigest() == hash_value:
                    byte_sequence = content
                    break
    if byte_sequence is None:
        return None
    int_sequence = np.frombuffer(byte_sequence, dtype=np.uint8)
    desired_length = image_dim * image_dim
    if len(int_sequence) > desired_length:
        int_sequence = int_sequence[:desired_length]
    else:
        int_sequence = np.pad(int_sequence, (0, desired_length - len(int_sequence)), 'constant')
    image = int_sequence.reshape(image_dim, image_dim).astype(np.uint8)
    return image / 255.0

def load_samples(data_dir, csv_path=None, image_dim=256):
    x_data = []
    y_data = []
    malicious_hashes = load_malicious_hashes_from_csv(csv_path) if csv_path else []
    benign_hashes = convert_files_to_hashes(data_dir)
    for hash_value in malicious_hashes + benign_hashes:
        image = binary_to_image(hash_value, data_dir, csv_path, image_dim)
        if image is not None:
            x_data.append(image)
            y_data.append(1 if hash_value in malicious_hashes else 0)
    return np.array(x_data), np.array(y_data)

def GatedCNNBlock(filters, kernel_size, stride=(1, 1), dropout_rate=0.3):
    def block(x):
        conv = Conv2D(filters, kernel_size, padding='same', activation='relu', strides=stride)(x)
        conv = Dropout(dropout_rate)(conv)
        gate = Conv2D(filters, kernel_size, padding='same', activation='sigmoid', strides=stride)(x)
        gate = Dropout(dropout_rate)(gate)
        gated_output = Multiply()([conv, gate])
        return gated_output
    return block

def build_model(input_shape=(256, 256, 1), num_filters=32, kernel_size=(5,5), dropout_rate=0.3):
    input_tensor = Input(shape=input_shape)
    x = GatedCNNBlock(num_filters, kernel_size, dropout_rate=dropout_rate)(input_tensor)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model

def random_configuration(search_space):
    return {param: random.choice(values) for param, values in search_space.items()}

def evaluate(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return max(history.history['val_accuracy'])

def select_top(population, performances, top_k=5):
    top_indices = sorted(range(len(performances)), key=lambda i: performances[i], reverse=True)[:top_k]
    return [population[i] for i in top_indices]

def crossover(parent1, parent2, search_space):
    child = {}
    for param in search_space.keys():
        child[param] = random.choice([parent1[param], parent2[param]])
    return child

def mutate(config, search_space):
    param_to_mutate = random.choice(list(search_space.keys()))
    config[param_to_mutate] = random.choice(search_space[param_to_mutate])
    return config

def evolutionary_optimization(x_train, y_train, x_val, y_val, search_space, num_generations=10, population_size=20, top_k=5):
    population = [random_configuration(search_space) for _ in range(population_size)]
    for generation in range(num_generations):
        performances = []
        for config in population:
            model = build_model(num_filters=config['num_filters'], kernel_size=config['kernel_size'], dropout_rate=config['dropout_rate'])
            performance = evaluate(model, x_train, y_train, x_val, y_val, config['epochs'], config['batch_size'])
            performances.append(performance)
        top_performers = select_top(population, performances, top_k=top_k)
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(top_performers, 2)
            child = crossover(parent1, parent2, search_space)
            child = mutate(child, search_space)
            new_population.append(child)
        population = new_population
    best_config = select_top(population, performances, top_k=1)[0]
    return best_config

def main():
    data_dir = '/home/user/'
    csv_path = '/home/user/full.csv'
    image_dim = 256
    test_size = 0.2
    search_space = {
        'num_filters': [16, 32, 64],
        'kernel_size': [(3, 3), (5, 5)],
        'dropout_rate': [0.3, 0.4, 0.5],
        'batch_size': [32, 64],
        'epochs': [10, 20]
    }

    x_data, y_data = load_samples(data_dir, csv_path, image_dim)
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    best_config = evolutionary_optimization(x_train, y_train, x_val, y_val, search_space)
    logging.info(f"Best configuration: {best_config}")

    model = build_model(input_shape=(image_dim, image_dim, 1), 
                        num_filters=best_config['num_filters'], 
                        kernel_size=best_config['kernel_size'], 
                        dropout_rate=best_config['dropout_rate'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=best_config['epochs'], batch_size=best_config['batch_size'])

    loss, accuracy = model.evaluate(x_test, y_test)
    logging.info(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
