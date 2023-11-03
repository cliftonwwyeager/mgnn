import csv
import hashlib
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Multiply, GlobalMaxPooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import logging


def load_malicious_hashes_from_csv(csv_path):
    hashes = ['/home/user/full.csv']
    if not csv_path or not os.path.exists(csv_path):
        raise ValueError(f"Invalid path to CSV: {csv_path}")
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 1:
                    hashes.append(row[1])
        return hashes
    except Exception as e:
        raise RuntimeError(f"Error reading SHA-256 hashes from {csv_path}: {str(e)}")



def convert_files_to_hashes(directory):
    file_hashes = ['/home/user/stuff']
    if not directory or not os.path.exists(directory):
        raise ValueError(f"Invalid directory path: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                content = f.read()
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes.append(file_hash)
    return file_hashes





def binary_to_image(hash_value, data_dir, csv_path=None, image_dim=256):
    if len(hash_value) != 64:  # Not a SHA-256 hash
        logging.error(f"Provided input data is not a valid SHA-256 hash.")
        return None

    malicious_hashes = load_malicious_hashes_from_csv(csv_path) if csv_path else []
    benign_hashes = convert_files_to_hashes(data_dir)

    # Check if hash_value is either from malicious list or benign list
    if hash_value not in malicious_hashes and hash_value not in benign_hashes:
        logging.error(f"Hash {hash_value} is not from an allowed source.")
        return None

    try:
        # Locate the corresponding binary file in the directory based on the hash
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    if hashlib.sha256(content).hexdigest() == hash_value:
                        byte_sequence = content
                        break

        int_sequence = np.frombuffer(byte_sequence, dtype=np.uint8)

        # Truncate or pad the int_sequence to match the desired image dimensions
        desired_length = image_dim * image_dim
        if len(int_sequence) > desired_length:
            int_sequence = int_sequence[:desired_length]
        else:
            int_sequence = np.pad(int_sequence, (0, desired_length - len(int_sequence)), 'constant')

        image = int_sequence.reshape(image_dim, image_dim).astype(np.uint8)
        return image / 255.0
    except Exception as e:
        logging.error(f"Error converting hash {hash_value} to image: {str(e)}")
        return None

def load_samples(data_dir, csv_path='/home/user/full.csv', image_dim=256):
    x_data = []
    y_data = []
    malicious_hashes = load_malicious_hashes_from_csv(csv_path) if csv_path else []
    benign_hashes = convert_files_to_hashes(data_dir)

    # Process malicious samples
    for hash_value in malicious_hashes:
        image = binary_to_image(hash_value, data_dir=data_dir, image_dim=image_dim)
        if image is not None:
            x_data.append(image)
            y_data.append(1)  # Mark as malicious

    # Process benign samples
    for hash_value in benign_hashes:
        image = binary_to_image(hash_value, data_dir=data_dir, image_dim=image_dim)
        if image is not None:
            x_data.append(image)
            y_data.append(0)  # Mark as benign

    return np.array(x_data), np.array(y_data)

    x_data = []
    y_data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            image = binary_to_image(full_path, image_dim)
            if image is not None:
                x_data.append(image)
                y_data.append(1 if file.endswith('.*-*') else 0)  # Assuming .*-* files are malicious
    return np.array(x_data), np.array(y_data)

data_dir = '/home/user/stuff'  # Adjust this path


def load_samples_generator(data_dir, csv_path='/home/user/full.csv', image_dim=256, batch_size=32, file_limit=10000):
    x_data = []
    y_data = []
    file_count = 0
    malicious_hashes = load_malicious_hashes_from_csv(csv_path) if csv_path else []
    benign_hashes = convert_files_to_hashes(data_dir)

    # Process malicious samples
    for hash_value in malicious_hashes:
        image = binary_to_image(hash_value, data_dir=data_dir, image_dim=image_dim)
        if image is not None:
            x_data.append(image)
            y_data.append(1)  # Mark as malicious
            file_count += 1
            if file_count >= file_limit:
                yield np.array(x_data), np.array(y_data)
                x_data, y_data, file_count = [], [], 0

    # Process benign samples
    for hash_value in benign_hashes:
        image = binary_to_image(hash_value, data_dir=data_dir, image_dim=image_dim)
        if image is not None:
            x_data.append(image)
            y_data.append(0)  # Mark as benign
            file_count += 1
            if file_count >= file_limit:
                yield np.array(x_data), np.array(y_data)
                x_data, y_data, file_count = [], [], 0

    x_data = []
    y_data = []
    file_count = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file_limit and file_count >= file_limit:
                yield np.array(x_data), np.array(y_data)
                x_data, y_data = [], []
                return
            full_path = os.path.join(root, file)
            image = binary_to_image(full_path, image_dim)
            if image is not None:
                x_data.append(image)
                y_data.append(1 if file.endswith('.*-*') else 0)
                file_count += 1
                if len(x_data) == batch_size:
                    yield np.array(x_data), np.array(y_data)
                    x_data, y_data = [], []
    if x_data:
        yield np.array(x_data), np.array(y_data)
        
x_data, y_data = next(load_samples_generator(data_dir, file_limit=10000))
x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Ensure data shape compatibility
x_train = x_train.reshape(-1, 256, 256, 1)
x_val = x_val.reshape(-1, 256, 256, 1)
x_test = x_test.reshape(-1, 256, 256, 1)


def GatedCNNBlock(filters, kernel_size, stride=(1,1)):
    def block(x):
        # Original convolution layer with ReLU activation
        conv = Conv2D(filters, kernel_size, padding='same', activation='relu', strides=stride)(x)
        
        # Parallel convolution layer with sigmoid activation for gating
        gate = Conv2D(filters, kernel_size, padding='same', activation='sigmoid', strides=stride)(x)
        
        # Multiply the ReLU output with the sigmoid output for gating
        gated_output = Multiply()([conv, gate])
        
        return gated_output
    return block

# Define the enhanced network architecture
input_tensor = Input(shape=(256, 256, 1))
x = GatedCNNBlock(32, (5,5), stride=(2,2))(input_tensor)
x = GlobalMaxPooling2D()(x)
x = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_tensor, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Data Augmentation using TensorFlow's ImageDataGenerator
data_gen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.8, 1.2)
)

# Augmented data generator
augmented_data_gen = data_gen.flow(x_train, y_train, batch_size=32)

model.fit(augmented_data_gen, validation_data=(x_val, y_val), epochs=10, steps_per_epoch=math.ceil(len(x_train)/32))

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

try:
    model.fit(augmented_data_gen, validation_data=(x_val, y_val), epochs=10, steps_per_epoch=len(x_train)//32)
except Exception as e:
    print(f"Error during model training: {str(e)}")

import random
import tensorflow as tf
search_space = {'learning_rate': [0.001, 0.0001, 1e-05], 'batch_size': [16, 32, 64], 'num_filters': [16, 32, 64], 'kernel_size': [(3, 3), (5, 5)], 'epochs': [5, 10, 15]}
def random_configuration():
    return {param: random.choice(values) for param, values in search_space.items()}

def evaluate(config, x_train, y_train, x_val, y_val):
    model = build_model(config['num_filters'], config['kernel_size'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                        epochs=config['epochs'], batch_size=config['batch_size'], verbose=0)
    # Return the best validation accuracy
    return max(history.history['val_accuracy'])

def select_top(population, performances, top_k=5):
    # Select the top-k performing configurations
    return [population[i] for i in sorted(range(len(performances)), key=lambda i: performances[i])[-top_k:]]

def crossover(parent1, parent2):
    child = {}
    for param in search_space.keys():
        child[param] = random.choice([parent1[param], parent2[param]])
    return child

def mutate(config):
    mutated_config = config.copy()
    # Randomly choose a hyperparameter to mutate
    param_to_mutate = random.choice(list(search_space.keys()))
    mutated_config[param_to_mutate] = random.choice(search_space[param_to_mutate])
    return mutated_config

def build_model(num_filters, kernel_size):
    input_tensor = tf.keras.Input(shape=(256, 256, 1))
    x = GatedCNNBlock(num_filters, kernel_size)(input_tensor)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return model

def evolutionary_optimization(x_train, y_train, x_val, y_val, num_generations=10, population_size=20):
    # Initialize population
    population = [random_configuration() for _ in range(population_size)]

    for generation in range(num_generations):
        # Evaluate performance for each configuration
        performances = [evaluate(config, x_train, y_train, x_val, y_val) for config in population]

        # Select top-performers
        top_performers = select_top(population, performances)

        # Create new population through crossover and mutation
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(top_performers), random.choice(top_performers)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Get the best configuration after all generations
    best_config = select_top(population, performances)[0]
    return best_config

def scan_directory_for_malware(directory_path, model, image_dim=256):
    results = {}
    for root, _, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                image_representation = binary_to_image(full_path, image_dim)
                if image_representation is not None:
                    prediction = model.predict(image_representation.reshape(1, image_dim, image_dim, 1))
                    is_malware = prediction[0][0] > 0.5
                    results[full_path] = "Malicious" if is_malware else "Benign"
                else:
                    results[full_path] = "Error: Could not convert to image"
            except Exception as e:
                results[full_path] = f"Error: {str(e)}"
    return results


def prepare_data(data_dir, csv_path=None, image_dim=256, test_size=0.4):
    x_data, y_data = next(load_samples_generator(data_dir, csv_path, file_limit=10000))
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Ensure data shape compatibility
    x_train = x_train.reshape(-1, image_dim, image_dim, 1)
    x_val = x_val.reshape(-1, image_dim, image_dim, 1)
    x_test = x_test.reshape(-1, image_dim, image_dim, 1)

    return x_train, y_train, x_val, y_val, x_test, y_test
    print(f"{filepath}: {result}")
    
model.save("/home/user/mgnn")
