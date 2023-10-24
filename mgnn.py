import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Multiply, Flatten, Dense
import os
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def binary_to_image(binary_file_path, image_dim=256):
    try:
        with open(binary_file_path, 'rb') as f:
            byte_sequence = f.read()
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
        logging.error(f"Error converting binary to image for {binary_file_path}: {str(e)}")
        return None

def load_samples(data_dir, image_dim=256):
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

data_dir = '/home/user/BazaarCollection'  # Adjust this path

def load_samples_generator(data_dir, image_dim=256, batch_size=32, file_limit=10000):
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


def GatedCNNBlock(filters, kernel_size):
    def block(x):
        conv = Conv2D(filters, kernel_size, padding='same')(x)
        gate = Conv2D(filters, kernel_size, padding='same', activation='sigmoid')(x)
        return Multiply()([conv, gate])
    return block

input_tensor = Input(shape=(256, 256, 1))
x = GatedCNNBlock(32, (3,3))(input_tensor)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_tensor, outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for batch_x, batch_y in load_samples_generator(data_dir, file_limit=10000):
        model.fit(batch_x, batch_y, validation_data=(x_val, y_val), epochs=10, batch_size=32)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

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

model.save("/home/user/mgnn")

try:
    for batch_x, batch_y in load_samples_generator(data_dir, file_limit=10000):
        model.fit(batch_x, batch_y, validation_data=(x_val, y_val), epochs=10, batch_size=32)
except Exception as e:
    print(f"Error during model training: {str(e)}")

directory_to_scan = '/home/user/'  # Adjust this to the directory you want to scan
scan_results = scan_directory_for_malware(directory_to_scan, model)

for filepath, result in scan_results.items():
    print(f"{filepath}: {result}")

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
