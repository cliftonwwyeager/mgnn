import csv
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Multiply, GlobalMaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import os
from sklearn.model_selection import train_test_split
import logging
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam
import requests
import zipfile
import io

logging.basicConfig(level=logging.ERROR)

def download_and_extract_csv(url, extract_to='/home/user/full.csv'):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(path=os.path.dirname(extract_to))
                for filename in z.namelist():
                    if filename.endswith('.csv'):
                        os.rename(os.path.join(os.path.dirname(extract_to), filename), extract_to)
                        break
            logging.info(f"Downloaded and extracted CSV to {extract_to}")
        else:
            logging.error(f"Failed to download file from {url}, status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error downloading or extracting CSV: {str(e)}")

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

def load_and_preprocess_data(directory, img_size=(256, 256), batch_size=32, validation_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    train_generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, validation_generator

def GatedCNNBlock(filters, kernel_size, stride=(1, 1), dropout_rate=0.3):
    def block(x):
        conv = Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
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

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    train_gen, val_gen = load_and_preprocess_data(data_dir)
    model = build_model((256, 256, 3))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min'),
        LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
    ]
    model.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=callbacks)

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

def run_evolutionary_optimization_generation(x_train, y_train, x_val, y_val, search_space):
    population_size = 20
    population = [random_configuration(search_space) for _ in range(population_size)]
    performances = []
    for config in population:
        model = build_model(num_filters=config['num_filters'], kernel_size=config['kernel_size'], dropout_rate=config['dropout_rate'])
        performance = evaluate(model, x_train, y_train, x_val, y_val, config['epochs'], config['batch_size'])
        performances.append(performance)
    best_index = performances.index(max(performances))
    best_config = population[best_index]
    best_performance = performances[best_index]
    return best_config, best_performance

def evolutionary_optimization_with_feedback(x_train, y_train, x_val, y_val, initial_search_space, max_generations=100, performance_threshold=0.01):
    search_space = initial_search_space.copy()
    last_best_performance = 0
    generation = 0
    while generation < max_generations:
        best_config, best_performance = run_evolutionary_optimization_generation(x_train, y_train, x_val, y_val, search_space)
        if abs(best_performance - last_best_performance) < performance_threshold:
            print(f"Optimization converged at generation {generation} with performance {best_performance}")
            break
        last_best_performance = best_performance
        adjust_search_space_based_on_performance(search_space, best_config)
        generation += 1

def analyze_file(file_path, data_dir):
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
    image = binary_to_image(file_path)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    model = load_model(os.path.join(data_dir, 'best_model.h5'))
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    result = {
        'file_path': file_path,
        'predicted_class': int(predicted_class),
        'confidence': float(predictions[0][predicted_class])
    }
    return result

def main(data_dir):
    data_augmentation = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    learning_rate_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
    csv_path = os.path.join(data_dir, 'full.csv')
    image_dim = 256
    test_size = 0.2
    search_space = {
        'num_filters': [16, 32, 64],
        'kernel_size': [(3, 3), (5, 5)],
        'dropout_rate': [0.3, 0.4, 0.5],
        'batch_size': [32, 64],
        'epochs': [10, 20]
    }

    download_and_extract_csv('https://bazaar.abuse.ch/export/csv/full/', csv_path)
    x_data, y_data = load_samples(data_dir, csv_path, image_dim)
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=test_size, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    best_config = evolutionary_optimization_with_feedback(x_train, y_train, x_val, y_val, search_space)
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
    data_dir = '/path/to/upload/folder'
    main(data_dir)
