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
        int_sequence = np.pad(int_sequence, (0, max(0, image_dim * image_dim - len(int_sequence))), 'constant')
        image = int_sequence[:image_dim * image_dim].reshape(image_dim, image_dim).astype(np.uint8)
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
                y_data.append(1 if file.endswith('.gen-*') else 0)  # Assuming .gen-* files are malicious
    return np.array(x_data), np.array(y_data)

data_dir = '/home/user/BazaarCollection'  # Adjust this path
x_data, y_data = load_samples(data_dir)
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

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

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
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
except Exception as e:
    print(f"Error during model training: {str(e)}")

directory_to_scan = '/home/user/files'  # Adjust this to the directory you want to scan
scan_results = scan_directory_for_malware(directory_to_scan, model)

for filepath, result in scan_results.items():
    print(f"{filepath}: {result}")
