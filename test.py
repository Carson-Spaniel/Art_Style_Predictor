import os
import requests
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import io
from tqdm import tqdm
import concurrent.futures
import time

style = {
    12: 'Impressionism',
    21: 'Realism',
    4: 'Baroque',
    20: 'Post Impressionism',
    23: 'Romanticism',
    3: 'Art Nouveau',
    17: 'Northern Renaissance',
    24: 'Symbolism',
    15: 'Naive Art Primitivism',
    9: 'Expressionism',
    7: 'Cubism',
    2: 'Analytical Cubism',
    10: 'Abstract Expressionism',
    0: 'Abstract Expressionism',
    18: 'Pointillism',
}

# Function to preprocess data
def preprocess_data(data):
    img_url = data['row']['image']['src']
    response = requests.get(img_url)
    
    if response.status_code != 200:
        print("Failed to fetch image:", response.status_code)
        return None, None
    
    img = image.img_to_array(image.load_img(io.BytesIO(response.content), target_size=(150, 150)))
    img = img / 255.0
    label = data['row']['style']
    return img, label

# Fetch data
def fetch_data(offset, length):
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "huggan/wikiart",
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        return None

# Load data
def load_data(num_samples):
    offset = 0
    data = []
    pbar = tqdm(total=num_samples, desc="Loading data")
    while len(data) < num_samples:
        batch_size = min(num_samples - len(data), 100)
        batch_data = fetch_data(offset, batch_size)
        if batch_data is None:
            continue
        data.extend(batch_data['rows'])
        offset += batch_size
        pbar.update(batch_size)
    pbar.close()
    return data

model_path = 'art_style_predictor_model.h5'

# Try to load the model if it exists
try:
    print("Trying to load the model...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    from label import label_map
except IOError:
    print("Model not found. Defining and training a new model.")

    # Load and preprocess data
    num_samples = 30000
    data = load_data(num_samples)
    if data is None:
        exit()

    def process_row(row):
        img_array, label = preprocess_data(row)
        return img_array, label

    X = []
    y = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, row) for row in data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            img_array, label = future.result()
            if img_array is not None and label is not None:
                X.append(img_array)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    f = open("label.py", "w")
    f.write(f"label_map = {label_map}")
    f.close()

    y_mapped = np.array([label_map[label] for label in y])

    X_train, X_val, y_train, y_val = train_test_split(X, y_mapped, test_size=0.2, random_state=42)

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

    # Define and compile the model with increased dropout
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.6),  # Increased dropout rate
        Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Training the model...")
    history = model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=[early_stopping])

    model.save(model_path)
    print("Model saved.")

    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(val_generator)
    print('Test accuracy:', test_acc)

    # Plot training history if training was done
    if 'history' in locals():
        print("Plotting training history...")
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

# Function to preprocess a new image
def preprocess_new_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the style of a new image
def predict_top_styles(img_path, top_n=-1):
    img_array = preprocess_new_image(img_path)
    prediction = model.predict(img_array)
    top_n_indices = np.argsort(prediction[0])[::-1][:top_n]
    top_n_styles = [(idx, prediction[0][idx]) for idx in top_n_indices]
    return top_n_styles

# Function to get the human-readable style names
def get_style_names(label_map, top_n_styles):
    style_names = []
    for idx, prob in top_n_styles:
        for label, label_idx in label_map.items():
            if label_idx == idx:
                try:
                    name = style[label]
                except:
                    name = label
                style_names.append((name, prob))
                break
    return style_names

def list_all_styles(img_path):
    top_styles = predict_top_styles(img_path)
    style_names = get_style_names(label_map, top_styles)
    print(f"Predicted art styles for {img_path}:")
    for style_name, prob in style_names:
        print(f"\t{style_name}: {round(prob, 2)}")

# Example usage:
img_paths = ['impressionism.jpg', 'france.jpg', 'cubism.jpg', 'romanticism.jpeg']
for img_path in img_paths:
    list_all_styles(img_path)
