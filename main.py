import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import deeplake
from tqdm import tqdm
import os

# File path for saving and loading the TensorFlow dataset
dataset_file = 'wiki_art_dataset'

# Check if the dataset file exists
if os.path.exists(dataset_file):
    # If the file exists, load the TensorFlow dataset
    print('Loading TensorFlow dataset from file...')
    dataset = tf.data.Dataset.load(dataset_file)
else:
    # If the file doesn't exist, load the dataset from Deep Lake and save it
    print('Loading art...')
    ds = deeplake.load('hub://activeloop/wiki-art')

    print('Creating TensorFlow dataset...')
    dataset = ds.tensorflow()

    # Save the TensorFlow dataset to a file
    print('Saving TensorFlow dataset to file...')
    dataset.save(dataset_file)

# Define a function to preprocess the data for TensorFlow
def preprocess_data(image, label, target_shape=(150, 150)):
    # Resize images to the target shape
    image = tf.image.resize(image, target_shape)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Define a function to get the size of a dataset
def get_dataset_size(dataset):
    size = 0
    for _ in tqdm(dataset, desc="Calculating dataset size", unit="sample"):
        size += 1
    return size

for data in dataset.take(1):
    print("Image:", data['images'].shape[0])
    print("Label:", data['labels'])

# Calculate the size of training and validation sets
# print('Calculating dataset size...')
dataset_size = get_dataset_size(dataset)

print('Calculating training size...')
train_size = int(0.8 * dataset_size)

print('Calculating validation size...')
validation_size = dataset_size - train_size


# Split the dataset into train and validation sets
print('Splitting training dataset...')
train_dataset = dataset.take(train_size)

print('Splitting validation dataset...')
validation_dataset = dataset.skip(train_size)


# Filter out None values
print('Filtering dataset...')
train_dataset = train_dataset.filter(lambda x: x['images'].shape[0] is not None)
validation_dataset = validation_dataset.filter(lambda x: x['images'].shape[0] is not None)


# Map the preprocess_data function to the train and validation splits
print('Creating training dataset...')
train_dataset = train_dataset.map(lambda x: preprocess_data(x['images'].shape,x['labels']))

print('Creating validation dataset...')
validation_dataset = validation_dataset.map(lambda x: preprocess_data(x['images'].shape,x['labels']))


# Define the CNN model
print('Creating model...')
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
    Dropout(0.5),
    Dense(len(ds.labels['style']), activation='softmax')
])

# Compile the model
print('Compiling model...')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

# Train the model and accumulate history
print('Training model...')
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
for epoch in tqdm(range(30), desc='Training Progress'):
    epoch_history = model.fit(
        train_dataset.batch(32),
        epochs=1,  # Train for 1 epoch at a time
        validation_data=validation_dataset.batch(32),
        verbose=0,  # Set verbose to 0 to disable default progress bar
    )
    # Append metrics for the current epoch to history
    history['accuracy'].append(epoch_history.history['accuracy'])
    history['val_accuracy'].append(epoch_history.history['val_accuracy'])
    history['loss'].append(epoch_history.history['loss'])
    history['val_loss'].append(epoch_history.history['val_loss'])

# Save the model
print('Saving model...')
model.save('art_style_model.h5')

# Evaluate the model
print('Evaluating model...')
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

def predict_art_style(img_path, model, class_indices):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_labels = {v: k for k, v in class_indices.items()}
    return class_labels[predicted_class]

# Example prediction
img_path = 'impressionism.jpg'
print('Predicting art style...')
predicted_style = predict_art_style(img_path, model, {i: label for i, label in enumerate(ds.labels['style'])})
print(f'The predicted art style is: {predicted_style}')
