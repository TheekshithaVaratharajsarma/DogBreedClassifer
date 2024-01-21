# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:02:03 2024

@author: user
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Stanford Dogs dataset from TensorFlow Datasets
(train_ds, test_ds), info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True
)

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Preprocess images and create batches
def preprocess_img(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Save class labels to labels.txt
class_labels = [str(i) for i in range(info.features['label'].num_classes)]
with open('labels_DENSENET_MODIFIED.txt', 'w') as file:
    file.write('\n'.join(class_labels))

# Load DenseNet201 model pre-trained on ImageNet
base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model by adding a GlobalAveragePooling2D layer, Dropout, and a Dense layer
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(120, activation='softmax')  # 120 classes in Stanford Dogs dataset
])

# Compile the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Data Augmentation using map
def apply_data_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

train_ds_augmented = train_ds.map(apply_data_augmentation)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with augmented data and early stopping
history = model.fit(train_ds_augmented, epochs=10, validation_data=test_ds, callbacks=[early_stopping])
with open('training_history_DENSENET_MODIFIED.txt', 'w') as file:
    file.write(str(history.history))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test Accuracy: {test_acc}')

# Save evaluation output to evaluation_output_DENSENET_MODIFIED.txt
with open('evaluation_output_DENSENET_MODIFIED.txt', 'w') as file:
    file.write(f'Test Loss: {test_loss}\n')
    file.write(f'Test Accuracy: {test_acc}\n')

# Save the model to a file in h5 format
model.save('dog_classifier_model_DENSENET_MODIFIED.h5')
