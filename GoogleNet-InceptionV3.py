# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 00:22:24 2024

@author: user
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

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
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Save class labels to labels.txt
class_labels = [str(i) for i in range(info.features['label'].num_classes)]
with open('labels_googlenet.txt', 'w') as file:
    file.write('\n'.join(class_labels))

# Load InceptionV3 model pre-trained on ImageNet
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Create a custom model by adding a GlobalAveragePooling2D layer and a Dense layer
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(120, activation='softmax')  # 120 classes in Stanford Dogs dataset
])

# Compile the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Train the model and save training history to training_history.txt
history = model.fit(train_ds, epochs=5, validation_data=test_ds)
with open('training_history_googlenet.txt', 'w') as file:
    file.write(str(history.history))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test Accuracy: {test_acc}')

# Save evaluation output to evaluation_output.txt
with open('evaluation_output_googlenet.txt', 'w') as file:
    file.write(f'Test Loss: {test_loss}\n')
    file.write(f'Test Accuracy: {test_acc}\n')

# Save the model to a file in h5 format
model.save('dog_classifier_model_googlenet_inceptionv3.h5')
