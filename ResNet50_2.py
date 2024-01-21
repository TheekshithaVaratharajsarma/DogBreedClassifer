# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:00:25 2024

@author: user
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

stanford_dogs = tfds.load("stanford_dogs")

(train_data, test_data), info = tfds.load("stanford_dogs", split=["train[:80%]", "train[80%:]"], with_info=True)

batch_size = 32
img_size = 224

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

def preprocess_image(data):
    image = tf.image.resize(data['image'], (img_size, img_size))
    image = data_augmentation(image)
    image = preprocess_input(image)
    label = data['label']
    return image, label

train_data = train_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

img_shape = (img_size, img_size, 3)

base_model = ResNet50(input_shape=img_shape, include_top=False, weights='imagenet')

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

model.save('dog_breed_classifier_model_updated.h5')
with open('training_history_updated.txt', 'w') as f:
    f.write(str(history.history))

evaluation_result = model.evaluate(test_data)
accuracy, loss = evaluation_result[1], evaluation_result[0]

with open('evaluation_output_updated.txt', 'w') as f:
    f.write(f'Test Accuracy: {accuracy}\nTest Loss: {loss}')

