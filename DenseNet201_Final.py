# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:46:44 2024

@author: user
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Loading Stanford Dogs dataset from TensorFlow Datasets
(train_ds, test_ds), info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True
)

# Defining image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Preprocesings images and creating batches
def preprocess_img(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.keras.applications.densenet.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Saving class labels to labels.txt
class_labels = [str(i) for i in range(info.features['label'].num_classes)]
with open('labels.txt', 'w') as file:
    file.write('\n'.join(class_labels))

# Loading DenseNet201 model pre-trained on ImageNet
base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freezing convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Creating a custom model by adding a GlobalAveragePooling2D layer and a Dense layer
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(120, activation='softmax')  # 120 classes in Stanford Dogs dataset
])

# Compiling the model
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])


# Training the model and saving training history to training_history.txt
history = model.fit(train_ds, epochs=5, validation_data=test_ds)
with open('training_history.txt', 'w') as file:
    file.write(str(history.history))

# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test Accuracy: {test_acc}')

# Saving evaluation output to evaluation_output.txt
with open('evaluation_output.txt', 'w') as file:
    file.write(f'Test Loss: {test_loss}\n')
    file.write(f'Test Accuracy: {test_acc}\n')

# Saving the model to a file in h5 format
model.save('dog_classifier_model.h5')
