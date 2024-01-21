
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

stanford_dogs = tfds.load("stanford_dogs")

(train_data, test_data), info = tfds.load("stanford_dogs", split=["train[:80%]", "train[80%:]"], with_info=True)
batch_size = 64
img_size = 224

def preprocess_image(data):
    image = tf.image.resize(data['image'], (img_size, img_size))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    label = data['label']
    return image, label

train_data = train_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

img_shape = (img_size, img_size, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=20,
    validation_data=test_data
)

# Save Labels as txt File
with open('labels2.txt', 'w') as f:
    f.write('\n'.join(info.features['label'].names))

# Save Model
model.save('dog_breed_classifier_model2.h5')

# Save Training History to Text File
with open('training_history2.txt', 'w') as f:
    f.write(str(history.history))

# Evaluate the model on test data
evaluation_result = model.evaluate(test_data)
accuracy, loss = evaluation_result[1], evaluation_result[0]

# Save Evaluation Output to Text File
with open('evaluation_output2.txt', 'w') as f:
    f.write(f'Test Accuracy: {accuracy}\nTest Loss: {loss}')
