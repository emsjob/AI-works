'''
Emil Sjöberg

Solving fashion mnist using the pretrained network vgg16
'''
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, LeakyReLU
from keras.utils import to_categorical
from keras.callbacks import History, EarlyStopping
import cv2
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist
import tensorflow_datasets as tfds

# Load datset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Number of classes in fashion MNIST
num_classes = 10

# Shuffle data
shuffle_train = np.random.permutation(len(y_train))
shuffle_test = np.random.permutation(len(y_test))

x_train = x_train[shuffle_train, :]
y_train = y_train[shuffle_train]

x_test = x_test[shuffle_test, :]
y_test = y_test[shuffle_test]

# Reshape image data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))
x_train = x_train / 255
x_train = tf.image.resize_with_pad(x_train, 48, 48, antialias=False)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))
x_test = x_test / 255
x_test = tf.image.resize_with_pad(x_test, 48, 48, antialias=False)

# Split training data into training and validation data
x_validate = x_train[:10000]
y_validate = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]

# Reshape output vectors
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)
y_test = to_categorical(y_test)

# Load vgg16
vgg = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))

# Input layer
inp = vgg.input

# Make new layers
new_layer_1 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')
new_class_layer = Dense(num_classes, activation='softmax')

# Add the new layers
out = new_layer_1((vgg.layers[-5].output))
out = Flatten()(out)
out = Dropout(0.25)(out)
out = new_class_layer(out)

# Create a new model
model = Model(inp, out)

# Freeze every layer except last ones
for l, layer in enumerate(model.layers[:-4]):
  layer.trainable = False

# Make sure the last layers is not frozen
for l, layer in enumerate(model.layers[-4:]):
  layer.trainable = True

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
history = History()
training = model.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_validate, y_validate))

# Plot accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Evaluate model
_, acc = model.evaluate(x_test, y_test)
print("Accuracy = ", acc)
