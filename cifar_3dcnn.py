# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:46:37 2019
@author: Sachin Kumar Manjhi
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.
Assignment 2 Solution 
Deep Learning Course 
IISC Bangalore
Building multiple CNN for CIFAR Dataset
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD

# %% Plot Model History
import matplotlib.pyplot as plt
import numpy as np
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
# %% Train CNN Model

batch_size = 32
num_classes = 10
epochs = 15

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
colors = x_train.shape[3]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, colors, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, colors, 1)
input_shape = (img_rows, img_cols, colors, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# CNN Training parameters
batch_size = 256   #256
nb_classes = 10
nb_epoch = 10
learning_rate = 0.01   #0.05


model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3),activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

sgd=SGD(lr=learning_rate)


model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=1/6.25)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#plot_model_history(history)
