# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:59:18 2019
@author: Sachin Kumar Manjhi
Building a DNN and CNN for MNIST Dataset
"""

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
#from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D


batch_size = 128
num_classes = 10
epochs = 15


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

#%% DNN Model 

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

dnn_model = Sequential()
dnn_model.add(Dense(512, activation='relu', input_shape=(784,)))
#dnn_model.add(Dense(512, activation='relu'))
dnn_model.add(Dense(512, activation='relu'))
dnn_model.add(Dense(num_classes, activation='softmax'))

dnn_model.summary()

# Setting the learning rate and momentum parameter.
learning_rate = 0.01
sgd = optimizers.SGD(lr=learning_rate, momentum=0.0)   # momentum=0.5, 0.9

dnn_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = dnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=1/6)

#history = dnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = dnn_model.evaluate(x_test, y_test, verbose=0)

print('Learning Rate:', learning_rate)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_model_history(history)

#%% CNN Model 

#batch_size = 1                            # 1, 32, 128, 1024
num_classes = 10
epochs = 15
learning_rate = 0.01                      # 0.001; 0.01; 0.05; 0.1.

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




cnn_model = Sequential()
cnn_model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))

batch_size_list = [1024, 128, 32, 1] 
learning_rate_list = [0.001, 0.01, 0.05, 0.1] 

for learning_rate in learning_rate_list:
#for batch_size in batch_size_list:
    #print('Learning Rate:', learning_rate)
    print('Batch Size:', batch_size)
    sgd = optimizers.SGD(lr=learning_rate, momentum=0.9)    
    cnn_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=sgd,
                      metrics=['accuracy'])
    history = cnn_model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=epochs,
                            verbose=0,
                            validation_data=(x_test, y_test))
    score = cnn_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

plot_model_history(history)

