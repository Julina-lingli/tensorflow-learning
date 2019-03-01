import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt

print("tf version", tf.__version__)


#导入mnist 手写数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('shape: ', x_train.shape)
print("train set: ", len(x_train))
print("test set: ", len(x_test))


#可视化
def visualize_figure(num, data, label):
    fig = plt.figure(figsize=(20,30))
    for i in range(num):
        ax = fig.add_subplot(1, num, i + 1, xticks=[], yticks=[])
        ax.imshow(data[i], cmap = 'gray')
        ax.set_title(str(label[i]))
    plt.show()
#visualize_figure(6, x_train, y_train)

def visualize_input(img, ax):
    ax.imshow(img, cmap = 'gray')
    width, height = img.shape
    print(img.shape)
    thresh = img.max()/2.5
    print(img.max())
    print(thresh)
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy = (y,x),
                        horizontalalignment = 'center',
                        verticalalignment = 'center',
        color = 'white' if img[x][y] < thresh else 'black')
    plt.show()

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
#visualize_input(x_train[0], ax)

#数据处理, 转换为二维向量，并进行归一化【0， 1】处理
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#对标签进行one-hot编码
num_classes = 10
batch_size = 128
epochs = 20

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#define the model
model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])