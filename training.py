## -*- coding: utf-8 -*-
import keras
from keras.utils import plot_model
from keras import metrics
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout,AvgPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
BATCH_SIZE = 32
EPOCHS = 64
# read data
data = pd.read_csv('K://DATA//python//face//traingingdata//set/fer2013_.csv')
print(data)
# merge

pixels = data['pixels']
emotions = data['emotion_p']

usages = data['Usage']

num_classes = 10
x_train, y_train, x_test, y_test = [], [], [], []

for emotion, img, usage in zip(emotions, pixels, usages):
    #zip_可迭代的对象作为参数,将对象中对应的元素打包成元组,返回由这些元组组成的列表
    try:
        emotion = keras.utils.to_categorical(emotion, num_classes)  # 独热编码
        #print(emotion)
        val = img.split(" ")
        pixels = np.array(val, 'float32')

        if (usage == 'Training' or usage == 'PublicTest'):
            x_train.append(pixels)
            y_train.append(emotion)
        elif (usage == 'PrivateTest'):
            x_test.append(pixels)
            y_test.append(emotion)
    except:
        print("", end="")

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(-1, 48, 48, 1)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape(-1, 48, 48, 1)

model = Sequential()

model.add(Conv2D(16, (5, 5), strides=1, padding='same', input_shape=(48, 48, 1))) #16维(5*5滤波器) SAME卷积n_o = n_i / s
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (1, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  #多维->一维 卷积过渡全连接 （GlobalAvgPool2D替代）

model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100 * train_score[1])
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100 * test_score[1])

print(history.history)
plt.plot(history.history["loss"], 'r-', label='loss')
plt.plot(history.history["accuracy"], 'b-', label='accu')
plt.title('Loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()


model.save("K://DATA//python//face//model/model_v1.3.h5")
#v1.0 68%
#v1.1 69%
#v1.2 71%
#v1.3 71%