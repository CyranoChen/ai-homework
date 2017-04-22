#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:53:43 2017

@author: cyrano
"""
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model

def VGG_16(weights_path = None):    
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1), input_shape=(32,32,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
        
    if (weights_path):
        model.load_weights(weights_path)

    return model


def VGG_19(weights_path = None):    
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1), input_shape=(32,32,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
        
    if (weights_path):
        model.load_weights(weights_path)

    return model

# 开始下载数据集
t0 = time.time()  # 打开深度学习计时器

# CIFAR100 图片数据集
(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()  # 32×32


X_train = X_train.astype('float32')  # uint8-->float32
X_test = X_test.astype('float32')

#X_train = Reshape(X_train, [-1,224,224,3])
#X_test = Reshape(X_test, [-1,224,224,3])

X_train /= 255  # 归一化到0~1区间
X_test /= 255

print('训练样例:', X_train.shape, Y_train.shape, ', 测试样例:', X_test.shape, Y_test.shape)

nb_classes = 100
# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

print("取数据耗时: ",(time.time() - t0), "s ..." )

# Train pretrained model
batch = 128
epochs = 1

model = VGG_16('vgg16_weights.h5')
plot_model(model, to_file='Homework4-vgg16-keras.png', show_shapes=True) 

model.summary() # 模型小节

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-8)
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
#result = model.fit(X_train, Y_train, batch, epochs, shuffle=True, validation_data=(X_test, Y_test))
#
#import h5py 
#from keras.models import model_from_json 
# 
#json_string = model.to_json()  
#open('Homework4-cifar100-vgg16-keras.json','w').write(json_string)  
#model.save_weights('Homework4-cifar100-vgg16-keras.h5')

im = cv2.resize(cv2.imread('input/images/example.jpg'), (224, 224)).astype(np.float32)

im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68

im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

out = model.predict(im)

print(np.argmax(out))

#Y_pred = model.predict_proba(X_test, verbose=0)  # Keras预测概率Y_pred
#print(Y_pred[:3, ])  # 取前三张图片的十类预测概率

#score = model.evaluate(X_test, Y_test, verbose=0) # 评估测试集loss损失和精度acc
#print("测试集 score(val_loss): ", score[0])  # loss损失
#print("测试集 accuracy: ", score[1]) # 精度acc
#print("耗时: ",(time.time() - t0), "s")


 #读取model  
#model = model_from_json(open('my_model_architecture.json').read())  
#model.load_weights('my_model_weights.h5')  

# plot the result

#plt.figure
#plt.title("cost function")
#plt.plot(result.epoch,result.history['loss'],label="loss")
#plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
#plt.scatter(result.epoch,result.history['loss'])
#plt.scatter(result.epoch,result.history['val_loss'])
#plt.legend(loc='upper right')
#plt.show()
#
#
#plt.figure
#plt.title("accuracy")
#plt.plot(result.epoch,result.history['acc'],label="accuracy")
#plt.plot(result.epoch,result.history['val_acc'],label="val_accuracy")
#plt.scatter(result.epoch,result.history['acc'])
#plt.scatter(result.epoch,result.history['val_acc'])
#plt.legend(loc='lower right')
#plt.show()