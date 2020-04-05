# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 17:07:33 2018

@author: Administrator
"""

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from matplotlib import pyplot as plt

dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

Y_labels = to_categorical(Y,num_classes= 3)

seed = 7
np.random.seed(seed)

def create_model(optimalizer = 'rmsprop',init = 'glorot_uniform'):
    
    model = Sequential()
    model.add(Dense(units = 4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units = 6,activation='relu',kernel_initializer=init))
    model.add(Dense(units = 3,activation='softmax',kernel_initializer=init))
    
    model.compile(loss = 'categorical_crossentropy',optimizer=optimalizer,metrics=['accuracy'])
    
    return model

model = create_model()

history = model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0)

scores = model.evaluate(x,Y_labels,verbose=0)
print('%s: %.2f%%' %(model.metrics_names[1],scores[1]*100))

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='uppper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='uppper left')
plt.show()