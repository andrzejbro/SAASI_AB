# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:46:33 2019

@author: andrz
"""

import numpy as np
from keras import models
from keras import layers
from keras import losses
from keras import optimizers


import time
start = time.time()

#%%
data_train = np.load('NPZ/train_features_dense.npz')
data_test = np.load('NPZ/test_features_dense.npz')
data_val = np.load('NPZ/val_features_dense.npz')

#%%
train_features = data_train['a']
test_features = data_test['a']
val_features = data_val['a']

y_train = data_train['b']
y_test = data_test['b']
y_val = data_val['b']

train_features = np.reshape(train_features, (train_features.shape[0], 2*2*1024))
test_features = np.reshape(test_features, (test_features.shape[0], 2*2*1024))
val_features = np.reshape(val_features, (val_features.shape[0], 2*2*1024))

model = models.Sequential()
model.add(layers.Dense(512, activation='sigmoid', input_dim=2*2*1024))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation="softmax"))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adamax(),
              metrics=['accuracy'])

history = model.fit(train_features, y_train,
          batch_size=256,
          epochs=30,
          verbose=1,
          validation_data=(val_features, y_val))

score = model.evaluate(test_features, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#%%
end = time.time()
print(end - start)