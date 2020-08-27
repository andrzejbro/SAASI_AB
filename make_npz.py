# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:14:36 2019

@author: andrz
"""

import numpy as np
from keras.applications.densenet import DenseNet121
from keras.utils import to_categorical
from keras.applications.densenet import preprocess_input as pre_dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import time
start = time.time()

   
#%%
image_generator = ImageDataGenerator()

image_data = image_generator.flow_from_directory(
    directory=r"IM/3Color",
    target_size=(64, 64),
    color_mode="rgb",
    class_mode="binary",
    batch_size=9000,
    )

for image_batch,label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

X = np.array(image_batch)
y = to_categorical(np.array(label_batch))

#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=11)

#%%
dnsnet = DenseNet121(weights='imagenet', include_top=False, input_shape = (64,64,3), classes = 2)

X_dense_train = pre_dense(X_train)
X_dense_val = pre_dense(X_val)
X_dense_test = pre_dense(X_test)

features_dense_train = dnsnet.predict(np.array(X_dense_train), batch_size=256, verbose=1)
features_dense_val = dnsnet.predict(np.array(X_dense_val), batch_size=256, verbose=1)
features_dense_test = dnsnet.predict(np.array(X_dense_test), batch_size=256, verbose=1)


#%%
np.savez("train_features_dense", a = features_dense_train, b = y_train)
np.savez("val_features_dense", a = features_dense_val, b = y_val)
np.savez("test_features_dense", a = features_dense_test, b = y_test)

end = time.time()
print(end - start)