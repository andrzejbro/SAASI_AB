# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:40:00 2019

@author: andrz
"""

import numpy as np
from keras import models
from keras import layers
from keras import losses
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold


import time
start = time.time()

#%% Create model for KerasClassifier
def create_model(hpar3_relu,
                 hpar2_dropout,
                 hpar5_optimizer):
    
    model = models.Sequential()
    model.add(layers.Dense(512, activation=hpar3_relu, input_dim=2*2*1024))
    model.add(layers.Dropout(hpar2_dropout))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss=losses.categorical_crossentropy,optimizer=hpar5_optimizer,metrics=['accuracy'])
    return model


# GridSearch in action
model = KerasClassifier(build_fn=create_model,
                        hpar3_relu = 'relu',
                        hpar2_dropout = 0.5,
                        hpar5_optimizer = 'Adamax',
                        epochs=10, 
                        batch_size=512, verbose=1)

#%%
data_dense = np.load('NPZ/train_features_dense.npz')

features_dense = data_dense['a']
y_dense = data_dense['b']
features_dense = np.reshape(features_dense, (features_dense.shape[0], 2*2*1024))


#%%
kf = KFold(n_splits=5)
kf.get_n_splits(features_dense)
print(kf)


#%%
"""
hpar1_batch = [64, 128, 256, 512]
hpar2_dropout = [0.3 , 0.5 , 0.7]
hpar3_relu = ['relu','tanh','sigmoid','linear']
hpar4_epoch = [5, 10, 15, 20]
hpar5_optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
"""

hpar1_batch = [256, 512]
hpar2_dropout = [0.5 , 0.6]
hpar3_relu = ['relu','sigmoid']
hpar4_epoch = [10, 15]
hpar5_optimizer = [ 'Adam', 'Adamax']


# Prepare the Grid
param_grid = dict(batch_size=hpar1_batch, 
                  hpar2_dropout=hpar2_dropout,
                  hpar3_relu=hpar3_relu,
                  epochs=hpar4_epoch,
                  hpar5_optimizer=hpar5_optimizer
                  )


grid = GridSearchCV(estimator=model,                     
                    param_grid=param_grid,
                    cv=kf)

grid_result = grid.fit(features_dense, y_dense)


#%% Show the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    

end = time.time()
print(end - start)