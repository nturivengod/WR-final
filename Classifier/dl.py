from .baseClassifier import BaseClassifier
from keras import backend as K
import numpy as np
from keras.models import load_model
from keras.layers import advanced_activations
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout

class DL(BaseClassifier):
    config = None
    clf = None
    def __init__(self, arg):
        self.clf = None

    def fit(self, X, y, val_X, val_y):
        checkpoint = ModelCheckpoint(filepath='m4.hdf5',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max')
        callbacks = [checkpoint]
        self.clf = Sequential()
        self.clf.add(Dense(256, input_dim=X.shape[1],init='uniform'))#, use_bias=True))
        self.clf.add(advanced_activations.ELU(alpha=1.0))
        self.clf.add(Dropout(0.6))
        self.clf.add(Dense(128,  init = 'uniform'))
        self.clf.add(advanced_activations.ELU(alpha=1.0))
        self.clf.add(Dropout(0.6))
        self.clf.add(Dense(64, use_bias=True, init = 'uniform'))
        self.clf.add(advanced_activations.ELU(alpha=1.0))
        self.clf.add(Dropout(0.6))
        #model.add(Dense(4, activation='sigmoid', use_bias=True))
        #model.add(Dropout(0.5))
        self.clf.add(Dense(1, activation='sigmoid', init = 'uniform'))
        adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
        self.clf.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
        return self.clf.fit(X, y,
          epochs=50,
          batch_size=2800,validation_data = (val_X, val_y), verbose = 1,callbacks = callbacks)
    def predict(self, X):
        self.clf.load_weights('m4.hdf5')
        r = self.clf.predict(X)
        y = []
        for i in range(len(r)):
          y.append(round(r[i][0]))
        return y

