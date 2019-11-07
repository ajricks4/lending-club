from sklearn.model_selection import train_test_split
import numpy as np
import numpy.random as rand
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
from keras import regularizers


def define_nn_mlp_model(X_train, y_train_ohe):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too

    model = Sequential() # sequence of layers
    num_neurons_in_layer = 12 # number of neurons in a layer (is it enough?)
    num_inputs = X_train.shape[1] # number of features (784) (keep)
    num_classes = y_train_ohe.shape[1]  # number of classes, 0-9 (keep)
    model.add(Dense(units=num_neurons_in_layer,
                    input_dim=num_inputs,
                    kernel_initializer='uniform',
                    activation='softplus')) # is tanh the best activation to use here?
    model.add(Dense(units=num_classes,
                    input_dim=num_neurons_in_layer,
                    kernel_initializer='uniform',
                    activation='softmax')) # keep softmax as last layer
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9) # learning rate, weight decay, momentum; using stochastic gradient descent (keep)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] ) # (keep)
    return model

class Neural_Net:



    def __init__(self, hidden_layers,hidden_nodes, act_funcs):

        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.act_funcs = act_funcs

    def set_params(self,optimizer,loss,epochs,batch):
        self.optimizer = optimizer
        self.loss = loss
        self.epochs
        self.batch = batch

    def fit_predict(self,X_train,y_train,X_test,y_test):
        self.model = keras.Sequential()
        self.model.add(Dense(units=X_train.shape[1],
                            input_dim = X_train.shape[1]
                            kernel_initializer='uniform',
                            activation='relu'))
        for i in range(0,len(self.num_neurons_in_layer)):
            self.model.add(Dense(units = self.num_neurons_in_layer[i],
                                 kernel_initializer='uniform',
                                 activation=self.act_funcs[i],
                                 kernel_regularizer=regularizers.l2(0.01),
                                 activity_regularizer=regularizers.l1(0.01)))
        self.model.compile(loss = self.loss,optimizer = self.optimizer, metrics = ['accuracy'])
        self.model.layers.activa tion('sigmoid')
        self.model.fit(X_train,y_train,epochs=self.epochs,batch_size=self.batch)
        self.preds = model.predict(X_test,verbose=1,batch_size=self.batch)
        self.score = model.evaluate(X_test, y_test, verbose=1,batch_size=self.batch)
        return self.preds, self.score
