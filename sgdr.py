 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:35 2019

@author: bruno
"""

# import time
import util
import numpy as np
from sklearn.linear_model import SGDRegressor


class SGDR(util.Util):

    def _init_params(self):
        super()._init_params()
        self.loss = 'squared_loss'
        self.penalty='l2'
        self.alpha = 0.0001
        self.l1_ratio = 0.15 #0 a 1
        self.tol = 1e-3
        self.epsilon = 0.1
        self.learning_rate = 'invscaling'
        self.eta0 = 0.01
        self.power_t = 0.25
        self.early_stopping = False
        self.validation_fraction = 0.1
        self.n_iter_no_change = 5
        self.fit_intercept = True
        self.average = False
        #self.max_iter = 5 #The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
        
        self._accepted_params.extend(['loss','penalty','alpha','l1_ratio','tol'\
                                      'epsilon','learning_rate','eta0','power_t',\
                                      'early_stopping','validation_fraction',\
                                      'n_iter_no_change','fit_intercept','average'])
        self.__trained = False

    def __init__(self, param_dict={}):
        # default values
        self._init_params()

        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                raise NameError("Parameter does not exist!")
        #self.parse_seed(self.seed)
        self.model = SGDRegressor(loss=self.loss, penalty=self.penalty, alpha=self.alpha,\
                                  l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,\
                                  tol=self.tol,epsilon=self.epsilon,random_state=self.seed, \
                                  learning_rate=self.learning_rate,eta0=self.eta0,\
                                  power_t=self.power_t, early_stopping=self.early_stopping,\
                                  validation_fraction=self.validation_fraction,\
                                  n_iter_no_change=self.n_iter_no_change, average=self.average,\
                                  warm_start=True)

    def train(self,X,Y):
        if self.__trained:
            self.model.partial_fit(X,Y.reshape(-1))
        else:
            self.model.fit(X,Y.reshape(-1))
            self.__trained = True
    def predict(self,X):
        return self.model.predict(X)

