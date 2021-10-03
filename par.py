#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:35 2019

@author: bruno
"""

# import time
import util
import numpy as np
from sklearn.linear_model import PassiveAggressiveRegressor


class PAR(util.Util):

    def _init_params(self):
        super()._init_params()
        self.C = 1.0
        self.fit_intercept = True
        self.tol = 1e-3
        self.early_stopping = False
        self.validation_fraction = 0.1
        self.n_iter_no_change = 5
        self.loss = 'epsilon_insensitive' # ou 'squared_epsilon_insensitive'
        self.epsilon = 0.1
        self.average = False
        #self.max_iter = 5 #The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
        self._accepted_params.extend(['C','fit_intercept','tol','early_stopping',\
                                      'validation_fraction','n_iter_no_change',\
                                      'loss','epsilon','average'])
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
        self.model = PassiveAggressiveRegressor(C=self.C,fit_intercept=self.fit_intercept,\
                                                tol=self.tol,early_stopping=self.early_stopping,\
                                                validation_fraction=self.validation_fraction,\
                                                n_iter_no_change=self.n_iter_no_change,\
                                                loss=self.loss, epsilon=self.epsilon,\
                                                random_state=self.seed,warm_start=True,\
                                                average=self.average)

    def train(self,X,Y):
        if self.__trained:
            self.model.partial_fit(X,Y.reshape(-1))
        else:
            self.model.fit(X,Y.reshape(-1))
            self.__trained = True
    def predict(self,X):
        return self.model.predict(X)

