#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import kelm
import numpy as np


class RKELM(kelm.KELM):
    def _init_params(self):
        super()._init_params()
        self.number_of_hidden_neurons = 100
        self._support = []
        self.seed = 0
        self._accepted_params.extend(['number_of_hidden_neurons', 'seed'])

    def __init__(self, param_dict={}):
        self._init_params()
        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                print(key)
                raise NameError("Parameter does not exist!")
        self.parse_seed(self.seed)

    def train(self, X, Y):
        aux_time = time.time()
        self._support = self.seed.permutation(X.shape[0])[:self.number_of_hidden_neurons]
        self.Xtr = X[self._support, :]
        Omega_train = np.transpose(self._kernel_matrix(X))
        if Omega_train.shape[0] >= Omega_train.shape[1]:
            pinv = np.eye(Omega_train.shape[1]) / self.regularization_parameter
            pinv = pinv + np.matmul(np.transpose(Omega_train), Omega_train)
            q = np.matmul(np.transpose(Omega_train), Y)
            self.output_weight = np.linalg.solve(pinv, q)
        else:
            pinv = np.eye(Omega_train.shape[0]) / self.regularization_parameter
            pinv = pinv + np.matmul(Omega_train, np.transpose(Omega_train))
            q = np.linalg.solve(pinv, Y)
            self.output_weight = np.matmul(np.transpose(Omega_train), q)
        self.train_time = time.time() - aux_time


if __name__ == "__main__":
    a = RKELM()
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=(150, 1))
    a.train(X, Y)
    yh = a.predict(X)
