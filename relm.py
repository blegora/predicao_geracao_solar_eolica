#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:35 2019

@author: bruno
"""

import time
import elm
import numpy as np


class RELM(elm.ELM):

    def _init_params(self):
        super()._init_params()
        self.regularization_parameter = 1000
        self.alpha = 0
        self._accepted_params.extend(['regularization_parameter', 'alpha'])

    def __init__(self, param_dict):
        # default values
        self._init_params()

        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                raise NameError("Parameter does not exist!")
        if self.number_of_input_neurons == []:
            raise NameError("Number of input neurons not defined!")
        self.activation_function = self.parse_activation_function(
            self.activation_function)
        self.parse_seed(self.seed)

    def generate_random_weights(self, num_input, num_hidden, X):
        if self.use_auto_encoder:
            self.use_auto_encoder = False
            self.train(X, X)
            self.use_auto_encoder = True
            return np.transpose(self.output_weight)
        else:
            return super().generate_random_weights(num_input, num_hidden, X)

    def train(self, X, Y):
        aux_time = time.time()
        self.input_weight = self.generate_random_weights(
            self.number_of_input_neurons,
            self.number_of_hidden_neurons,
            X)
        self.bias_of_hidden_neurons = self.seed.randn(
            1,
            self.number_of_hidden_neurons)
        H = self.calculate_hidden_matrix(X)
        if self.alpha == 0:
            if H.shape[0] >= H.shape[1]:
                # pinv = np.linalg.pinv(np.matmul(np.transpose(H), H) +
                #      np.eye(H.shape[1])/self.regularization_parameter)
                # pinv = np.matmul(pinv, np.transpose(H))
                pinv = np.matmul(np.transpose(H), H) + \
                    np.eye(H.shape[1])/self.regularization_parameter
                q = np.matmul(np.transpose(H), Y)
                self.output_weight = np.linalg.solve(pinv, q)
            else:
                # pinv = np.linalg.pinv(np.matmul(H, np.transpose(H)) + \
                #     np.eye(H.shape[0])/self.regularization_parameter)
                # pinv = np.matmul(np.transpose(H), pinv)
                # self.output_weight = np.matmul(pinv, Y)
                pinv = np.matmul(H, np.transpose(H)) + \
                    np.eye(H.shape[0])/self.regularization_parameter
                q = np.linalg.solve(pinv, Y)
                self.output_weight = np.matmul(np.transpose(H), q)

        else:
            import sklearn.linear_model as lm
            self.output_weight = np.zeros(shape=(
                self.number_of_hidden_neurons,
                Y.shape[1]))
            for j in range(Y.shape[1]):
                _, aux, _ = lm.enet_path(
                    H,
                    Y[:, j],
                    l1_ratio=self.alpha,
                    n_alphas=1,
                    alphas=[1/self.regularization_parameter])
                self.output_weight[:, j] = np.reshape(
                    aux,
                    newshape=(self.number_of_hidden_neurons))
        self.train_time = time.time() - aux_time


if __name__ == "__main__":
    a = RELM({'number_of_input_neurons': 4})
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=(150, 1))
    a.train(X, Y)
    yh = a.predict(X)
    # [a,b,c] = lm.enet_path(X, Y, l1_ratio=0.5, n_alphas=1, alphas=[100])
