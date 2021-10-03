#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:35 2019

@author: bruno
"""

import util
import time
import numpy as np


class ELM(util.Util):

    def _init_params(self):  # "Protected" method
        # default values
        super()._init_params()
        self.number_of_hidden_neurons = 1000
        self.activation_function = 'sig'
        self.number_of_input_neurons = []
        self.input_weight = []
        self.bias_of_hidden_neurons = []
        self.output_weight = []
        self.use_parallel_layer = False
        self.input_weight2 = []
        self.use_auto_encoder = False
#        self.seed = 0
        self._accepted_params.extend([
            'number_of_hidden_neurons',
            'activation_function',
            'number_of_input_neurons',
            'use_parallel_layer',
            'use_auto_encoder'
        ])

    def __init__(self, param_dict):

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
        self.parse_seed(self.seed)  # hw to mk more readable Util.parse_seed?

    def calculate_hidden_matrix(self, X):
        temp_H = np.matmul(X, self.input_weight) + self.bias_of_hidden_neurons
        H = self.activation_function(temp_H)
        temp_H = []
        if self.use_parallel_layer:
            if self.input_weight2 == []:
                self.input_weight2 = self.generate_random_weights(
                    self.number_of_input_neurons,
                    self.number_of_hidden_neurons,
                    X)  # readability?
            temp_H2 = np.matmul(X, self.input_weight2)
            H2 = self.activation_function(temp_H2)
            temp_H2 = []
            H = H * H2
        return H

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
        if H.shape[0] >= H.shape[1]:
            # pinv = np.linalg.pinv(np.matmul(np.transpose(H), H))
            # pinv = np.matmul(pinv, np.transpose(H))
            pinv = np.matmul(np.transpose(H), H)
            q = np.matmul(np.transpose(H), Y)
            self.output_weight = np.linalg.solve(pinv, q)
        else:
            # pinv = np.linalg.pinv(np.matmul(H, np.transpose(H)))
            # pinv = np.matmul(np.transpose(H), pinv)
            pinv = np.matmul(H, np.transpose(H))
            q = np.linalg.solve(pinv, Y)
            self.output_weight = np.matmul(np.transpose(H), q)

        self.train_time = time.time() - aux_time

    def predict(self, X):
        aux_time = time.time()
        H = self.calculate_hidden_matrix(X)
        Yh = np.matmul(H, self.output_weight)
        self.last_test_time = time.time() - aux_time
        return Yh


if __name__ == "__main__":
    a = ELM({'number_of_input_neurons': 4})
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=(150, 1))
    a.train(X, Y)
    yh = a.predict(X)
