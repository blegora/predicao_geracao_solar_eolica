#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:35 2019

@author: bruno
"""

import util
import time
import numpy as np


class KELM(util.Util):
    def _init_params(self):
        self.kernel_type = 'RBF_kernel'
        self.kernel_param = [0.1]
        self.regularization_parameter = 1000.0
        self.output_weight = []
        self.Xtr = []
        self._accepted_params = [
            'kernel_type',
            'kernel_param',
            'regularization_parameter'
        ]

    def __init__(self, param_dict={}):
        self._init_params()
        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                raise NameError("Parameter does not exist!")

    def _kernel_matrix(self, Xte=[]):
        num_samples = self.Xtr.shape[0]
        flag = isinstance(Xte, list) and len(Xte) == 0
        if self.kernel_type == 'RBF_kernel':
            if flag:
                XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
                XXh = np.matmul(XXh, np.ones((1, num_samples)))
                omega = XXh + np.transpose(XXh) - 2 * np.matmul(
                    self.Xtr,
                    np.transpose(self.Xtr))
                omega = np.exp(-omega/self.kernel_param[0])
            else:
                XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
                XXh = np.matmul(XXh, np.ones((1, Xte.shape[0])))
                XXh2 = np.sum(Xte*Xte, axis=1, keepdims=True)
                XXh2 = np.matmul(XXh2, np.ones((1, num_samples)))
                omega = XXh + np.transpose(XXh2) - 2 * np.matmul(
                    self.Xtr,
                    np.transpose(Xte))
                omega = np.exp(-omega/self.kernel_param[0])
        elif self.kernel_type == 'lin_kernel':
            if flag:
                omega = np.matmul(self.Xtr, np.transpose(self.Xtr))
            else:
                omega = np.matmul(self.Xtr, np.transpose(Xte))
        elif self.kernel_type == 'poly_kernel':
            if flag:
                omega = np.matmul(self.Xtr, np.transpose(self.Xtr))
                omega = omega + self.kernel_param[0]
                omega = np.power(omega, self.kernel_param[1])
            else:
                omega = np.matmul(self.Xtr, np.transpose(Xte))
                omega = omega + self.kernel_param[0]
                omega = np.power(omega, self.kernel_param[1])
        elif self.kernel_type == 'wav_kernel':
            if flag:
                XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
                XXh = np.matmul(XXh, np.ones((1, num_samples)))
                omega = XXh + np.transpose(XXh)
                omega = omega - 2*np.matmul(self.Xtr, np.transpose(self.Xtr))
                XXh1 = np.sum(self.Xtr, axis=1, keepdims=True)
                XXh1 = np.matmul(XXh1, np.ones((1, num_samples)))
                omega1 = XXh1 - np.transpose(XXh1)
                omega = np.cos(
                    self.kernel_param[2] * omega1 / self.kernel_param[1])
                omega = omega*np.exp(-omega/self.kernel_param[0])
            else:
                XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
                XXh = np.matmul(XXh, np.ones((1, Xte.shape[0])))
                XXh2 = np.sum(Xte*Xte, axis=1, keepdims=True)
                XXh2 = np.matmul(XXh2, np.ones(1, num_samples))
                omega = XXh + np.transpose(XXh2) - 2 * np.matmul(
                    self.Xtr,
                    np.transpose(Xte))

                XXh11 = np.sum(self.Xtr, axis=1)
                XXh11 = np.matmul(XXh11, np.ones((1, Xte.shape[0])))
                XXh22 = np.sum(Xte, axis=1)
                XXh22 = np.matmul(XXh22, np.ones((1, num_samples)))
                omega1 = XXh11 - np.transpose(XXh22)
                omega = np.cos(
                    self.kernel_param[2] * omega1 / self.kernel_param[1])
                omega = omega*np.exp(-omega/self.kernel_param[0])
        return omega

    def train(self, X, Y):
        aux_time = time.time()
        self.Xtr = X
        Omega_train = self._kernel_matrix()
        pinv = np.linalg.pinv(
            Omega_train + np.eye(Y.shape[0])/self.regularization_parameter)
        self.output_weight = np.matmul(pinv, Y)
        self.train_time = time.time() - aux_time

    def predict(self, X):
        aux_time = time.time()
        Omega = self._kernel_matrix(X)
        Yhat = np.matmul(np.transpose(Omega), self.output_weight)
        self.last_test_time = time.time() - aux_time
        return Yhat


if __name__ == "__main__":
    a = KELM()
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=150)
    a.train(X, Y)
    yh = a.predict(X)
