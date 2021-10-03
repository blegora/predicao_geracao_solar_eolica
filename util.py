#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:31:35 2019

@author: bruno
"""

import abc
import numpy as np


class Util(abc.ABC):  # Abstract class
    def _init_params(self):
        self.train_time = 0
        self.use_random_orthogonalization = False
        self.last_test_time = 0
        self.train_partial_metric = 0
        self.test_partial_metric = 0
        self.seed = 0
        self._accepted_params = ['use_random_orthogonalization', 'seed']

    @abc.abstractmethod
    def train(self, X, Y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    def parse_activation_function(self, act_fun):
        if type(act_fun) == str:
            if (act_fun == 'sig' or act_fun == 'sigmoid'):
                return lambda x: 1 / (1 + np.exp(x))
            elif (act_fun == 'sin' or act_fun == 'sine'):
                return lambda x: np.sin(x)
            elif act_fun == 'hardlim':
                return lambda x: (x > 1.0).astype('float64')
            elif act_fun == 'radbas' or act_fun == 'gauss':
                return lambda x: np.exp(-np.power(x,2))
            elif act_fun == 'multi':
                return lambda x: np.sqrt(1+np.power(x,2))
            else:
                raise NameError("Activation Function not supported. \
                    Please use a supported string or lambda function")
        elif not act_fun.__name__ == '<lambda>':
            raise NameError("Activation Function not supported. \
                Please use a supported string or lambda function")
        else:
            return act_fun

    def parse_seed(self, seed):
        if isinstance(seed,(int,np.int32,np.int64,np.uint32,np.uint64)):
            self.seed = np.random.mtrand.RandomState(seed)
        elif isinstance(seed, np.random.mtrand.RandomState):
            self.seed = seed
        else:
            raise NameError("Seed not supported. Please give an integer \
                or a numpy.random.mtrand.RandomState object")

    def generate_random_weights(self, first_dim, second_dim, X=[]):
        if not isinstance(self.seed, np.random.mtrand.RandomState):
            self.parse_seed(self.seed)
        w = self.seed.rand(first_dim, second_dim)*2 - 1
        if self.use_random_orthogonalization:
            import scipy.linalg as linalg
            w = linalg.orth(np.transpose(w))
            w = np.transpose(w)
            if second_dim < first_dim:
                Wpca = self.PCA(X, second_dim)
                w = np.matmul(Wpca, w[:, :second_dim])
        return w

    @staticmethod
    def normalize_data(data, param=[], lower_limit=-1, upper_limit=1):
        if param == []:
            minimum = np.amin(data, axis=0)  # axis=0: min of each column
            maximum = np.amax(data, axis=0)
            for i in range(0, len(maximum)):
                if maximum[i] == minimum[i]:
                    maximum[i] = maximum[i] + np.spacing(maximum[i])
            param = (minimum, np.asarray(maximum))
        else:
            minimum = param[0]
            maximum = param[1]  # Param is a tuple!
        dta = lower_limit + (upper_limit-lower_limit) * \
            (data - minimum)/(maximum - minimum)
        return (dta, param)

    @staticmethod
    def unnormalize_data(data, param, lower_limit=-1, upper_limit=1):
        minimum = param[0]
        maximum = param[1]
        aux = ((data-lower_limit)/(upper_limit - lower_limit)) * \
            (maximum - minimum) + minimum
        return aux

    @staticmethod
    def PCA(matrix, num_eigs):
        mean = np.mean(matrix, axis=0)  # axis=0: mean of each column
        B = matrix - mean
        C = np.cov(B, rowvar=False)
        w, v = np.linalg.eigh(C)
        Ve2 = v[:, np.argsort(-w)]
        return Ve2[:, 0:num_eigs]


if __name__ == "__main__":
    # a = Util() #Abstract class, cant instantiate
    # a = np.random.rand(3, 3)
    # dta,params = Util.normalize_data(a)
    # dta2 = Util.unormalize_data(dta,params)
    pass
