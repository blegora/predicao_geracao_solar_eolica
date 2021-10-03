#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:21:09 2021

@author: bruno
"""

# import time
import util
import numpy as np
from mondrianforest import MondrianForestRegressor


class MFR(util.Util):

    def _init_params(self):
        super()._init_params()
        self.n_trees = 10
        self._accepted_params.extend(['n_trees'])

    def __init__(self, param_dict={}):
        # default values
        self._init_params()

        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                raise NameError("Parameter does not exist!")
        #self.parse_seed(self.seed)
        self.model = MondrianForestRegressor(n_tree=self.n_trees)

    def train(self,X,Y):
        self.model.partial_fit(X,Y.reshape(-1))
    def predict(self,X):
        return self.model.predict(X)

