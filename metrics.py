#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 13:00:35 2019

@author: bruno
"""

import abc
import numpy as np


class Metric(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(pred, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def worst_case(pred, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def is_regression_metric(pred, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def is_better(val1, val2):
        pass


class RMSE(Metric):

    @staticmethod
    def worst_case():
        return np.Inf

    @staticmethod
    def is_regression_metric():
        return True

    @staticmethod
    def is_better(val1, val2):
        return val1 < val2

    @staticmethod
    def calculate(pred, target):
        aux = pred - target
        aux = np.reshape(np.asarray(
            np.sqrt(np.mean(np.mean(aux*aux)))),
            (1, 1))
        return aux


class Accuracy(Metric):

    @staticmethod
    def worst_case():
        return 0

    @staticmethod
    def is_regression_metric():
        return False

    @staticmethod
    def calculate(pred, target):
        predMax = np.argmax(pred, axis=1)
        targetMax = np.argmax(target, axis=1)
        value = np.sum(predMax == targetMax)/predMax.size
        return value

    def is_better(value1, value2):
        return value1 > value2


class RRMSE(RMSE):
    @staticmethod
    def calculate(pred, target):
        aux = pred - target
        mean = np.mean(target, axis=0)
        den = mean - target
        den = np.sum(den*den, axis=0)
        num = np.sum(aux*aux, axis=0)
        aux = np.sqrt(num/den)
        return aux


class aRRMSE(RRMSE):
    @staticmethod
    def calculate(pred, target):
        aux = np.mean(RRMSE.calculate(pred, target))
        return aux
    
    
## https://pyfts.github.io/pyFTS/build/html/_modules/pyFTS/benchmarks/Measures.html

class MAPE(Metric):
    @staticmethod
    def worst_case():
        return np.Inf

    @staticmethod
    def is_regression_metric():
        return True

    @staticmethod
    def is_better(val1, val2):
        return val1 < val2

    @staticmethod
    def calculate(forecasts, targets):
        if isinstance(targets, list):
            targets = np.array(targets)
        if isinstance(forecasts, list):
            forecasts = np.array(forecasts)
        return np.nanmean(np.abs(np.divide(np.subtract(targets, forecasts), targets))) * 100
    
class SMAPE(MAPE):
    @staticmethod
    def calculate(forecasts, targets, type=2):
        if isinstance(targets, list):
            targets = np.array(targets)
        if isinstance(forecasts, list):
            forecasts = np.array(forecasts)
        if type == 1:
            return np.nanmean(np.abs(forecasts - targets) / ((forecasts + targets) / 2))
        elif type == 2:
            return np.nanmean(np.abs(forecasts - targets) / (np.abs(forecasts) + abs(targets))) * 100
        else:
            return np.nansum(np.abs(forecasts - targets)) / np.nansum(forecasts + targets)

# if __name__ == '__main__':
#     a = aRRMSE()
#     print(a.calculate(np.array([1,2,3]),np.array([4,5,6])))
