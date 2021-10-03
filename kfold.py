#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:35 2019

@author: bruno
"""

import util
import metrics
import time
import numpy as np
import relm


class KFold:
    def _init_params(self,numberOfFolds, classifierLambda, paramNames, paramValues,
                 metric,shuffleSamples, stratified, seedFolds, seedClass):
        self.paramNames = paramNames
        self.paramValues = paramValues
        
        if callable(classifierLambda): #probably a lambda..
            if isinstance(classifierLambda(),util.Util):
                self.classifierLambda = classifierLambda
            else:
                raise NameError("Classifier not supported. Please give a lambda to an util.Util classifier")
        else:
            raise NameError("Classifier not supported. Please give a lambda to an util.Util classifier")
        self.numberOfFolds = numberOfFolds
        
        if isinstance(metric,metrics.Metric):
            self.metric = metric
        else:
            raise NameError("metric not supported. Please give an metrics.Metric class")
                            
        self.shuffleSamples = shuffleSamples
        self.stratified = stratified
        if isinstance(seedFolds, int):
            self.seedFolds = np.random.mtrand.RandomState(seedFolds)
        elif isinstance(seedFolds, np.random.mtrand.RandomState):
            self.seedFolds = seedFolds
        else:
            raise NameError("Seed not supported. Please give an integer \
                or a numpy.random.mtrand.RandomState object")
        if isinstance(seedClass, int):
            self.seedClass = np.random.mtrand.RandomState(seedClass)
        elif isinstance(seedClass, np.random.mtrand.RandomState):
            self.seedClass = seedClass
        else:
            raise NameError("Seed not supported. Please give an integer \
                or a numpy.random.mtrand.RandomState object")        
        #self.seedFolds = seedFolds
        #self.seedClass = seedClass

    def __getGridIndices(self):
        gridLenghts = list(map(len,self.paramValues))
        gridPos = []
        for i in range(0,len(gridLenghts)):
            gridPos.append(list(range(gridLenghts[i])))
        gridIndices = np.meshgrid(*gridPos, indexing='ij')
        gridIndices = list(map(np.ndarray.flatten,gridIndices))
        indices = np.zeros((len(gridIndices[0]),len(gridLenghts)))
        for i in range(0,len(gridIndices)):
            indices[:,i] = gridIndices[i]
        return indices
    
    def __init__(self, numberOfFolds, classifierLambda, paramNames, paramValues,
                 metric, shuffleSamples=True, stratified=True, seedFolds=0, seedClass=0):
        self._init_params(numberOfFolds, classifierLambda, paramNames, paramValues,
                 metric,shuffleSamples, stratified, seedFolds, seedClass)
        
    def start(self, trData, trLab):
        
        if self.shuffleSamples:
            perm = self.seedFolds.permutation(np.arange(0,trData.shape[0])).astype(int)
            # trData = trData[indices,:]
            # trLab = trLab[indices,:]
        else:
            perm = np.arange(0,trData.shape[0]).astype(int)
        tamFold = np.floor(trData.shape[0]/self.numberOfFolds).astype(int)
        
        if not self.metric.is_regression_metric() and self.stratified():
            #I have to translate this...
            pass 
        
        bestMetric = self.metric.worst_case()
        indices = self.__getGridIndices()
        
        for i in range(0,indices.shape[0]):
            if np.mod(i,round(indices.shape[0]/10)) == 0:
                print(i,"/",indices.shape[0])
            classParams = {}
            for j in range(0,indices.shape[1]):
                # aux = self.paramValues[j]
                classParams[self.paramNames[j]] = self.paramValues[j][indices[i,j].astype(int)]
            
            metric = [self.metric.worst_case()]*self.numberOfFolds
            try: 
                for k in range(0,self.numberOfFolds):
                    kClassifier = self.classifierLambda(dict({'seed':self.seedClass},**classParams))
                    
                    testFoldIdx = perm[k*tamFold:(k+1)*tamFold]
                    trainFoldIdx = np.setdiff1d(perm, testFoldIdx)
                    
                    kClassifier.train(trData[trainFoldIdx,:],trLab[trainFoldIdx,:])
                    pred = kClassifier.predict(trData[testFoldIdx,:])
                    metric[k] = self.metric.calculate(trLab[testFoldIdx,:],pred)
            except: #Em alguns casos o algoritmo não converge e levanta uma exceção. 
                # Vamos ignorar esse conjunto de parâmetros e torcer pra um outro conjunto dar certo
                continue
                
            if i == 1 or self.metric.is_better(np.mean(metric),bestMetric):
                paramStruct = classParams
                bestMetric = np.mean(metric)
        
        return paramStruct,bestMetric


if __name__ == "__main__":
    kfold = KFold(5, lambda x={}: relm.RELM(dict({'number_of_input_neurons':2},**x)),
                  ['regularization_parameter','number_of_hidden_neurons'], 
                  [[2**x for x in range(-20,21)],np.arange(100,1001,100)],
                  metrics.RMSE())
    paramStruct,bestMetric = kfold.start(np.random.randn(100, 2),np.random.randint(0,2,(100,1)))
