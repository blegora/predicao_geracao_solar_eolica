#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:22:50 2021

@author: bruno
"""

import osrelm
import time
import util
import numpy as np
import copy
import metrics


def _add_duples(t1,t2):
    aux1 = t1[0] + t2[0]
    aux2 = t1[1] + t2[1]
    return (aux1,aux2)
    

class _OSFDSNModule(osrelm.OSRELM):

    def _init_params(self):
        super()._init_params()
        self.total_number_of_input = 0
        self.number_of_output_neurons = []
        self.trained = False
        self._accepted_params.extend([
            'number_of_output_neurons',
                'input_weight', 
                'input_weight2',
                'bias_of_hidden_neurons',
                'total_number_of_input'
        ])

    def calculate_hidden_before_act(self, X):
        h = np.matmul(X, self.input_weight) + self.bias_of_hidden_neurons
        if self.use_parallel_layer:
            h2 = np.matmul(X, self.input_weight2)
            h = (h, h2)
        return h

    def hidden_layer_output(self, X, last_hidden, last_layer_out):
        if last_layer_out == []:
            last_hidden = self.calculate_hidden_before_act(X)
        else:
            if self.use_parallel_layer:
                duple = (np.matmul(last_layer_out[0], self.input_weight),\
                         np.matmul(last_layer_out[1], self.input_weight2))
                last_hidden = _add_duples(last_hidden,duple)
            else:
                last_hidden = last_hidden + \
                    np.matmul(last_layer_out, self.input_weight)
                    
        if self.use_parallel_layer:
            H1 = self.activation_function(last_hidden[0])
            H2 = self.activation_function(last_hidden[1])
            H = H1*H2
        else:
            H = self.activation_function(last_hidden)
        return (H, last_hidden)

    def train(self, X, Y, bias, last_hidden, last_layer_out):
        if len(last_layer_out) == 0: #first module
            self.bias_of_hidden_neurons = bias;
        aux_time = time.time()
        
        self.input_weight = self.generate_random_weights(
            self.number_of_input_neurons,
            self.number_of_hidden_neurons,
            X)
        
        (H, last_hidden) = self.hidden_layer_output(
            X, last_hidden, last_layer_out)
        
        C = self.regularization_parameter
        if len(self.pMat) == 0:
            if H.shape[0] < self.number_of_hidden_neurons:
                self.hMat = H
                self.tMat = Y
                del Y
                aux = np.matmul(self.hMat, np.transpose(self.hMat))
                aux = aux + np.eye(self.hMat.shape[0])/C
                self.pMat = np.linalg.inv(aux)
                self.output_weight = np.matmul(
                    self.pMat,
                    self.tMat
                )
                self.output_weight = np.matmul(
                    np.transpose(self.hMat),
                    self.output_weight
                )
            else:
                aux = np.matmul(np.transpose(H), H)
                aux = aux + np.eye(H.shape[1])/C
                self.pMat = np.linalg.inv(aux)
                self.output_weight = np.matmul(
                    self.pMat,
                    np.transpose(H))
                self.output_weight = np.matmul(self.output_weight, Y)
        else:
            if self.pMat.shape[0] < self.number_of_hidden_neurons:
                if ((self.pMat.shape[0]+H.shape[0]) >
                        self.number_of_hidden_neurons):
                    self.hMat = np.concatenate(
                        (self.hMat, H),
                        axis=0
                    )
                    self.tMat = np.concatenate(
                        (self.tMat, Y),
                        axis=0
                    )
                    del Y
                    aux = np.matmul(np.transpose(self.hMat), self.hMat)
                    aux = aux + np.eye(self.hMat.shape[1])/C
                    self.pMat = np.linalg.inv(aux)
                    self.output_weight = np.matmul(
                        self.pMat,
                        np.transpose(self.hMat))
                    self.output_weight = np.matmul(
                        self.output_weight,
                        self.tMat)
                else:
                    aux = np.matmul(H, np.transpose(H)) + np.eye(H.shape[0])/C
                    invS = np.matmul(self.hMat, np.transpose(H))
                    invS = np.matmul(
                        np.transpose(invS),
                        np.matmul(self.pMat, invS))
                    invS = np.linalg.pinv(aux - invS)
                    A = np.matmul(np.transpose(self.hMat), self.pMat)
                    A = np.matmul(H, A)
                    A = np.matmul(invS, A)
                    A = np.matmul(np.transpose(H), A)
                    A = np.matmul(self.hMat, A)
                    A = np.matmul(self.pMat, A)
                    A = self.pMat + A
                    B = np.matmul(np.transpose(H), invS)
                    B = np.matmul(self.hMat, B)
                    B = -np.matmul(self.pMat, B)
                    C = np.matmul(np.transpose(self.hMat), self.pMat)
                    C = np.matmul(H, C)
                    C = -np.matmul(invS, C)
                    self.hMat = np.concatenate(
                        (self.hMat, H),
                        axis=0
                    )
                    A = np.concatenate((A, B), axis=1)
                    C = np.concatenate((C, invS), axis=1)
                    self.pMat = np.concatenate((A, C), axis=0)
                    self.tMat = np.concatenate(
                        (self.tMat, Y),
                        axis=0
                    )
                    del Y, A, B, C, invS
                    self.output_weight = np.matmul(self.pMat, self.tMat)
                    self.output_weight = np.matmul(
                        np.transpose(self.hMat),
                        self.output_weight
                    )
            else:
                aux = np.matmul(self.pMat, np.transpose(H))
                aux = np.matmul(H, aux) + np.eye(H.shape[0])
                aux = np.linalg.inv(aux)
                aux = np.matmul(aux, H)
                aux = np.matmul(aux, self.pMat)
                aux = np.matmul(np.transpose(H), aux)
                aux = np.matmul(self.pMat, aux)
                self.pMat = self.pMat - aux
                aux = Y - np.matmul(H, self.output_weight)
                aux = np.matmul(np.transpose(H), aux)
                aux = np.matmul(self.pMat, aux)
                self.output_weight = self.output_weight + aux
        
        self.train_time = time.time() - aux_time
        return (last_hidden,np.matmul(H,self.output_weight))

    def predict(self, X, last_hidden, last_layer_out):
        H, last_hidden = self.hidden_layer_output(
            X, last_hidden, last_layer_out)
        out = np.matmul(H, self.output_weight)
        return (out, last_hidden)
    
    
class _kfold_osfdnmodule(osrelm.OSRELM):

    def _init_params(self):
        super()._init_params()
        self.total_number_of_input = 0
        self.number_of_output_neurons = []
        self.trained = False
        self.test_last_hidden = []
        self.test_last_output = []
        self.is_first_layer = False
        self.last_hidden_before_act = []
        self.last_layer_output= []
        self.kfold_indices = []
        self._accepted_params.extend([
            'number_of_output_neurons',
            'is_first_layer',
            'last_hidden_before_act',
            'last_layer_output',
            'kfold_indices',
            'test_last_hidden',
            'test_last_output',
                'input_weight', 
                'input_weight2',
                'bias_of_hidden_neurons',
                'total_number_of_input'
        ])

    def calculate_hidden_before_act(self, X):
        h = np.matmul(X, self.input_weight) + self.bias_of_hidden_neurons
        if self.use_parallel_layer:
            h2 = np.matmul(X, self.input_weight2)
            h = (h, h2)
        return h

    def hidden_layer_output(self, X, last_hidden, last_layer_out):
        if last_layer_out == []:
            last_hidden = self.calculate_hidden_before_act(X)
        else:
            if self.use_parallel_layer:
                duple = (np.matmul(last_layer_out[0], self.input_weight),\
                         np.matmul(last_layer_out[1], self.input_weight2))
                last_hidden = _add_duples(last_hidden,duple)
            else:
                last_hidden = last_hidden + \
                    np.matmul(last_layer_out, self.input_weight)
                    
        if self.use_parallel_layer:
            H1 = self.activation_function(last_hidden[0])
            H2 = self.activation_function(last_hidden[1])
            H = H1*H2
        else:
            H = self.activation_function(last_hidden)
        return (H, last_hidden)

    def train(self, X, Y):
        # if len(last_layer_out) == 0: #first module
            # self.bias_of_hidden_neurons = bias;
        aux_time = time.time()
        
        
        
        self.input_weight = self.generate_random_weights(
            self.number_of_input_neurons,
            self.number_of_hidden_neurons,
            X)
        
        last_hidden = self.last_hidden_before_act[self.kfold_indices,:]
        last_layer_out = self.last_layer_output[self.kfold_indices,:]
        
        (H, last_hidden) = self.hidden_layer_output(
            X, last_hidden, last_layer_out)
        
        C = self.regularization_parameter
        if len(self.pMat) == 0:
            if H.shape[0] < self.number_of_hidden_neurons:
                self.hMat = H
                self.tMat = Y
                #del Y
                aux = np.matmul(self.hMat, np.transpose(self.hMat))
                aux = aux + np.eye(self.hMat.shape[0])/C
                self.pMat = np.linalg.inv(aux)
                self.output_weight = np.matmul(
                    self.pMat,
                    self.tMat
                )
                self.output_weight = np.matmul(
                    np.transpose(self.hMat),
                    self.output_weight
                )
            else:
                aux = np.matmul(np.transpose(H), H)
                aux = aux + np.eye(H.shape[1])/C
                self.pMat = np.linalg.inv(aux)
                self.output_weight = np.matmul(
                    self.pMat,
                    np.transpose(H))
                self.output_weight = np.matmul(self.output_weight, Y)
        else:
            if self.pMat.shape[0] < self.number_of_hidden_neurons:
                if ((self.pMat.shape[0]+H.shape[0]) >
                        self.number_of_hidden_neurons):
                    self.hMat = np.concatenate(
                        (self.hMat, H),
                        axis=0
                    )
                    self.tMat = np.concatenate(
                        (self.tMat, Y),
                        axis=0
                    )
                    #del Y
                    aux = np.matmul(np.transpose(self.hMat), self.hMat)
                    aux = aux + np.eye(self.hMat.shape[1])/C
                    self.pMat = np.linalg.inv(aux)
                    self.output_weight = np.matmul(
                        self.pMat,
                        np.transpose(self.hMat))
                    self.output_weight = np.matmul(
                        self.output_weight,
                        self.tMat)
                else:
                    aux = np.matmul(H, np.transpose(H)) + np.eye(H.shape[0])/C
                    invS = np.matmul(self.hMat, np.transpose(H))
                    invS = np.matmul(
                        np.transpose(invS),
                        np.matmul(self.pMat, invS))
                    invS = np.linalg.pinv(aux - invS)
                    A = np.matmul(np.transpose(self.hMat), self.pMat)
                    A = np.matmul(H, A)
                    A = np.matmul(invS, A)
                    A = np.matmul(np.transpose(H), A)
                    A = np.matmul(self.hMat, A)
                    A = np.matmul(self.pMat, A)
                    A = self.pMat + A
                    B = np.matmul(np.transpose(H), invS)
                    B = np.matmul(self.hMat, B)
                    B = -np.matmul(self.pMat, B)
                    C = np.matmul(np.transpose(self.hMat), self.pMat)
                    C = np.matmul(H, C)
                    C = -np.matmul(invS, C)
                    self.hMat = np.concatenate(
                        (self.hMat, H),
                        axis=0
                    )
                    A = np.concatenate((A, B), axis=1)
                    C = np.concatenate((C, invS), axis=1)
                    self.pMat = np.concatenate((A, C), axis=0)
                    self.tMat = np.concatenate(
                        (self.tMat, Y),
                        axis=0
                    )
                    #del Y, A, B, C, invS
                    self.output_weight = np.matmul(self.pMat, self.tMat)
                    self.output_weight = np.matmul(
                        np.transpose(self.hMat),
                        self.output_weight
                    )
            else:
                aux = np.matmul(self.pMat, np.transpose(H))
                aux = np.matmul(H, aux) + np.eye(H.shape[0])
                aux = np.linalg.inv(aux)
                aux = np.matmul(aux, H)
                aux = np.matmul(aux, self.pMat)
                aux = np.matmul(np.transpose(H), aux)
                aux = np.matmul(self.pMat, aux)
                self.pMat = self.pMat - aux
                aux = Y - np.matmul(H, self.output_weight)
                aux = np.matmul(np.transpose(H), aux)
                aux = np.matmul(self.pMat, aux)
                self.output_weight = self.output_weight + aux
        
        self.train_time = time.time() - aux_time
        return (last_hidden,np.matmul(H,self.output_weight))

    def predict(self, X, test_idx=[]):
        if test_idx == []:
            H, last_hidden = self.hidden_layer_output(
                X, self.test_last_hidden, self.test_last_output)
        else:
            H, last_hidden = self.hidden_layer_output(
                X, self.lastHiddenBeforeAct[test_idx,:], self.last_layer_output[test_idx,:])
        out = np.matmul(H, self.output_weight)
        return (out, last_hidden)

class _KFold_osfdsn:
    def _init_params(self,numberOfFolds, classifierLambda, paramNames, paramValues,
                 metric,shuffleSamples, stratified, seedFolds, seedClass):
        self.paramNames = paramNames
        self.paramValues = paramValues
        self.test_fold = []
        self.test_targets = []
        self.test_last_hidden = []
        self.test_last_output = []
        
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
            classParams = {}
            for j in range(0,indices.shape[1]):
                # aux = self.paramValues[j]
                classParams[self.paramNames[j]] = self.paramValues[j][indices[i,j].astype(int)]
            
            metric = [self.metric.worst_case()]*self.numberOfFolds
            
            ##
            if len(self.test_fold) == 0:
                for k in range(0,self.numberOfFolds):
                    testFoldIdx = perm[k*tamFold:(k+1)*tamFold]
                    trainFoldIdx = np.setdiff1d(perm, testFoldIdx)
                    
                    if isinstance(self.seedClass, int):
                        kStream = np.random.mtrand.RandomState(self.seedClass)
                    elif isinstance(self.seedClass, np.random.mtrand.RandomState):
                        kStream = self.seedClass
                    
                    kClassifier = self.classifierLambda(dict({'seed':kStream,
                                                             'kfold_indices':trainFoldIdx},**classParams))
                    kClassifier.train(trData[trainFoldIdx,:],trLab[trainFoldIdx,:])
                    pred = kClassifier.predict(trData[testFoldIdx,:])
                    metric[k] = self.metric.calculate(trLab[testFoldIdx,:],pred)
            else:
                metric = []
                trainFoldIdx = list(range(0,trData.shape[0]))
                if isinstance(self.seedClass, int):
                    kStream = np.random.mtrand.RandomState(self.seedClass)
                elif isinstance(self.seedClass, np.random.mtrand.RandomState):
                    kStream = self.seedClass
                
                kClassifier = self.classifierLambda(dict({'seed':kStream,
                                                         'kfold_indices':trainFoldIdx,
                                                         'test_last_hidden':self.test_last_hidden,
                                                         'test_last_output':self.test_last_output},**classParams))
                kClassifier.train(trData,trLab)
                pred,_ = kClassifier.predict(self.test_fold)
                metric.append(self.metric.calculate(self.test_targets,pred))
                
                
            
            ##
            
            # for k in range(0,self.numberOfFolds):
            #     kClassifier = self.classifierLambda(dict({'seed':self.seedClass},**classParams))
                
            #     testFoldIdx = perm[k*tamFold:(k+1)*tamFold]
            #     trainFoldIdx = np.setdiff1d(perm, testFoldIdx)
                
            #     kClassifier.train(trData[trainFoldIdx,:],trLab[trainFoldIdx,:])
            #     pred = kClassifier.predict(trData[testFoldIdx,:])
            #     metric[k] = self.metric.calculate(trLab[testFoldIdx,:],pred)
                
            if i == 1 or self.metric.is_better(np.mean(metric),bestMetric):
                paramStruct = classParams
                bestMetric = np.mean(metric)
        
        return paramStruct,bestMetric


class Module_Error:
    def __init__(self):
        self.error = []
        
class OSFDSN(util.Util):
    def _init_params(self):
        super()._init_params()
        # self.max_number_of_modules = 100
        self.activation_function = 'sig'
        self.number_of_hidden_neurons = 100
        self.number_of_input_neurons = []
        self.number_of_output_neurons = []
        self.regularization_parameter = 1000
        self._stacked_modules = []
        self.use_parallel_layer = False
        self.alpha = False
        self.use_auto_encoder = False
        self.bias_of_hidden_neurons = []
        self.trained = False
        self.isSequential = True
        # from metrics import aRRMSE
        self.metric = metrics.RMSE()
        self.weight_vector = []
        self.window_size = 20
        self.module_life_threshold = 1
        self.MSE = []
        self.YR = []
        self.YP = []
        self.module_life = []
        self.module_error = []
        self.param_grid = [[2**x for x in range(-20,21)]];
        # self.param_grid[1].append(np.inf)
        self.param_names = ['regularization_parameter']
        self.__lastInputData = []
        self.__lastOutputData = []
        self._accepted_params.extend([
            # 'max_number_of_modules',
            'module_life_threshold',
            'window_size'
            'activation_function',
            'number_of_output_neurons',
            'number_of_hidden_neurons',
            'number_of_input_neurons',
            'bias_of_hidden_neurons',
            'regularization_parameter',
            'alpha',
            'use_auto_encoder',
            'param_grid',
            'param_names'
            'use_parallel_layer'
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
        self._stacked_modules = []
    
    def generate_random_weights(self, num_input, num_hidden, X):
        if self.use_auto_encoder:
            input_w = super().generate_random_weights(num_input,num_hidden,X)
            bias = self.seed.randn(1,num_hidden)
            param_dict = {
                'number_of_hidden_neurons': num_hidden,
                'number_of_input_neurons': num_input,
                'use_auto_encoder': False,
                'input_weight' : input_w,
                'bias_of_hidden_neurons': bias,
                'regularization_parameter': self.regularization_parameter}
            if (self.use_parallel_layer):
                i_w2 = super().generate_random_weights(num_input,num_hidden,X)
                param_dict['input_weight2'] = i_w2
            a = _OSFDSNModule(param_dict)
            a.train(X, X, [], [])
            return np.transpose(a.output_weight)
        else:
            return super().generate_random_weights(num_input, num_hidden, X)

    def train(self, X, Y):
        aux_time = time.time()

        # last_layer_out = []
        # last_hidden = []
        # param_dict = {}
        # input_dim = self.number_of_input_neurons
        
        # self.bias_of_hidden_neurons = self.seed.randn(1,self.number_of_hidden_neurons)

        
        
        if len(self._stacked_modules) == 0:
            
            dataSeedFolds = self.seed.randint(0,4294967296)
            dataSeedClassK = self.seed.randint(0,4294967296)
            self.number_of_output_neurons = Y.shape[1]
            from relm import RELM #equivalent to OSRELM/OSFDSNModule when receiving the first batch
            method = lambda x={}: RELM(dict({'number_of_input_neurons':X.shape[1],
                                             'number_of_hidden_neurons':self.number_of_hidden_neurons,**x}))
            
            from kfold import KFold
            k_kfold = min(5,X.shape[0])
            kfold = KFold(k_kfold, method, self.param_names,self.param_grid,self.metric,
                          False,False,dataSeedFolds,dataSeedClassK)
            
            paramStruct,bestMetric = kfold.start(X,Y)
            
        
             
            self.input_weight = self.generate_random_weights(self.number_of_input_neurons, 
                                                             self.number_of_hidden_neurons, X)
            self.bias_of_hidden_neurons = self.seed.randn(1,self.number_of_hidden_neurons)
            
            self.__addModule(X, Y, [], [])
            
            self.module_life = [-1]
            self.MSE = [np.zeros((1,self.number_of_output_neurons))]
            self.module_error.append(Module_Error())
            self.__lastInputData = X
            self.__lastOutputData = Y
            self.weight_vector = [1]
            
        else:
            pred_list,lastHiddenBeforeAct = self.__predict_modules(X)
            lastLayerOutput = pred_list[-1]
            
            self.YP.append(self.__weight_list(pred_list))
            self.YR.append(Y)
            
            ## kfold (holdout, actually)
            dataSeedFolds = self.seed.randint(0,4294967296)
            dataSeedClassK = self.seed.randint(0,4294967296)
            
            
            method = lambda x={}: _kfold_osfdnmodule(dict({'number_of_input_neurons':Y.shape[1],
                                                            'total_number_of_input':X.shape[1],
                                                            'number_of_hidden_neurons':self.number_of_hidden_neurons,
                                                            'is_first_layer':False,
                                                            'last_hidden_before_act':lastHiddenBeforeAct,
                                                            'last_layer_output':lastLayerOutput,**x}))
            
            k_kfold = min(5,X.shape[0])
            kfold = _KFold_osfdsn(k_kfold,method, self.param_names, self.param_grid, self.metric, 
                                 False, False, dataSeedFolds, dataSeedClassK)
            
            kfold.test_fold = self.__lastInputData
            kfold.test_targets = self.__lastOutputData
            
            [pred_list,kfold.test_last_hidden] = self.__predict_modules(self.__lastInputData)
            kfold.test_last_output = pred_list[-1]
            
            
            paramStruct,bestMetric = kfold.start(X,Y)
            
            for key, val in paramStruct.items():
                if key in self._accepted_params:
                    setattr(self, key, val)
                else:
                    raise NameError("Parameter does not exist!")
                    
            #Caso 1
            self.__addModule([], Y, lastHiddenBeforeAct, lastLayerOutput)
            
            lastSetOutput_Module = self.__predict_without_weights(self.__lastInputData)
            
            Case_Modules = [copy.deepcopy(self._stacked_modules)]
            self._stacked_modules.pop()
            
            #Caso 2
            _,lastLayerOutput = self.__update_last_module(X, Y)
            Case_Modules.append(copy.deepcopy(self._stacked_modules))
            lastSetOutput_UpdateLast = self.__predict_without_weights(self.__lastInputData)
            
            metricz = [self.metric.calculate(lastSetOutput_Module,self.__lastOutputData)]
            metricz.append(self.metric.calculate(lastSetOutput_UpdateLast,self.__lastOutputData))
            
            bestArg = 1 # se empatar, atualiza
            bestMetric = metricz[bestArg]
            for i in range(0,len(metricz)):
                if self.metric.is_better(metricz[i],bestMetric):
                    bestArg = i
                    bestMetric = metricz[bestArg]
                    
            self._stacked_modules = Case_Modules[bestArg]
            self.__lastInputData = X
            self.__lastOutputData = Y
            
            if bestArg == 0:
                self.module_life.append(-1)
                self.MSE.append(np.zeros((1,self.number_of_output_neurons)))
                self.module_error.append(Module_Error())
                self.weight_vector = np.append(self.weight_vector,0)
            
        numMods = len(self._stacked_modules)
        
        outputs,_ = self.__predict_modules(X)
        
        
        for k in range(0,numMods):
            self.module_life[k] += 1
            eNow = np.power(Y - outputs[k],2)
            eNow = np.mean(eNow)
            self.module_error[k].error.append(eNow)
            if self.module_life[k] == 0:
                self.MSE[k] = eNow
            elif self.module_life[k] < self.module_life_threshold:
                self.MSE[k] = ((self.module_life[k]-1)/self.module_life[k])*self.MSE[k] + \
                    (1/self.module_life[k])*eNow
            else:
                self.MSE[k] = self.MSE[k] + eNow/self.module_life_threshold - \
                    self.module_error[k].error[0]/self.module_life_threshold
                self.module_error[k].error = self.module_error[k].error[1:]
        
        for k in range(0,numMods):
            self.weight_vector[k] = self.MSE[k] - np.median(self.MSE)
            eps = np.finfo(float).eps
            self.weight_vector[k] = self.weight_vector[k]/(np.median(self.MSE) + eps)
            self.weight_vector[k] = np.exp(-self.weight_vector[k])
            
        self.weight_vector = np.asarray(self.weight_vector)
        self.weight_vector = self.weight_vector/np.sum(self.weight_vector)
            
            
            
            # pass
        

        # while (len(self._stacked_modules) < self.max_number_of_modules):
        #     param_dict['number_of_input_neurons'] = input_dim
        #     input_weight1 = self.generate_random_weights(
        #         input_dim,
        #         self.number_of_hidden_neurons,
        #         X)  # readability?
        #     param_dict['input_weight'] = input_weight1
        #     if self.use_parallel_layer:
        #         input_weight2 = self.generate_random_weights(
        #             input_dim,
        #             self.number_of_hidden_neurons,
        #             X)  # readability?
        #         param_dict['input_weight2'] = input_weight2
            
        #     new_module = _OSFDSNModule(param_dict)
            
        #     last_hidden, last_layer_out = new_module.train(
        #         X , Y, last_hidden, last_layer_out)
        #     # train
        #     X = last_layer_out
        #     input_dim = Y.shape[1]
        #     param_dict['bias_of_hidden_neurons'] = []
        #     self._stacked_modules.append(new_module)
        self.train_time = time.time() - aux_time
        # self.bias_of_hidden_neurons = []

    def __predict_without_weights(self, X):
        aux_time = time.time()
        last_layer_out = []
        last_hidden = []
        for module in self._stacked_modules:
            [last_layer_out, last_hidden] = module.predict(
                X, last_hidden, last_layer_out)
        self.last_test_time = time.time() - aux_time
        return last_layer_out
    
    def __addModule(self, X,Y,lastHiddenBeforeAct,lastLayerOutput):
        param_dict = {}
        if len(self._stacked_modules) == 0:
            param_dict['number_of_input_neurons'] = self.number_of_input_neurons
        else:
            param_dict['number_of_input_neurons'] = self.number_of_output_neurons
        param_dict['input_weight'] = self.input_weight
        param_dict['number_of_output_neurons'] = self.number_of_output_neurons
        param_dict['number_of_hidden_neurons'] = self.number_of_hidden_neurons
        param_dict['regularization_parameter'] = self.regularization_parameter
        param_dict['seed'] = self.seed
        param_dict['activation_function'] = self.activation_function
        # param_dict['use_random_orthogonalization'] = self.use_random_orthogonalization
        # param_dict['use_parallel_layer'] = self.use_parallel_layer
        # param_dict['alpha'] = self.alpha
        # param_dict['bias_of_hidden_neurons'] = self.bias_of_hidden_neurons
        
        new_module = _OSFDSNModule(param_dict)
        [last_hidden,last_layer_out] = new_module.train(X,Y, self.bias_of_hidden_neurons,
                                              lastHiddenBeforeAct,lastLayerOutput)
        self._stacked_modules.append(new_module)
        return [last_hidden,last_layer_out]
    
    def __predict_modules(self,inputData):
        pred_list = []
        llo = []
        lhba = []
        
        for i in range(0,len(self._stacked_modules)):
            [llo,lhba] = self._stacked_modules[i].predict(inputData,lhba,llo)
            pred_list.append(llo)
        
        return pred_list,lhba
    
    def __weight_list(self,pred_list):
        pred = np.zeros(pred_list[0].shape)
        for i in range(0,len(pred_list)):
            pred += self.weight_vector[i]*pred_list[i]
        return pred
            
    def predict(self,X):
        pred_list,_ = self.__predict_modules(X)
        pred = np.zeros(pred_list[0].shape)
        for i in range(0,len(pred_list)):
            pred += self.weight_vector[i]*pred_list[i]
        return pred
    
    def __update_last_module(self,X,Y):
        if len(self._stacked_modules) > 1:
            llo = []
            lhba = []
            for i in range(0,len(self._stacked_modules)-1):
                [llo,lhba] = self._stacked_modules[i].predict(X,lhba,llo)
            lhba,llo = self._stacked_modules[-1].train([],Y,[],lhba,llo)
        else:
            lhba,llo = self._stacked_modules[-1].train(X,Y,self.bias_of_hidden_neurons,[],[])
        return lhba,llo
        
        
class OSFDSN_init(util.Util):
    def _init_params(self):
        super()._init_params()
        # self.max_number_of_modules = 100
        self.activation_function = 'sig'
        self.number_of_hidden_neurons = 100
        self.number_of_input_neurons = []
        self.number_of_output_neurons = []
        self.regularization_parameter = 1000
        self._stacked_modules = []
        self.use_parallel_layer = False
        self.alpha = False
        self.use_auto_encoder = False
        self.bias_of_hidden_neurons = []
        self.trained = False
        self.isSequential = True
        # from metrics import aRRMSE
        self.metric = metrics.RMSE()
        self.weight_vector = []
        self.window_size = 20
        self.module_life_threshold = 1
        self.MSE = []
        self.YR = []
        self.YP = []
        self.module_life = []
        self.module_error = []
        self.param_grid = [[2**x for x in range(-20,21)]];
        # self.param_grid[1].append(np.inf)
        self.param_names = ['regularization_parameter']
        self.__lastInputData = []
        self.__lastOutputData = []
        self._accepted_params.extend([
            # 'max_number_of_modules',
            'module_life_threshold',
            'window_size'
            'activation_function',
            'number_of_output_neurons',
            'number_of_hidden_neurons',
            'number_of_input_neurons',
            'bias_of_hidden_neurons',
            'regularization_parameter',
            'alpha',
            'use_auto_encoder',
            'param_grid',
            'param_names'
            'use_parallel_layer'
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
        self._stacked_modules = []
    
    def generate_random_weights(self, num_input, num_hidden, X):
        if self.use_auto_encoder:
            input_w = super().generate_random_weights(num_input,num_hidden,X)
            bias = self.seed.randn(1,num_hidden)
            param_dict = {
                'number_of_hidden_neurons': num_hidden,
                'number_of_input_neurons': num_input,
                'use_auto_encoder': False,
                'input_weight' : input_w,
                'bias_of_hidden_neurons': bias,
                'regularization_parameter': self.regularization_parameter}
            if (self.use_parallel_layer):
                i_w2 = super().generate_random_weights(num_input,num_hidden,X)
                param_dict['input_weight2'] = i_w2
            a = _OSFDSNModule(param_dict)
            a.train(X, X, [], [])
            return np.transpose(a.output_weight)
        else:
            return super().generate_random_weights(num_input, num_hidden, X)

    def train(self, X, Y):
        aux_time = time.time()

        # last_layer_out = []
        # last_hidden = []
        # param_dict = {}
        # input_dim = self.number_of_input_neurons
        
        # self.bias_of_hidden_neurons = self.seed.randn(1,self.number_of_hidden_neurons)

        
        
        if len(self._stacked_modules) == 0:
            
            dataSeedFolds = self.seed.randint(0,4294967296)
            dataSeedClassK = self.seed.randint(0,4294967296)
            self.number_of_output_neurons = Y.shape[1]
            from relm import RELM #equivalent to OSRELM/OSFDSNModule when receiving the first batch
            method = lambda x={}: RELM(dict({'number_of_input_neurons':X.shape[1],
                                             'number_of_hidden_neurons':self.number_of_hidden_neurons,**x}))
            
            from kfold import KFold
            k_kfold = min(5,X.shape[0])
            kfold = KFold(k_kfold, method, self.param_names,self.param_grid,self.metric,
                          False,False,dataSeedFolds,dataSeedClassK)
            
            paramStruct,bestMetric = kfold.start(X,Y)
            
        
             
            self.input_weight = self.generate_random_weights(self.number_of_input_neurons, 
                                                             self.number_of_hidden_neurons, X)
            self.bias_of_hidden_neurons = self.seed.randn(1,self.number_of_hidden_neurons)
            
            self.__addModule(X, Y, [], [])
            
            self.module_life = [-1]
            self.MSE = [np.zeros((1,self.number_of_output_neurons))]
            self.module_error.append(Module_Error())
            self.__lastInputData = X
            self.__lastOutputData = Y
            self.weight_vector = [1]
            
        else:
            pred_list,lastHiddenBeforeAct = self.__predict_modules(X)
            lastLayerOutput = pred_list[-1]
            
            self.YP.append(self.__weight_list(pred_list))
            self.YR.append(Y)
            
            # ## kfold (holdout, actually)
            # dataSeedFolds = self.seed.randint(0,4294967296)
            # dataSeedClassK = self.seed.randint(0,4294967296)
            
            
            # method = lambda x={}: _kfold_osfdnmodule(dict({'number_of_input_neurons':Y.shape[1],
            #                                                 'total_number_of_input':X.shape[1],
            #                                                 'number_of_hidden_neurons':self.number_of_hidden_neurons,
            #                                                 'is_first_layer':False,
            #                                                 'last_hidden_before_act':lastHiddenBeforeAct,
            #                                                 'last_layer_output':lastLayerOutput,**x}))
            
            # k_kfold = min(5,X.shape[0])
            # kfold = _KFold_osfdsn(k_kfold,method, self.param_names, self.param_grid, self.metric, 
            #                      False, False, dataSeedFolds, dataSeedClassK)
            
            # kfold.test_fold = self.__lastInputData
            # kfold.test_targets = self.__lastOutputData
            
            # [pred_list,kfold.test_last_hidden] = self.__predict_modules(self.__lastInputData)
            # kfold.test_last_output = pred_list[-1]
            
            
            # paramStruct,bestMetric = kfold.start(X,Y)
            
            # for key, val in paramStruct.items():
            #     if key in self._accepted_params:
            #         setattr(self, key, val)
            #     else:
            #         raise NameError("Parameter does not exist!")
                    
            #Caso 1
            self.__addModule([], Y, lastHiddenBeforeAct, lastLayerOutput)
            
            lastSetOutput_Module = self.__predict_without_weights(self.__lastInputData)
            
            Case_Modules = [copy.deepcopy(self._stacked_modules)]
            self._stacked_modules.pop()
            
            #Caso 2
            _,lastLayerOutput = self.__update_last_module(X, Y)
            Case_Modules.append(copy.deepcopy(self._stacked_modules))
            lastSetOutput_UpdateLast = self.__predict_without_weights(self.__lastInputData)
            
            metricz = [self.metric.calculate(lastSetOutput_Module,self.__lastOutputData)]
            metricz.append(self.metric.calculate(lastSetOutput_UpdateLast,self.__lastOutputData))
            
            bestArg = 1 # se empatar, atualiza
            bestMetric = metricz[bestArg]
            for i in range(0,len(metricz)):
                if self.metric.is_better(metricz[i],bestMetric):
                    bestArg = i
                    bestMetric = metricz[bestArg]
                    
            self._stacked_modules = Case_Modules[bestArg]
            self.__lastInputData = X
            self.__lastOutputData = Y
            
            if bestArg == 0: #adiciona novo módulo, então prepara as variáveis...
                self.module_life.append(-1)
                self.MSE.append(np.zeros((1,self.number_of_output_neurons)))
                self.module_error.append(Module_Error())
                self.weight_vector = np.append(self.weight_vector,0)
            
        numMods = len(self._stacked_modules)
        
        outputs,_ = self.__predict_modules(X)
        
        
        for k in range(0,numMods):
            self.module_life[k] += 1
            eNow = np.power(Y - outputs[k],2)
            eNow = np.mean(eNow)
            self.module_error[k].error.append(eNow)
            if self.module_life[k] == 0:
                self.MSE[k] = eNow
            elif self.module_life[k] < self.module_life_threshold:
                self.MSE[k] = ((self.module_life[k]-1)/self.module_life[k])*self.MSE[k] + \
                    (1/self.module_life[k])*eNow
            else:
                self.MSE[k] = self.MSE[k] + eNow/self.module_life_threshold - \
                    self.module_error[k].error[0]/self.module_life_threshold
                self.module_error[k].error = self.module_error[k].error[1:]
        
        for k in range(0,numMods):
            self.weight_vector[k] = self.MSE[k] - np.median(self.MSE)
            eps = np.finfo(float).eps
            self.weight_vector[k] = self.weight_vector[k]/(np.median(self.MSE) + eps)
            self.weight_vector[k] = np.exp(-self.weight_vector[k])
            
        self.weight_vector = np.asarray(self.weight_vector)
        self.weight_vector = self.weight_vector/np.sum(self.weight_vector)
            
            
            
            # pass
        

        # while (len(self._stacked_modules) < self.max_number_of_modules):
        #     param_dict['number_of_input_neurons'] = input_dim
        #     input_weight1 = self.generate_random_weights(
        #         input_dim,
        #         self.number_of_hidden_neurons,
        #         X)  # readability?
        #     param_dict['input_weight'] = input_weight1
        #     if self.use_parallel_layer:
        #         input_weight2 = self.generate_random_weights(
        #             input_dim,
        #             self.number_of_hidden_neurons,
        #             X)  # readability?
        #         param_dict['input_weight2'] = input_weight2
            
        #     new_module = _OSFDSNModule(param_dict)
            
        #     last_hidden, last_layer_out = new_module.train(
        #         X , Y, last_hidden, last_layer_out)
        #     # train
        #     X = last_layer_out
        #     input_dim = Y.shape[1]
        #     param_dict['bias_of_hidden_neurons'] = []
        #     self._stacked_modules.append(new_module)
        self.train_time = time.time() - aux_time
        # self.bias_of_hidden_neurons = []

    def __predict_without_weights(self, X):
        aux_time = time.time()
        last_layer_out = []
        last_hidden = []
        for module in self._stacked_modules:
            [last_layer_out, last_hidden] = module.predict(
                X, last_hidden, last_layer_out)
        self.last_test_time = time.time() - aux_time
        return last_layer_out
    
    def __addModule(self, X,Y,lastHiddenBeforeAct,lastLayerOutput):
        param_dict = {}
        if len(self._stacked_modules) == 0:
            param_dict['number_of_input_neurons'] = self.number_of_input_neurons
        else:
            param_dict['number_of_input_neurons'] = self.number_of_output_neurons
        param_dict['input_weight'] = self.input_weight
        param_dict['number_of_output_neurons'] = self.number_of_output_neurons
        param_dict['number_of_hidden_neurons'] = self.number_of_hidden_neurons
        param_dict['regularization_parameter'] = self.regularization_parameter
        param_dict['seed'] = self.seed
        param_dict['activation_function'] = self.activation_function
        # param_dict['use_random_orthogonalization'] = self.use_random_orthogonalization
        # param_dict['use_parallel_layer'] = self.use_parallel_layer
        # param_dict['alpha'] = self.alpha
        # param_dict['bias_of_hidden_neurons'] = self.bias_of_hidden_neurons
        
        new_module = _OSFDSNModule(param_dict)
        [last_hidden,last_layer_out] = new_module.train(X,Y, self.bias_of_hidden_neurons,
                                              lastHiddenBeforeAct,lastLayerOutput)
        self._stacked_modules.append(new_module)
        return [last_hidden,last_layer_out]
    
    def __predict_modules(self,inputData):
        pred_list = []
        llo = []
        lhba = []
        
        for i in range(0,len(self._stacked_modules)):
            [llo,lhba] = self._stacked_modules[i].predict(inputData,lhba,llo)
            pred_list.append(llo)
        
        return pred_list,lhba
    
    def __weight_list(self,pred_list):
        pred = np.zeros(pred_list[0].shape)
        for i in range(0,len(pred_list)):
            pred += self.weight_vector[i]*pred_list[i]
        return pred
            
    def predict(self,X):
        pred_list,_ = self.__predict_modules(X)
        pred = np.zeros(pred_list[0].shape)
        for i in range(0,len(pred_list)):
            pred += self.weight_vector[i]*pred_list[i]
        return pred
    
    def __update_last_module(self,X,Y):
        if len(self._stacked_modules) > 1:
            llo = []
            lhba = []
            for i in range(0,len(self._stacked_modules)-1):
                [llo,lhba] = self._stacked_modules[i].predict(X,lhba,llo)
            lhba,llo = self._stacked_modules[-1].train([],Y,[],lhba,llo)
        else:
            lhba,llo = self._stacked_modules[-1].train(X,Y,self.bias_of_hidden_neurons,[],[])
        return lhba,llo

if __name__ == '__main__':
    # a = OSFDSN_init({'number_of_input_neurons':4,'seed':0,'number_of_hidden_neurons':100})
    # from relm import RELM
    # b = RELM({'number_of_input_neurons':4,'seed':0,'number_of_hidden_neurons':100})
    # X = np.random.randn(150, 4)
    # Y = np.random.randint(2, size=(150, 1))
    # a.train(X, Y)
    # b.train(X, Y)
    # yh = a.predict(X)
    # yh2 = b.predict(X)
    import CHATAO
    CHATAO.__main__()
