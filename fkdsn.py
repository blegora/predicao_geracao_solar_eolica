#!/usr/bin/env python3

import rkelm
import time
import util
import numpy as np

def _add_duples(t1,t2):
    aux1 = t1[0] + t2[0]
    aux2 = t1[1] + t2[1]
    return (aux1,aux2)
    

class _FKDSNModule(rkelm.RKELM):

    def _init_params(self):
        super()._init_params()
        self._total_number_of_input = 0
        self.is_first_layer = True
        self._accepted_params.extend([
                'input_weight', 
                'input_weight2',
                '_support',
                'is_first_layer'
        ])
        
    def __init__(self, param_dict={}):
        self._init_params()
        for key, val in param_dict.items():
            if key in self._accepted_params:
                setattr(self, key, val)
            else:
                raise NameError("Parameter does not exist!")
        if not type(self.kernel_param) == list:
            self.kernel_param = [self.kernel_param] #fix....
                    
        
    def _kernel_matrix(self, last_omega, Xte=[]):
        num_samples = self.Xtr.shape[0]
        flag = isinstance(Xte, list) and len(Xte) == 0
        if self.kernel_type == 'RBF_kernel':
            if flag:
                XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
                XXh = np.matmul(XXh, np.ones((1, num_samples)))
                last_omega_before = last_omega + XXh + np.transpose(XXh) - 2 * np.matmul(
                    self.Xtr,
                    np.transpose(self.Xtr))
                omega = np.exp(-last_omega_before/self.kernel_param[0])
            else:
                XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
                XXh = np.matmul(XXh, np.ones((1, Xte.shape[0])))
                XXh2 = np.sum(Xte*Xte, axis=1, keepdims=True)
                XXh2 = np.matmul(XXh2, np.ones((1, num_samples)))
                last_omega_before = last_omega + XXh + np.transpose(XXh2) - 2 * np.matmul(
                    self.Xtr,
                    np.transpose(Xte))
                omega = np.exp(-last_omega_before/self.kernel_param[0])
        elif self.kernel_type == 'lin_kernel':
            if flag:
                omega = last_omega + np.matmul(self.Xtr, np.transpose(self.Xtr))
            else:
                omega = last_omega + np.matmul(self.Xtr, np.transpose(Xte))
            last_omega_before = omega;
        elif self.kernel_type == 'poly_kernel':
            if flag:
                last_omega_before  = np.matmul(self.Xtr, np.transpose(self.Xtr))
                last_omega_before = last_omega_before + last_omega + self.kernel_param[0]
                omega = np.power(last_omega_before, self.kernel_param[1])
            else:
                last_omega_before = np.matmul(self.Xtr, np.transpose(Xte))
                last_omega_before = last_omega_before + last_omega + self.kernel_param[0]
                omega = np.power(last_omega_before, self.kernel_param[1])
        elif self.kernel_type == 'wav_kernel':
            # Not sure...
            raise NameError("wav_kernel not implemented yet...")
            # if flag:
            #     XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
            #     XXh = np.matmul(XXh, np.ones((1, num_samples)))
            #     omega = XXh + np.transpose(XXh)
            #     omega = omega - 2*np.matmul(self.Xtr, np.transpose(self.Xtr))
            #     XXh1 = np.sum(self.Xtr, axis=1, keepdims=True)
            #     XXh1 = np.matmul(XXh1, np.ones((1, num_samples)))
            #     omega1 = XXh1 - np.transpose(XXh1)
            #     omega = np.cos(
            #         self.kernel_param[2] * omega1 / self.kernel_param[1])
            #     omega = omega*np.exp(-omega/self.kernel_param[0])
            # else:
            #     XXh = np.sum(self.Xtr*self.Xtr, axis=1, keepdims=True)
            #     XXh = np.matmul(XXh, np.ones((1, Xte.shape[0])))
            #     XXh2 = np.sum(Xte*Xte, axis=1, keepdims=True)
            #     XXh2 = np.matmul(XXh2, np.ones(1, num_samples))
            #     omega = XXh + np.transpose(XXh2) - 2 * np.matmul(
            #         self.Xtr,
            #         np.transpose(Xte))

            #     XXh11 = np.sum(self.Xtr, axis=1)
            #     XXh11 = np.matmul(XXh11, np.ones((1, Xte.shape[0])))
            #     XXh22 = np.sum(Xte, axis=1)
            #     XXh22 = np.matmul(XXh22, np.ones((1, num_samples)))
            #     omega1 = XXh11 - np.transpose(XXh22)
            #     omega = np.cos(
            #         self.kernel_param[2] * omega1 / self.kernel_param[1])
            #     omega = omega*np.exp(-omega/self.kernel_param[0])
        return omega,last_omega_before

    def train(self, X, Y, last_omega, last_layer_out):
        aux_time = time.time()
        
        Nsamples = X.shape[0]
        Nsupp = self._support.shape[0]
        
        if self.is_first_layer:
            last_omega = np.zeros((Nsupp,Nsamples))
            self.Xtr = X[self._support,:]
        else:
            self.Xtr = last_layer_out[self._support,:]
        
        if self.is_first_layer:
            Omega_train, last_omega = self._kernel_matrix(last_omega,X)
        else:
            Omega_train, last_omega = self._kernel_matrix(last_omega,last_layer_out)
            
        Omega_train = Omega_train.transpose()
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
        return (last_omega,np.matmul(Omega_train,self.output_weight))

    def predict(self, X, last_omega, last_layer_out):

        Nsamples = X.shape[0]
        if self.is_first_layer:
            last_omega = np.zeros((self.Xtr.shape[0],Nsamples))
            out,last_omega = self._kernel_matrix(last_omega,X)
        else:
            out,last_omega = self._kernel_matrix(last_omega,last_layer_out)

        out = np.matmul(out.transpose(), self.output_weight)
        return (out, last_omega)


class FKDSN(util.Util):
    def _init_params(self):
        super()._init_params()
        self.max_number_of_modules = 10
        self.kernel_type = 'RBF_kernel'
        self.kernel_param = 0.1
        # self.activation_function = 'sig'
        self.number_of_hidden_neurons = 100
        self.number_of_input_neurons = []
        self.regularization_parameter = 1000
        self._stacked_modules = []
        self.use_parallel_layer = False
        self.alpha = False
        self.use_auto_encoder = False
        self.__support = []
        # self.bias_of_hidden_neurons = []
        self._accepted_params.extend([
            'max_number_of_modules',
            'kernel_type',
            'kernel_param',
            'number_of_hidden_neurons',
            'number_of_input_neurons',
            'regularization_parameter',
            'alpha',
            'use_auto_encoder',
            'use_parallel_layer'
        ])
    
    def __init__(self, param_dict={}):
        self._init_params()
        if len(param_dict) > 0:
            for key, val in param_dict.items():
                if key in self._accepted_params:
                    setattr(self, key, val)
                else:
                    raise NameError("Parameter does not exist!")
        # if self.number_of_input_neurons == []:
            # raise NameError("Number of input neurons not defined!")
        # self.activation_function = self.parse_activation_function(
        #     self.activation_function)
        self.parse_seed(self.seed)  # hw to mk more readable Util.parse_seed?
        self._stacked_modules = []
    
    # def generate_random_weights(self, num_input, num_hidden, X):
    #     if self.use_auto_encoder:
    #         input_w = super().generate_random_weights(num_input,num_hidden,X)
    #         bias = self.seed.randn(1,num_hidden)
    #         param_dict = {
    #             'number_of_hidden_neurons': num_hidden,
    #             'number_of_input_neurons': num_input,
    #             'use_auto_encoder': False,
    #             'input_weight' : input_w,
    #             'bias_of_hidden_neurons': bias,
    #             'regularization_parameter': self.regularization_parameter}
    #         if (self.use_parallel_layer):
    #             i_w2 = super().generate_random_weights(num_input,num_hidden,X)
    #             param_dict['input_weight2'] = i_w2
    #         a = _FKDSNModule(param_dict)
    #         a.train(X, X, [], [])
    #         return np.transpose(a.output_weight)
    #     else:
    #         return super().generate_random_weights(num_input, num_hidden, X)

    def train(self, X, Y):
        aux_time = time.time()

        last_layer_out = []
        last_omega = []
        param_dict = {}
        # input_dim = X.shape[1]
        # output_dim = Y.shape[1]
        
        if self.number_of_hidden_neurons > X.shape[0]:
            self.__support = self.seed.permutation(X.shape[0])
        else:
            self.__support = self.seed.permutation(X.shape[0])[:self.number_of_hidden_neurons]
        
        
        # self.bias_of_hidden_neurons = self.seed.randn(1,self.number_of_hidden_neurons)
        param_dict['number_of_hidden_neurons'] = self.number_of_hidden_neurons
        # param_dict['number_of_input_neurons'] = self.number_of_input_neurons
        param_dict['regularization_parameter'] = self.regularization_parameter
        param_dict['seed'] = self.seed
        # param_dict['activation_function'] = self.activation_function
        # param_dict['use_random_orthogonalization'] = self.use_random_orthogonalization
        # param_dict['use_parallel_layer'] = self.use_parallel_layer
        # param_dict['alpha'] = self.alpha
        # param_dict['bias_of_hidden_neurons'] = self.bias_of_hidden_neurons
        param_dict['is_first_layer'] = True
        param_dict['_support'] = self.__support
        param_dict['kernel_type'] = self.kernel_type
        param_dict['kernel_param'] = self.kernel_param

        while (len(self._stacked_modules) < self.max_number_of_modules):
            new_module = _FKDSNModule(param_dict)
            [last_omega, last_layer_out] = new_module.train(X, Y, last_omega,  last_layer_out)
            # lastInputDim = lastInputDim + Y.shape[1]
            self._stacked_modules.append(new_module)
            param_dict['is_first_layer'] = False
            

        self.train_time = time.time() - aux_time
        self.bias_of_hidden_neurons = []

    def predict(self, X):
        aux_time = time.time()
        last_layer_out = []
        last_omega = []
        for module in self._stacked_modules:
            [last_layer_out, last_omega] = module.predict(
                X, last_omega, last_layer_out)
        self.last_test_time = time.time() - aux_time
        
        return last_layer_out


if __name__ == '__main__':
    a = FKDSN({'number_of_input_neurons':4})
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=(150, 1))
    a.train(X, Y)
    yh = a.predict(X)
    
