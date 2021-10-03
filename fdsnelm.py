#!/usr/bin/env python3

import relm
import time
import util
import numpy as np

def _add_duples(t1,t2):
    aux1 = t1[0] + t2[0]
    aux2 = t1[1] + t2[1]
    return (aux1,aux2)
    

class _FDSNELMModule(relm.RELM):

    def _init_params(self):
        super()._init_params()
        self._total_number_of_input = 0
        self._accepted_params.extend([
                'input_weight', 
                'input_weight2',
                'bias_of_hidden_neurons'
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

    def train(self, X, Y, last_hidden, last_layer_out):
        aux_time = time.time()
        (H, last_hidden) = self.hidden_layer_output(
            X, last_hidden, last_layer_out)
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
        return (last_hidden,np.matmul(H,self.output_weight))

    def predict(self, X, last_hidden, last_layer_out):
        H, last_hidden = self.hidden_layer_output(
            X, last_hidden, last_layer_out)
        out = np.matmul(H, self.output_weight)
        return (out, last_hidden)


class FDSNELM(util.Util):
    def _init_params(self):
        super()._init_params()
        self.max_number_of_modules = 100
        self.activation_function = 'sig'
        self.number_of_hidden_neurons = 100
        self.number_of_input_neurons = []
        self.regularization_parameter = 1000
        self._stacked_modules = []
        self.use_parallel_layer = False
        self.alpha = False
        self.use_auto_encoder = False
        self.bias_of_hidden_neurons = []
        self._accepted_params.extend([
            'max_number_of_modules',
            'activation_function',
            'number_of_hidden_neurons',
            'number_of_input_neurons',
            'bias_of_hidden_neurons',
            'regularization_parameter',
            'alpha',
            'use_auto_encoder',
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
        self._stackedModules = []
    
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
            a = _FDSNELMModule(param_dict)
            a.train(X, X, [], [])
            return np.transpose(a.output_weight)
        else:
            return super().generate_random_weights(num_input, num_hidden, X)

    def train(self, X, Y):
        aux_time = time.time()

        last_layer_out = []
        last_hidden = []
        param_dict = {}
        input_dim = self.number_of_input_neurons
        
        self.bias_of_hidden_neurons = self.seed.randn(1,self.number_of_hidden_neurons)
        param_dict['number_of_hidden_neurons'] = self.number_of_hidden_neurons
        param_dict['regularization_parameter'] = self.regularization_parameter
        param_dict['seed'] = self.seed
        param_dict['activation_function'] = self.activation_function
        param_dict['use_random_orthogonalization'] = self.use_random_orthogonalization
        param_dict['use_parallel_layer'] = self.use_parallel_layer
        param_dict['alpha'] = self.alpha
        param_dict['bias_of_hidden_neurons'] = self.bias_of_hidden_neurons

        while (len(self._stacked_modules) < self.max_number_of_modules):
            param_dict['number_of_input_neurons'] = input_dim
            input_weight1 = self.generate_random_weights(
                input_dim,
                self.number_of_hidden_neurons,
                X)  # readability?
            param_dict['input_weight'] = input_weight1
            if self.use_parallel_layer:
                input_weight2 = self.generate_random_weights(
                    input_dim,
                    self.number_of_hidden_neurons,
                    X)  # readability?
                param_dict['input_weight2'] = input_weight2
            
            new_module = _FDSNELMModule(param_dict)
            
            last_hidden, last_layer_out = new_module.train(
                X , Y, last_hidden, last_layer_out)
            # train
            X = last_layer_out
            input_dim = Y.shape[1]
            param_dict['bias_of_hidden_neurons'] = []
            self._stacked_modules.append(new_module)
        self.train_time = time.time() - aux_time
        self.bias_of_hidden_neurons = []

    def predict(self, X):
        aux_time = time.time()
        last_layer_out = []
        last_hidden = []
        for module in self._stacked_modules:
            [last_layer_out, last_hidden] = module.predict(
                X, last_hidden, last_layer_out)
        self.last_test_time = time.time() - aux_time
        return last_layer_out


if __name__ == '__main__':
    a = FDSNELM({'number_of_input_neurons': 4})
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=(150, 1))
    a.train(X, Y)
    yh = a.predict(X)
