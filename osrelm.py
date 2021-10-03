#!/usr/bin/env python3

import elm
import numpy as np
import time


class OSRELM(elm.ELM):
    """docstring for OSRELM."""

    def _init_params(self):
        super()._init_params()
        self.regularization_parameter = 1000
        self.hMat = []
        self.tMat = []
        self.pMat = []
        self._accepted_params.append('regularization_parameter')

    def __init__(self, paramDict):
        super(OSRELM, self).__init__(paramDict)

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
        C = self.regularization_parameter
        if self.pMat == []:
            if H.shape[0] < self.number_of_hidden_neurons:
                self.hMat = H
                self.tMat = Y
                del H, Y
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
                    del H, Y
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
                    del H, Y, A, B, C, invS
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


if __name__ == "__main__":
    a = OSRELM({'number_of_input_neurons': 4,
                'number_of_hidden_neurons': 1000})
    X = np.random.randn(150, 4)
    Y = np.random.randint(2, size=(150, 1))
    a.train(X[:100, :], Y[:100, :])
    yh = a.predict(X)
    a.train(X[100:, :], Y[100:, :])
    yh = a.predict(X)
