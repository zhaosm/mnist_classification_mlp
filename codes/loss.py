from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        exp_vector = np.exp(input)
        sum_vector = np.sum(exp_vector, axis=1)
        self._p_vector = exp_vector / (sum_vector.reshape(-1, 1))
        ln_p_vector = np.log(self._p_vector)
        return -np.mean(target * ln_p_vector)

    def backward(self, input, target):
        '''Your codes here'''
        return (self._p_vector - target) / input.shape[0]

