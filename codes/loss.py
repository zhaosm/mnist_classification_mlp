from __future__ import division
import numpy as np
import scipy.sparse as sparse


# def extendTarget(target, colnum):
#     length = len(target)
#     data = [1.0] * length
#     row = np.linspace(0, length - 1, dtype=int)
#     col = np.transpose(target)
#     answer = sparse.coo_matrix((data, (row, col)), shape=(length, colnum)).toarray()
#     return answer


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        dis = (input - target)**2
        dis = np.sum(dis) / (2 * input.shape[0])
        return dis

    def backward(self, input, target):
        '''Your codes here'''
        dis = (input - target) / input.shape[0]
        return dis
