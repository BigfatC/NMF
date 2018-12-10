#!/usr/bin/python3
# encoding: utf-8
"""
@author: Erin Cai
@contact: charlottecaiir@gmail.com
@file: NMF_feature_extraction.py
@time: 18-12-3 下午8:28
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

n_row = 2
n_col = 4
image_shape = (64,64)


def train(V, components, iternum, e):
    '''
    Non-negative Matrix factorization Function
    :param V: Initial Matrix
    :param components: Number of feature to extract
    :param iternum: Iteration times
    :param e: The threshold of error
    :return:
    '''
    V = V.T #Transpose the matrix
    m,n = V.shape #Return the rows and cols
    # To random initiate two matrix
    W = np.random.random((m, components)) # 4096 * 8
    H = np.random.random((components, n)) # 8 * 64

    for iter in range(iternum):
        V_pre = np.dot(W, H)
        E = V - V_pre

        err = np.sum(E*E)
        temp_str = "In iteration:" + str(iter) +"th, Error is" + str(err)
        print(temp_str)
        if err < e:
            break
        a = np.dot(W.T, V)
        b = np.dot(W.T, np.dot(W, H))
        H[b != 0] = (H * a / b)[b != 0]

        c = np.dot(V, H.T)
        d = np.dot(W, np.dot(H, H.T))

        W[d != 0] = (W * c / d)[d != 0]
    return W, H

def plot_gallery(title, images, n_col=n_col, n_row = n_row ):
    '''
    The Plot Function
    :param title: The Title of picture
    :param images: the numerical images
    :param n_col: as it say
    :param n_row: as it say
    :return:
    '''
    plt.figure(figsize=(2.*n_col, 2.26*n_row))
    plt.suptitle(title, size = 16)
    for i,comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape).T, cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


if __name__ == '__main__':

    data = pd.read_csv('data.csv',sep='\t',header = None).values.T
    print(data.shape)
    t = time.time()
    W, H = train(data, 8, 1000, 1e-4)
    print(H.shape)#8*64
    # print('**********************')
    print(W.shape)#4096*8
    plot_gallery('%s - Train time %.1fs' % ('Non-negative components - NMF', time.time() - t),
                 W.T)
    plt.show()
    #print(H)