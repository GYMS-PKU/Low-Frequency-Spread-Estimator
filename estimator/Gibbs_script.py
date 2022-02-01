# Copyright (c) 2022 Dai HBG

"""
多进程采样
"""


import numpy as np
from multiprocessing import Pool

from Gibbs2004 import *
from time import time

import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator')
sys.path.append('C:/Users/Handsome Bad Guy/Desktop/Repositories/Low-Frequency-Spread-Estimator')

from SpreadEstimator.SpreadEstimator import SpreadEstimator


def test(t: tuple):
    close, num = t
    ti = time()
    c = np.zeros((len(close)-20, close.shape[1]))
    print('{} start'.format(num))
    for i in range(20, len(close)):
        c_s, q_s, sigma_u_s = gibbs(close[i-19:i+1],
                                    sigma_c=1e6, ig_alpha=2,
                                    ig_beta=1e-4, sample_num=100)
        c_s = np.mean(c_s, axis=1)
        c[i-20] = c_s
        print('No {} done {}. time used: {:.4f}s'.format(num, i-19, time()-ti))
    np.save('./cache/{}.npy'.format(num), c)


def main():
    se = SpreadEstimator()
    close = se.data.data_dic['close'].copy()
    for i in range(len(close)):
        for j in range(close.shape[1]):
            if np.isnan(close[i, j]):
                if ~np.isnan(close[i - 1, j]):
                    close[i, j] = close[i - 1, j]
    close[np.isnan(close)] = 0
    top = np.sum(np.isnan(se.data.data_dic['close']), axis=0) <= 15

    t = time()
    args = [(close[i*21: i*21+21+20, top], i) for i in range(10)]
    with Pool(10) as p:
        p.map(test, args)
    print('done. time used: {:.4f}s'.format(time()-t))


if __name__ == '__main__':
    main()
