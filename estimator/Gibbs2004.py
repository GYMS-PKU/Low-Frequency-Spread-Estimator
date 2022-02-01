# Copyright (c) 2021-2022 Dai HBG


"""
该代码实现论文
Hasbrouck, J., 2004. Liquidity in the futures pits: inferring market dynamics from incomplete data.
Journal of Financial and Quantitative Analysis 39, 305–326.
中的Gibbs抽样方法来估计spread c和sigma_u^2

日志
2021-11-26
- 实现pure python版本，预计需要改为cython版本
2022-01-31
- 需要并行进行采样，个数是截面的股票数
"""


import numpy as np
from scipy.stats import truncnorm


def gibbs(p: np.array, sigma_c: float = 1e6, ig_alpha: float = 2, ig_beta: float = 1e-4,
          sample_num: int = 1000):
    """
    :param p: 为T+1 * stock_num的真实价格序列，注意不是对数价格
    :param sigma_c: 先验分布中spread的参数c的方差
    :param ig_alpha: sigma_u^2先验的InvGamma分布参数alpha
    :param ig_beta: sigma_u^2先验的InvGamma分布参数beta
    :param sample_num: 采样数量
    :return: c, sigma_u^2, q的采样序列
    """
    delta_p = np.log(p[1:]) - np.log(p[:-1])
    q = np.ones(p.shape)  # 买卖方向序列初始化为全买单
    # c = 0.01  # 价差初始化为0.01
    sigma_u = np.var(delta_p, axis=0)  # 价格变化方差初始化为对数收益率方差
    q_s = np.zeros((sample_num, len(p), p.shape[1]))  # 存放所有的q序列
    c_s = np.zeros((p.shape[1], sample_num))
    sigma_u_s = np.zeros((p.shape[1], sample_num))
    for num in range(sample_num):  # 循环生成Gibbs抽样
        delta_q = q[1:] - q[:-1]

        # 采样c
        sigma = 1/sigma_c + np.sum(delta_q**2, axis=0)/sigma_u  # 长度为stock_num
        mean = np.sum(delta_q*delta_p, axis=0) / sigma  # 长度为stock_num
        c = truncnorm.rvs(-mean/np.sqrt(sigma), 10000, loc=mean/np.sqrt(sigma)) * np.sqrt(sigma)  # 长度为stock_num
        c_s[:, num] = c

        # 采样sigma_u
        sigma_u = 1 / np.random.gamma(ig_alpha, 1/(ig_beta + np.sum((delta_p - c*delta_q)**2) / 2))
        sigma_u_s[:, num] = sigma_u

        # 采样q
        q_t = np.zeros(q.shape)  # T+1 * stock_num
        minus = np.exp(c*(delta_p - c*q[1:])/sigma_u)  # T * stock_num
        add = np.exp(c*(delta_p + c*q[:-1]))  # T * stock_num
        pro = minus[1:] / add[:-1]  # T-1 * stock_num

        # 概率分界
        proba = np.zeros(q.shape)
        proba[0] = minus[0] / (1 + minus[0])
        proba[1: len(q_t)-1] = pro / (1 + pro)
        add = np.exp(c*(delta_p + c*q[:-1]) / sigma_u)
        proba[-1] = (1 + add[-1]) / add[-1]

        se = np.random.uniform(0, 1, q.shape)
        q_t[se <= proba] = -1
        q_t[se > proba] = 1

        q_s[num] = q_t
    return c_s, q_s, sigma_u_s
