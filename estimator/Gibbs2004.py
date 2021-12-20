# Copyright (c) 2021 Dai HBG


"""
该代码实现论文
Hasbrouck, J., 2004. Liquidity in the futures pits: inferring market dynamics from incomplete data.
Journal of Financial and Quantitative Analysis 39, 305–326.
中的Gibbs抽样方法来估计spread c和sigma_u^2

开发日志
2021-11-26
-- 实现pure python版本，预计需要改为cython版本
"""


import numpy as np
from scipy.stats import truncnorm


def gibbs(p, sigma_c=1e6, ig_alpha=2, ig_beta=1e-4, sample_num=10000):
    """
    :param p: 长度为T+1的真实价格序列，注意不是对数价格
    :param sigma_c: 先验分布中spread的参数c的方差
    :param ig_alpha: sigma_u^2先验的InvGamma分布参数alpha
    :param ig_beta: sigma_u^2先验的InvGamma分布参数beta
    :param sample_num: 采样数量
    :return: c, sigma_u^2, q的采用序列
    """
    delta_p = np.log(p[1:]) - np.log(p[:-1])
    q = np.ones(len(p))  # 买卖方向序列初始化为全买单
    # c = 0.01  # 价差初始化为0.01
    sigma_u = np.var(delta_p)  # 价格变化方差初始化为对数收益率方差
    q_s = np.zeros((sample_num, len(p)))  # 存放所有的q序列
    c_s = np.zeros(sample_num)
    sigma_u_s = np.zeros(sample_num)
    for num in range(sample_num):  # 循环生成Gibbs抽样
        delta_q = q[1:] - q[:-1]

        # 采样c
        sigma = 1/sigma_c + np.sum(delta_q**2)/sigma_u
        mean = np.sum(delta_q*delta_p) / sigma
        c = truncnorm.rvs(-mean/np.sqrt(sigma), 10000, loc=mean/np.sqrt(sigma)) * np.sqrt(sigma)
        c_s[num] = c

        # 采样sigma_u
        sigma_u = 1 / np.random.gamma(ig_alpha, 1/(ig_beta + np.sum((delta_p - c*delta_q)**2) / 2))
        sigma_u_s[num] = sigma_u

        # 采样q
        q_t = np.zeros(len(q))
        minus = np.exp(c*(delta_p - c*q[1:])/sigma_u)
        add = np.exp(c*(delta_p - c*q[:-1])/sigma_u)
        pro = minus[1:] / add[:-1]

        proba = np.array([minus[0], 1+minus[0]])
        proba /= np.sum(proba)
        q_t[0] = np.random.choice([-1, 1], p=proba)
        for i in range(1, len(q_t)-1):
            proba = np.array([pro[i-1], 1 + pro[i-1]])
            proba /= np.sum(proba)
            q_t[i] = np.random.choice([-1, 1], p=proba)
        proba = np.array([1 / add[-1], 1 + 1 / add[-1]])
        proba /= np.sum(proba)
        q_t[-1] = np.random.choice([-1, 1], p=proba)
        q_s[num] = q_t
    return c_s, q_s, sigma_u_s
