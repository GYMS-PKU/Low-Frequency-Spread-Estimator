# Copyright (c) 2021 Dai HBG


"""
该代码实现论文
Hasbrouck, J., 2004. Liquidity in the futures pits: inferring market dynamics from incomplete data.
Journal of Financial and Quantitative Analysis 39, 305–326.
中的Gibbs抽样方法来估计spread c和sigma_u^2
"""


import numpy as np


def gibbs(sigma_c=1e6, ig_alpha=1e-12, ig_beta=1e-12, sample_num=10000):
    """
    :param sigma_c: 先验分布中spread的参数c的方差
    :param ig_alpha: sigma_u^2先验的InvGamma分布参数alpha
    :param ig_beta: sigma_u^2先验的InvGamma分布参数beta
    :param sample_num:
    :return: c, sigma_u^2的均值
    """
    pass
