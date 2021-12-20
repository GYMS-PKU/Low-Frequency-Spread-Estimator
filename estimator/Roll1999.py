# Copyright (c) 2021 Dai HBG


"""
该代码实现论文
Roll, R. (1984) A simple implicit measure of the effective bid-ask spread in an efficient market,
Journal of Finance 39, 1127–1139.
中的求协方差的方法来估计spread

开发日志
2021-11-26
-- 实现pure python版本，后续应当实现cython版本，如果有加速效果
"""


import numpy as np


def roll(p: np.array, threshold: float = 0.01) -> np.array:
    """
    :param p: 原始价格序列，为二维array，days * num_stock形状
    :param threshold: 不正常价差估计的修正
    :return: 估计矩阵
    """
    delta_p = p[1:] - p[:-1]
    cov = np.sum(delta_p[1:]*delta_p[:-1], axis=0)
    cov -= np.mean(delta_p[1:], axis=0) * np.mean(delta_p[:-1], axis=0)
    cov[cov > -threshold**2] = -threshold ** 2
    return -2 * np.sqrt(-cov)
