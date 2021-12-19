# Copyright (c) 2021 Dai HBG

"""
该代码定义的类用于计算一个信号的平均IC等统计值
This code is used for calculating the average IC and other statistical values of a signal

日志
2021-08-30
- 定义：计算平均IC，信号自相关系数，IC_IR，IC为正的频率
2021-09-08
- 新增：统计信号排名最高的1个，5个，10个股票的平均收益，以评估信号的纯多头表现
2021-10-09
- 更新：统计平均收益应该也包含超额收益
2021-10-10
- 更新：stats应该存下top前50的所有超额收益率序列，因此可以后期选择需要的排位的股票
2021-10-15
- 更新：对于周频测试，可以使用字段fix_weekday指定计算周几的平均IC
"""

import numpy as np


class Stats:
    def __init__(self):
        self.ICs = []
        self.mean_IC = 0
        self.auto_corr = 0
        self.IC_IR = 0
        self.positive_IC_rate = 0
        self.top_n_ret = {i - 1: [] for i in range(1, 51)}  # 存储多头超额收益
        self.top_n_raw_ret = {i - 1: [] for i in range(1, 51)}  # 存储多头收益


class AutoTester:
    def __init__(self):
        pass

    @staticmethod
    def test(signal, ret, top=None):
        """
        :param signal: 信号矩阵
        :param ret: 和信号矩阵形状一致的收益率矩阵，意味着同一个时间维度已经做了delay
        :param top: 每个时间截面上进入截面的股票位置
        :return: 返回Stats类的实例
        """
        signal[np.isnan(signal)] = 0
        if top is None:
            top = signal != 0
        ics = []
        auto_corr = []
        assert len(signal) == len(ret)
        assert len(signal) == len(top)
        stats = Stats()
        for i in range(len(signal)):
            ics.append(np.corrcoef(signal[i, top[i]], ret[i, top[i]])[0, 1])
            arg = signal[i, top[i]].argsort()
            for j in range(1, 51):
                stats.top_n_ret[j-1].append(ret[i, top[i]][arg[-j]] - np.mean(ret[i, top[i]]))
                stats.top_n_raw_ret[j-1].append(ret[i, top[i]][arg[-j]])
            auto_corr.append(
                np.corrcoef(signal[i, top[i] & top[i - 1]], signal[i - 1, top[i] & top[i - 1]])[0, 1])

        ics = np.array(ics)
        ics[np.isnan(ics)] = 0
        auto_corr = np.array(auto_corr)
        auto_corr[np.isnan(auto_corr)] = 0

        stats.ICs = ics
        stats.mean_IC = np.mean(ics)
        stats.auto_corr = np.mean(auto_corr)

        if len(ics) > 1:
            stats.IC_IR = np.mean(ics) / np.std(ics)
        stats.positive_IC_rate = np.sum(ics > 0) / len(ics)
        return stats

    @staticmethod
    def cal_bin_ret(signal, ret, top=None, cell=20):
        signal[np.isnan(signal)] = 0
        if top is None:
            top = signal != 0
        z = [[] for i in range(cell)]
        r = [[] for i in range(cell)]

        for i in range(len(signal)):
            tmp = signal[i].copy()
            tmp[top[i] & (tmp != 0)] -= np.mean(tmp[top[i] & (tmp != 0)])
            tmp_ret = ret[i].copy()
            tmp_ret[np.isnan(tmp_ret)] = 0
            tmp_ret[top[i] & (tmp != 0)] -= np.mean(tmp_ret[top[i] & (tmp != 0)])
            tmp[top[i] & (tmp != 0)] /= np.std(tmp[top[i] & (tmp != 0)])
            tmp[np.isnan(tmp)] = 0
            # 放入分组
            signal_ret = []
            for j in range(len(tmp[top[i] & (tmp != 0)])):
                signal_ret.append((tmp[top[i] & (tmp != 0)][j], tmp_ret[top[i] & (tmp != 0)][j]))
            signal_ret = sorted(signal_ret)
            pos = 0

            while pos < cell:
                if pos < cell - 1:
                    for j in range(int(len(signal_ret) / cell * pos), int(len(signal_ret) / cell * (pos + 1))):
                        z[pos].append(signal_ret[j][0])
                        r[pos].append(signal_ret[j][1])
                else:
                    for j in range(int(len(signal_ret) / cell * pos), len(signal_ret)):
                        z[pos].append(signal_ret[j][0])
                        r[pos].append(signal_ret[j][1])
                pos += 1
        return z, r
