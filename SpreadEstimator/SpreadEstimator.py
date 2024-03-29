# Copyright (c) 2021-2022 Dai HBG


"""
SpreadEstimator
这是一个整合方法测试、模型拟合的利用低频数据进行高频流动性估计的模块

日志：
2023-08-16
- dataloader for SP500
2021-12-19
- 初始化
2022-01-30
- 将所有的操作符替换成cython版本
"""

import numpy as np
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator')
sys.path.append('C:/Users/Handsome Bad Guy/Desktop/Repositories/Low-Frequency-Spread-Estimator')
from dataloader.dataloader import DataLoader, DataLoader_SP500
from mytools.AutoTester import AutoTester, Stats
# from mytools.AutoFormula.AutoFormula_cy import *
from mytools.AutoFormula.AutoFormula import *


class SpreadEstimator:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/DailyData/data',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData', market='CSI'):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path
        if market == 'CSI':
            dl = DataLoader(data_path=data_path, back_test_data_path=back_test_data_path)
        elif market == 'SP500':
            dl = DataLoader_SP500(data_path=data_path, back_test_data_path=back_test_data_path)
        else:
            dl = DataLoader(data_path=data_path, back_test_data_path=back_test_data_path)
        self.data = dl.load()
        self.tester = AutoTester()
        # self.autoformula = AutoFormula_cy(self.data)
        self.autoformula = AutoFormula(self.data)

    def test_factor(self, formula: str, start_date: str = None, end_date: str = None,
                    method: str = 'cs', corr_type: str = 'linear',
                    spread_type: str = 'spread', verbose: bool = True,
                    back: int = 20) -> (Stats, np.array):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param start_date:
        :param end_date:
        :param method: 计算方式
        :param corr_type: linear或者log
        :param spread_type: 价差类型，可选spread或者relative_spread
        :param verbose: 是否打印结果
        :param back: 回溯天数
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        stats, signal = self.autoformula.test_formula(formula, self.data, start_date=start_date,
                                                      end_date=end_date, method=method, corr_type=corr_type,
                                                      spread_type=spread_type, back=back)
        if verbose:
            print('mean corr: {:.4f}, positive_corr_ratio: {:.4f}, corr_IR: {:.4f}'.
                  format(stats.mean_corr, stats.positive_corr_ratio, stats.corr_IR))
        return stats, signal
