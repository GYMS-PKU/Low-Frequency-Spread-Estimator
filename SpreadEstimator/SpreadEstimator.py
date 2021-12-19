# Copyright (c) 2021 Dai HBG


"""
SpreadEstimator
这是一个整合方法测试、模型拟合的利用低频数据进行高频流动性估计的模块

日志：
2021-12-19
- 初始化
"""

import numpy as np
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator')
from dataloader.dataloader import DataLoader
from mytools.AutoTester import AutoTester, Stats
from mytools.AutoFormula.AutoFormula import AutoFormula


class SpreadEstimator:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/DailyData/data',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData'):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path
        dl = DataLoader(data_path=data_path, back_test_data_path=back_test_data_path)
        self.data = dl.load()
        self.tester = AutoTester()
        self.autoformula = AutoFormula(self.data)

    def test_factor(self, formula: str, start_date: str = None, end_date: str = None,
                    method: str = 'cs', corr_type: str = 'linear', verbose: bool = True) -> (Stats, np.array):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param start_date:
        :param end_date:
        :param method: 计算方式
        :param corr_type: linear或者log
        :param verbose: 是否打印结果
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        stats, signal = self.autoformula.test_formula(formula, self.data, start_date=start_date,
                                                      end_date=end_date, method=method, corr_type=corr_type)

        print('mean corr: {:.4f}, positive_corr_ratio: {:.4f}, corr_IR: {:.4f}'.
              format(stats.mean_corr, stats.positive_corr_ratio, stats.corr_IR))
        return stats, signal
