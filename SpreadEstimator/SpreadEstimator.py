# Copyright (c) 2021 Dai HBG


"""
SpreadEstimator
这是一个整合方法测试、模型拟合的利用低频数据进行高频流动性估计的模块

日志：
2021-12-19
- 初始化
"""


import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator')
from dataloader.dataloader import DataLoader
from mytools.AutoTester import AutoTester


class SpreadEstimator:
    def __init__(self, data_path: str = 'D:/Documents/学习资料/DailyData/data',
                 back_test_data_path: str = 'D:/Documents/AutoFactoryData/BackTestData'):
        """
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path
        self.data = DataLoader(data_path=data_path, back_test_data_path=back_test_data_path).load()
        self.tester = AutoTester()

