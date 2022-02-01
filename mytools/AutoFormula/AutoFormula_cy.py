# Copyright (c) 2022 Dai HBG

"""
AutoFormula的Cython版本
"""
import numpy as np
import sys
import datetime

sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator')
sys.path.append('C:/Users/Handsome Bad Guy/Desktop/Repositories/Low-Frequency-Spread-Estimator')

from dataloader.dataloader import *
from mytools.AutoTester import *
from mytools.AutoFormula.FormulaTree_cy import *
from mytools.AutoFormula.SignalGenerator_cy import *


class AutoFormula_cy:
    def __init__(self, data: Data):
        """
        :param data: Data实例
        """
        self.tree_generator = FormulaTree()
        self.operation = SignalGenerator(data=data)
        self.formula_parser = FormulaParser()
        self.auto_tester = AutoTester()

    def cal_formula(self, tree: FormulaTree, data_dic: dict,
                    return_type: str = 'signal') -> np.array:  # 递归计算公式树的值
        """
        :param tree: 需要计算的公式树
        :param data_dic: 原始数据的字典，可以通过字段读取对应的矩阵
        :param return_type: 返回值形式
        :return: 返回计算好的signal矩阵
        """
        if return_type == 'signal':
            if tree.variable_type == 'data':
                if type(tree.name) == int or type(tree.name) == float:
                    return tree.name  # 直接挂载在节点上，但是应该修改成需要数字的就直接返回数字
                return data_dic[tree.name].copy()  # 当前版本需要返回一个副本
            elif tree.variable_type == 'intra_data':
                if tree.num_1 is not None:
                    return data_dic[tree.name][:, tree.num_1, :].copy()
                else:
                    return data_dic[tree.name].copy()  # 如果没有数字就直接返回原本的数据
            else:
                if tree.operation_type == '1':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input, tree.num_1)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input, tree.num_1)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num_num':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input, tree.num_1, tree.num_2)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input, tree.num_1, tree.num_2)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '1_num_num_num':
                    input = self.cal_formula(tree.left, data_dic, return_type)
                    if len(input.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input, tree.num_1, tree.num_2,
                                                                               tree.num_3)
                    elif len(input.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input, tree.num_1, tree.num_2,
                                                                               tree.num_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2':  # 此时需要判断有没有数字
                    if tree.num_1 is None:
                        input_1 = self.cal_formula(tree.left, data_dic, return_type)
                        input_2 = self.cal_formula(tree.right, data_dic, return_type)
                        if len(input_1.shape) == 2:
                            return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2)
                        elif len(input_1.shape) == 3:
                            return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2)
                    else:
                        # 暂时要求数字只能出现在后面
                        input_1 = self.cal_formula(tree.left, data_dic, return_type)
                        if len(input_1.shape) == 2:
                            return self.operation.operation_dic[tree.name + '_num_2d'](input_1, tree.num_1)
                        elif len(input_1.shape) == 3:
                            return self.operation.operation_dic[tree.name + '_num_3d'](input_1, tree.num_1)
                        else:
                            raise NotImplementedError('input shape is not right!')
                        # if tree.left is not None:
                        #     return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                        #                                                                     return_type),
                        #                                                    tree.num_1)
                        # else:
                        #     return self.operation.operation_dic[tree.name](tree.num_1,
                        #                                                    self.cal_formula(tree.right, data_dic,
                        #                                                                     return_type))
                if tree.operation_type == '2_num':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2_num_num':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1, tree.num_2)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1, tree.num_2)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '2_num_num_num':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, tree.num_1,
                                                                               tree.num_2, tree.num_3)
                    elif len(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, tree.num_1,
                                                                               tree.num_2, tree.num_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
                if tree.operation_type == '3':
                    input_1 = self.cal_formula(tree.left, data_dic, return_type)
                    input_2 = self.cal_formula(tree.middle, data_dic, return_type)
                    input_3 = self.cal_formula(tree.right, data_dic, return_type)
                    if len(input_1.shape) == 2:
                        return self.operation.operation_dic[tree.name + '_2d'](input_1, input_2, input_3)
                    elif en(input_1.shape) == 3:
                        return self.operation.operation_dic[tree.name + '_3d'](input_1, input_2, input_3)
                    else:
                        raise NotImplementedError('input shape is not right!')
        if return_type == 'str':
            if tree.variable_type == 'data':
                return tree.name  # 返回字符串
            elif tree.variable_type == 'intra_data':  # 这里也需要判断是否有数字
                if tree.num_1 is not Nonr:
                    return '{' + tree.name + ',{}'.format(tree.num_1) + '}'
                else:
                    return '{' + tree.name + '}'
            else:
                if tree.operation_type == '1':
                    return tree.name + '{' + (self.cal_formula(tree.left, data_dic, return_type)) + '}'
                if tree.operation_type == '1_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + '}'
                if tree.operation_type == '1_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '1_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + str(
                        tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '2':  # 此时需要判断是否有数字
                    if tree.num_1 is not None:
                        return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                               self.cal_formula(tree.right, data_dic, return_type) + '}'
                    else:
                        if tree.left is not None:
                            return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                                   str(tree.num_1) + '}'
                        else:
                            return tree.name + '{' + str(tree.num_1) + ',' + \
                                   self.cal_formula(tree.right, data_dic, return_type) + '}'
                if tree.operation_type == '2_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + '}'
                if tree.operation_type == '2_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + '}'
                if tree.operation_type == '2_num_num_num':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + ',' + \
                           str(tree.num_1) + ',' + str(tree.num_2) + ',' + str(tree.num_3) + '}'
                if tree.operation_type == '3':
                    return tree.name + '{' + self.cal_formula(tree.left, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.middle, data_dic, return_type) + ',' + \
                           self.cal_formula(tree.right, data_dic, return_type) + '}'

    def test_formula(self, formula: str, data: Data, start_date: str = None,
                     end_date: str = None, method: str = 'cs', corr_type: str = 'linear',
                     spread_type: str = 'spread') -> (Stats, np.array):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param data: Data类
        :param start_date:
        :param end_date:
        :param method: 计算方式
        :param corr_type: linear或者log
        :param spread_type: 价差类型，可选spread或者relative_spread
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        if type(formula) == str:
            formula = self.formula_parser.parse(formula)
        signal = self.cal_formula(formula, data.data_dic)  # 暂时为了方便，无论如何都计算整个回测区间的因子值

        if (start_date is None) or (end_date is None):
            start = 0
            end = len(data.spread) - 1
        else:
            start, end = data.get_real_date(start_date, end_date)
        if spread_type == 'spread':
            return self.auto_tester.test(signal[start:end + 1], data.spread[start:end + 1], method=method,
                                         corr_type=corr_type), signal
        elif spread_type == 'relative_spread':
            return self.auto_tester.test(signal[start:end + 1], data.spread_dic['relative_spread'][start:end + 1],
                                         method=method, corr_type=corr_type), signal
