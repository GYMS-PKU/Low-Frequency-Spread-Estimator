# Copyright (c) 2021 Dai HBG

"""
AutoFormula
该模块定义调用FormulaTree进行解析公式以及测试
当前版本不提供自动特征搜寻的功能

日志
2021-12-19
- 初始化
"""


import numpy as np
import datetime
import sys
sys.path.append('C:/Users/Administrator/Desktop/Repositories/Low-Frequency-Spread-Estimator')
from mytools.AutoTester import AutoTester
from FormulaTree import FormulaTree, Node, FormulaParser
from SignalGenerator import SignalGenerator


class AutoFormula:
    def __init__(self, start_date: str, end_date: str, data: Data):
        """
        :param start_date:
        :param end_date:
        :param data: Data实例
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tree_generator = FormulaTree()
        self.operation = SignalGenerator(data=data)
        self.formula_parser = FormulaParser()

    def cal_formula(self, tree: FormulaTree, data_dic: dict, return_type: str = 'signal') -> np.array:  # 递归计算公式树的值
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
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type))
                if tree.operation_type == '1_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num_1)
                if tree.operation_type == '1_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num_1, tree.num_2)
                if tree.operation_type == '1_num_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   tree.num_1, tree.num_2, tree.num_3)
                if tree.operation_type == '2':  # 此时需要判断有没有数字
                    if tree.num_1 is not None:
                        return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                                                                                        return_type),
                                                                       self.cal_formula(tree.right, data_dic,
                                                                                        return_type))
                    else:
                        if tree.left is not None:
                            return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic,
                                                                                            return_type),
                                                                           tree.num_1)
                        else:
                            return self.operation.operation_dic[tree.name](tree.num_1,
                                                                           self.cal_formula(tree.right, data_dic,
                                                                                            return_type))
                if tree.operation_type == '2_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   tree.num_1)
                if tree.operation_type == '2_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   tree.num_1, tree.num_2)
                if tree.operation_type == '2_num_num_num':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type),
                                                                   tree.num_1, tree.num_2, tree.num_3)
                if tree.operation_type == '3':
                    return self.operation.operation_dic[tree.name](self.cal_formula(tree.left, data_dic, return_type),
                                                                   self.cal_formula(tree.middle, data_dic, return_type),
                                                                   self.cal_formula(tree.right, data_dic, return_type))
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

    def test_formula(self, formula: str, data: Data) -> (Stats, np.array):
        """
        :param formula: 需要测试的因子表达式，如果是字符串形式，需要先解析成树
        :param data: Data类
        :return: 返回统计值以及该因子产生的信号矩阵
        """
        if type(formula) == str:
            formula = self.formula_parser.parse(formula)
        return self.cal_formula(formula, data.data_dic)