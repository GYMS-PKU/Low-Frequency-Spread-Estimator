# Copyright (c) 2021 Dai HBG


"""
2021-12-04
该代码用于使用高频数据计算所有股票的每天的四个价格以及当天的平均spread，存在
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess as sbp
import pickle
from time import time
from multiprocessing import Pool
import datetime


def main_tmp(arg):
    number = arg[0]  # 进程编号
    start = arg[1]  # 遍历日期开始
    end = arg[2]  # 遍历日期结束
    data_path = '/data/HFData/processed_data/basic_stats_data'
    write_path = '/home/daiyh/mypaper/undergraduate/data'
    # stats_path = '/home/daiyh/liquidity/2020_all/tasks/20211127/data'
    years = [2020]  # 按照时间顺序

    # 获得所有股票
    stock_set_0 = []
    data = pd.read_excel('/home/daiyh/liquidity/2020_all/股票信息.xlsx')
    for i in range(len(data)):
        if int(data['上市日期'][i][:4]) <= 2017:
            stock_set_0.append(data['代码'][i][:6])  # 只保留数字
        elif int(data['上市日期'][i][:4]) == 2018:
            if int(data['上市日期'][i][5:7]) <= 9:
                stock_set_0.append(data['代码'][i][:6])  # 只保留数字
    stock_set_0 = set(stock_set_0)
    # 筛掉2019-2020停盘超过三个月的
    stock_set_1 = []
    data = pd.read_excel('/home/daiyh/liquidity/2020_all/流通市值+涨跌停+停牌.xlsx', sheet_name=2)
    for i in range(len(data)):
        if np.sum(data.iloc[i, 2:].values != '交易') <= 60:
            stock_set_1.append(data['代码'][i][:6])
    stock_set_1 = set(stock_set_1)
    stock_set = stock_set_1 & stock_set_0  # 可用的股票
    stock_lst = list(stock_set)
    stock_lst = sorted(stock_lst, key=lambda x: int(x))
    print('total {} stocks are selected'.format(len(stock_set)))

    zdt = pd.read_excel('/home/daiyh/liquidity/2020_all/流通市值+涨跌停+停牌.xlsx', sheet_name=1)  # 涨跌停

    # 用一个表记录所有股票的每日均价，每日成交量
    lst = os.listdir(data_path)
    stocks = []
    for i in lst:
        if i[-4:] == 'XSHE':
            stocks.append(i)  # 所有的股票

    for year in years:
        '''
        for stock in stocks:
            days += os.listdir('{}/{}/{}'.format(data_path, stock, year))  # 获得该年所有交易日
        days = list(set(days))
        days = sorted(days, key=lambda x: int(x))

        tmp = os.listdir('{}/{}'.format(stats_path, year))
        '''
        print('processor {} start from {} to {}...'.format(number, start, end))

        # 统计总日期数量
        tmp_df = pd.read_csv('/home/daiyh/liquidity/2020_all/B_data/tables/0/ask.csv')
        days = []
        for x in tmp_df['Unnamed: 0']:
            x = str(x)
            if len(x) == 3:
                days.append('0' + x)
            else:
                days.append(x)
        print('there are {} days'.format(len(days)))

        now_time = time()
        for day in days:
            # 当日所有股票的表
            df = pd.DataFrame(np.nan, index=stock_lst, columns=['open', 'low', 'high', 'close', 'bid_ask_spread'])
            for stock in stock_lst:

                date = datetime.datetime(year, int(day[:2]), int(day[2:]))
                if zdt[zdt['代码'] == '{}.SZ'.format(stock)][date].values[0] in [-1, 1]:  # 涨跌停
                    continue
                # 读取数据
                try:
                    snapshot = pd.read_csv('{}/{}.XSHE/{}/{}/snapshot.csv'.format(data_path, stock, year, day))
                    order_based_stats = pd.read_csv('{}/{}.XSHE/{}/{}/order_based_stats.csv'.
                                                    format(data_path, stock, year, day))
                except FileNotFoundError:
                    # print('{}/{}/{}/{}/snapshot.csv not found'.format(data_path, stock, year, day))
                    continue
                df.at[stock, 'open'] = (snapshot['BidPX1'].values[0] + snapshot['OfferPX1'].values[0]) / 2
                df.at[stock, 'close'] = (snapshot['BidPX1'].values[-1] + snapshot['OfferPX1'].values[-1]) / 2
                df.at[stock, 'low'] = np.nanmin(snapshot['BidPX1'].values)
                df.at[stock, 'high'] = np.nanmax(snapshot['OfferPX1'].values)
                df.at[stock, 'bid_ask_spread'] = np.nanmean(snapshot['OfferPX1'].values-snapshot['BidPX1'].values)

            print('{}/{} done.'.format(year, day))
            print('time used: {:.4f}s'.format(time() - now_time))

            df.to_csv('{}/{}.csv'.format(write_path, day))


if __name__ == '__main__':
    '''
    args = [(i, i * 25, i * 25 + 25) for i in range(83)]
    for i in range(84):
        if str(i) not in os.listdir('/home/daiyh/liquidity/2020_all/tasks/20211127/data'):
            os.makedirs('/home/daiyh/liquidity/2020_all/tasks/20211127/data/{}'.format(i))
    args.append((83, 2075, 2081))
    cpu_worker_num = 84
    with Pool(cpu_worker_num) as p:
        p.map(main_tmp, args)
    '''
    main_tmp((0, 0, 2500))
