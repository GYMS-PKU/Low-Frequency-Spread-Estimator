# Copyright (c) 2022 Dai HBG

"""
定义得到signal矩阵的方法
2022-04-16
- init
"""


import numpy as np
from tqdm import tqdm
import torch


def get_signal_spread(se):  # 得到以spread为目标的signal
    sig = {}

    # HL
    beta = 'prod{tsmean{powv{minus{high,low},2},2},2}'
    high_1 = 'tsdelay{high,1}'
    low_1 = 'tsdelay{low,1}'
    con_1 = 'condition{gt{low,tsdelay{close,1}},minus{low,tsdelay{close,1}},minus{close,close}}'
    con_2 = 'condition{lt{high,tsdelay{close,1}},minus{high,tsdelay{close,1}},minus{close,close}}'
    con = 'add{' + con_1 + ',' + con_2 + '}'
    high_2 = 'minus{high,' + con + '}'
    low_2 = 'minus{low,' + con + '}'
    high = 'condition{ge{' + high_1 + ',' + high_2 + '},' + high_1 + ',' + high_2 + '}'
    low = 'condition{le{' + low_1 + ',' + low_2 + '},' + low_1 + ',' + low_2 + '}'

    gamma = 'powv{minus{' + high + ',' + low + '},2}'
    alpha = 'div{prod{' + 'powv{' + beta + ',0.5},0.4142},0.1716}'
    fml = 'minus{' + alpha + ',' + 'powv{' + 'div{' + gamma + ',0.1716},0.5}}'

    fml = 'div{' + 'minus{expv{' + fml + '},1},' + 'add{expv{' + fml + '},1}}'
    fml = 'condition{' + 'ge{' + fml + ',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'
    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='relative_spread')
    sig['HL'] = signal

    # Roll
    a = 'tsdelta{close,1}'
    b = 'tsdelay{tsdelta{close,1},1}'
    fml = 'prod{' + a + ',' + b +'}'
    fml = 'condition{' + 'ge{' + fml +',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'
    fml = 'powv{' + fml + ',0.5}'

    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='spread')
    sig['ROLL'] = signal

    # CHL
    a = 'minus{tsdelay{close,1},div{add{tsdelay{high,1},tsdelay{low,1}},2}}'
    b = 'minus{tsdelay{close,1},div{add{high,low},2}}'
    fml = 'prod{' + a + ',' + b +'}'
    fml = 'condition{' + 'ge{' + fml +',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'
    fml = 'powv{' + fml + ',0.5}'

    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='relative_spread')
    sig['CHL'] = signal

    fml = 'tsmean{close,20}'
    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='relative_spread')
    sig['close'] = signal

    signal = np.zeros((sig['HL'].shape[0], sig['HL'].shape[1], 4), dtype=np.float32)
    signal[:, :, 0] = sig['CHL']
    signal[:, :, 1] = sig['HL']
    signal[:, :, 2] = sig['ROLL']
    signal[:, :, 3] = sig['close']
    return signal


def get_signal_rela(se):  # 得到以relative spread为目标的signal
    sig = {}
    # HL
    beta = 'prod{tsmean{powv{minus{logv{high},logv{low}},2},2},2}'

    high_1 = 'logv{tsdelay{high,1}}'
    low_1 = 'logv{tsdelay{low,1}}'
    con_1 = 'condition{gt{logv{low},logv{tsdelay{close,1}}},minus{logv{low},logv{tsdelay{close,1}}},minus{close,close}}'
    con_2 = 'condition{lt{logv{high},logv{tsdelay{close,1}}},minus{logv{high},logv{tsdelay{close,1}}},minus{close,close}}'
    con = 'add{' + con_1 + ',' + con_2 + '}'
    high_2 = 'minus{logv{high},' + con + '}'
    low_2 = 'minus{logv{low},' + con + '}'
    high = 'condition{ge{' + high_1 + ',' + high_2 + '},' + high_1 + ',' + high_2 + '}'
    low = 'condition{le{' + low_1 + ',' + low_2 + '},' + low_1 + ',' + low_2 + '}'

    gamma = 'powv{minus{' + high + ',' + low + '},2}'
    alpha = 'div{prod{' + 'powv{' + beta + ',0.5},0.4142},0.1716}'
    fml = 'minus{' + alpha + ',' + 'powv{' + 'div{' + gamma + ',0.1716},0.5}}'

    fml = 'div{' + 'minus{expv{' + fml + '},1},' + 'add{expv{' + fml + '},1}}'
    fml = 'condition{' + 'ge{' + fml + ',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'

    fml = 'div{' + 'minus{expv{' + fml + '},1},' + 'add{expv{' + fml + '},1}}'
    fml = 'condition{' + 'ge{' + fml + ',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'
    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='relative_spread')
    sig['HL'] = signal

    # Roll
    a = 'tsdelta{logv{close},1}'
    b = 'tsdelay{tsdelta{logv{close},1},1}'
    fml = 'prod{' + a + ',' + b + '}'
    fml = 'condition{' + 'ge{' + fml + ',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'
    fml = 'powv{' + fml + ',0.5}'

    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='spread')
    sig['ROLL'] = signal

    # CHL
    a = 'minus{logv{tsdelay{close,1}},div{add{logv{tsdelay{high,1}},logv{tsdelay{low,1}}},2}}'
    b = 'minus{logv{tsdelay{close,1}},div{add{logv{high},logv{low}},2}}'
    fml = 'prod{' + a + ',' + b + '}'
    fml = 'condition{' + 'ge{' + fml + ',0},' + fml + ',add{minus{close,close},0}}'
    fml = 'tsmean{' + fml + ',20}'
    fml = 'powv{' + fml + ',0.5}'

    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='relative_spread')
    sig['CHL'] = signal

    fml = 'tsmean{close,20}'
    stats, signal = se.test_factor(fml, corr_type='linear', method='cs', spread_type='relative_spread')
    sig['close'] = signal

    signal = np.zeros((sig['HL'].shape[0], sig['HL'].shape[1], 4), dtype=np.float32)
    signal[:, :, 0] = sig['CHL']
    signal[:, :, 1] = sig['HL']
    signal[:, :, 2] = sig['ROLL']
    signal[:, :, 3] = sig['close']
    return signal


def get_target(se, name: str = 'relative_spread'):  # 得到target
    rel_sp = se.data.spread_dic['relative_spread'].copy()
    back = 20
    for i in range(rel_sp.shape[0]):
        if i < back - 1:
            rel_sp[i] = np.nan
        else:
            rel_sp[i] = np.nanmean(se.data.spread_dic[name][i-back+1:i+1], axis=0)
    return rel_sp


def get_full_batch(signal, target, univ):  # 获得full_batch的数据
    xx = []
    yy = []
    for i in range(len(signal)):
        se = univ[i] & (~np.isnan(target[i]))
        for j in range(signal.shape[2]):
            se = se & (~np.isnan(signal[i, :, j]))
        if np.sum(se) == 0:
            continue
        xx.append(signal[i, se, :])
        yy.append(target[i, se])
    return np.vstack(xx), np.hstack(yy)


def get_train_data_cs(signal, target, univ, s: int = 20, e: int = 100, device: str = 'cuda'):
    x_train_cs = []
    y_train_cs = []

    for i in tqdm(range(s,e)):
        sse = univ[i] & (~np.isnan(target[i]))
        for j in range(signal.shape[2]):
            sse = sse & (~np.isnan(signal[i,:,j]))
        if np.sum(sse) == 0:
            continue
        tmp = torch.Tensor(signal[i,sse]).to(device)
        tmp[torch.isnan(tmp)] = 0
        x_train_cs.append(tmp)
        tmp = torch.Tensor(target[i:i+1,sse].T).to(device)
        tmp[torch.isnan(tmp)] = 0
        y_train_cs.append(tmp)
    return x_train_cs, y_train_cs


def get_train_data_ts(signal, target, univ, s: int = 0, e: int = 800, device: str = 'cuda'):
    x_train_ts = []
    y_train_ts = []
    for i in tqdm(range(s, e)):
        sse = univ[:,i] & (~np.isnan(target[:,i]))
        for j in range(signal.shape[2]):
            sse = sse & (~np.isnan(signal[:,i,j]))
        if np.sum(sse) == 0:
            continue
        tmp = torch.Tensor(signal[sse,i]).to(device)
        tmp[torch.isnan(tmp)] = 0
        x_train_ts.append(tmp)
        tmp = torch.Tensor(target[sse,i:i+1]).to(device)
        tmp[torch.isnan(tmp)] = 0
        y_train_ts.append(tmp)
    return x_train_ts, y_train_ts
