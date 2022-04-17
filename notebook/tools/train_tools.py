# Copyright (c) 2022 Dai HBG

"""
定义训练模型的函数
2022-04-16
- init
"""

import numpy as np
import torch

import sys
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG')
sys.path.append('C:/Users/HBG/Desktop/Repositories/Daily-Frequency-Quant/QBG')

from Model.MyDeepModel import *
from Model.Loss import *
from Model.tools.fitting_tools import *
from Model.tools.test_tools import *


def test_ts(model, x: np.array, y: np.array, top, s, e, device: str = 'cuda'):
    corr = []

    for i in range(s, e):
        x_tmp = x[top[:, i], i]
        x_pre = model(torch.Tensor(x_tmp).to(device)).detach().cpu().numpy()
        x_tmp[np.isnan(x_tmp)] = 0
        se = ~np.isnan(y[top[:, i], i])
        pre = x_pre[se, 0]
        corr.append(np.corrcoef(pre, y[top[:, i], i][se])[0, 1])
    return corr


def train_cs(x, y, model, optimizer, loss_func, signal: np.array, target: np.array,
             epochs=5, batch_size=5, vs_s: int = 100, vs_e: int = 180, os_s: int = 180, os_e: int = 240):
    """
    :param x:
    :param y:
    :param optimizer:
    :param loss_func: 损失函数，为IC或者MSE
    :param model:
    :param signal: 信号矩阵
    :param target: 目标矩阵
    :param epochs:
    :param batch_size:
    :param vs_s: vs开始下标
    :param vs_e: vs结束下标
    :param os_s: os开始下标
    :param os_e: os开始下标
    """
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    bs = 0
    loss = 0
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch))

        all_loss = []
        for i in range(len(x)):
            out = model(x[i])
            tmp_loss = loss_func(out, y[i])
            all_loss.append(float(tmp_loss))
            loss += tmp_loss
            bs += 1
            if bs % batch_size == 0:
                loss /= batch_size
                loss.backward()
                optimizer.step()
                bs = 0
                loss = 0

        print(np.mean(all_loss))

        corr = test(model, signal[:, :, :n], target, univ, vs_s, vs_e)
        print('vs cs IC: {:.4f}'.format(np.mean(corr)))
        corr = test(model, signal[:, :, :n], target, univ, os_s, os_e)
        print('os cs IC: {:.4f}'.format(np.mean(corr)))


def train_ts(x, y, model, optimizer, loss_func, signal: np.array, target: np.array,
             epochs=5, batch_size=5, vs_s: int = 800, vs_e: int = 1400, os_s: int = 1400, os_e: int = 2081):
    """
    :param x:
    :param y:
    :param optimizer:
    :param loss_func: 损失函数，为IC或者MSE
    :param model:
    :param signal: 信号矩阵
    :param target: 目标矩阵
    :param epochs:
    :param batch_size:
    :param vs_s: vs开始下标
    :param vs_e: vs结束下标
    :param os_s: os开始下标
    :param os_e: os开始下标
    """

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    bs = 0
    loss = 0
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch))

        all_loss = []
        for i in range(len(x)):
            out = model(x[i])
            tmp_loss = loss_func(out, y[i])
            all_loss.append(float(tmp_loss))
            loss += tmp_loss
            bs += 1
            if bs % batch_size == 0:
                loss /= batch_size
                loss.backward()
                optimizer.step()
                bs = 0
                loss = 0

        print(np.mean(all_loss))

        corr = test_ts(model, signal[20:], rel_sp[20:], univ[20:], vs_s, vs_e)
        print('vs ts IC: {:.4f}'.format(np.mean(corr)))

        corr = test_ts(model, signal[20:], rel_sp[20:], univ[20:], os_s, os_e)
        print('os ts IC: {:.4f}'.format(np.mean(corr)))
