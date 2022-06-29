# Copyright (c) 2022 Dai HBG

"""
定义训练模型的函数
2022-04-16
- init
2022-04-28
- 记录vs和os，输出历史最优vs和对应的os
"""

import numpy as np
import torch

import sys
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/QBG')
sys.path.append('C:/Users/18316/Desktop/Daily-Frequency-Quant/QBG')
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

        se = (~np.isnan(y[top[:, i], i])) & (~np.isnan(x_pre[:, 0]))

        corr.append(np.corrcoef(x_pre[se,0], y[top[:, i], i][se])[0, 1])
    return corr


def train_cs(x, y, model, optimizer, loss_func, signal: np.array, target: np.array, univ: np.array,
             epochs=5, batch_size=5, vs_s: int = 100, vs_e: int = 180, os_s: int = 180, os_e: int = 240,
             verbose: int = 1):
    """
    :param x:
    :param y:
    :param optimizer:
    :param loss_func: 损失函数，为IC或者MSE
    :param model:
    :param signal: 信号矩阵
    :param target: 目标矩阵
    :param univ:
    :param epochs:
    :param batch_size:
    :param vs_s: vs开始下标
    :param vs_e: vs结束下标
    :param os_s: os开始下标
    :param os_e: os开始下标
    :param verbose: 多少epoch测试
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
    vs = []
    os = []
    for epoch in range(epochs):
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
                optimizer.zero_grad()

        if (epoch+1) % verbose == 0:
            print('epoch {}'.format(epoch+1))
            print(np.mean(all_loss))

            corr = test(model, signal, target, univ, vs_s, vs_e)
            print('vs cs IC: {:.4f}'.format(np.mean(corr)))
            vs.append(np.mean(corr))
            corr = test(model, signal, target, univ, os_s, os_e)
            print('os cs IC: {:.4f}'.format(np.mean(corr)))
            os.append(np.mean(corr))

    print('best vs: {:.4f}, os: {:.4f}'.format(np.max(vs), os[np.argmax(vs)]))


def train_ts(x, y, model, optimizer, loss_func, signal: np.array, target: np.array, univ: np.array,
             epochs=5, batch_size=5, vs_s: int = 800, vs_e: int = 1400, os_s: int = 1400, os_e: int = 2081,
             verbose: int = 1):
    """
    :param x:
    :param y:
    :param optimizer:
    :param loss_func: 损失函数，为IC或者MSE
    :param model:
    :param signal: 信号矩阵
    :param target: 目标矩阵
    :param univ:
    :param epochs:
    :param batch_size:
    :param vs_s: vs开始下标
    :param vs_e: vs结束下标
    :param os_s: os开始下标
    :param os_e: os开始下标
    :param verbose:
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
    vs = []
    os = []
    for epoch in range(epochs):
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
                optimizer.zero_grad()
        if (epoch + 1) % verbose == 0:
            print('epoch {}'.format(epoch + 1))
            print(np.mean(all_loss))

            corr = test_ts(model, signal[21:], target[21:], univ[21:], vs_s, vs_e)
            print('vs ts IC: {:.4f}'.format(np.mean(corr)))
            vs.append(np.mean(corr))

            corr = test_ts(model, signal[21:], target[21:], univ[21:], os_s, os_e)
            print('os ts IC: {:.4f}'.format(np.mean(corr)))
            os.append(np.mean(corr))

    print('best vs: {:.4f}, os: {:.4f}'.format(np.max(vs), os[np.argmax(vs)]))
