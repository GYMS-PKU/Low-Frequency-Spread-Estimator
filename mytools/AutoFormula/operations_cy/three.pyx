# Copyright (c) 2022 Dai HBG


"""
该代码是3型运算符源代码

日志
2022-02-01
- init
"""


import numpy as np
from libc.math cimport isnan


def condition_2d(double[:, :] a, double[:, :] b, double[:, :] c):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t i, j
    s = np.zeros((dim_1, dim_2))
    cdef double[:, :] s_view = s
    cdef nan = np.nan

    for i in range(dim_1):
        for j in range(dim_2):
            if isnan(a[i, j]):
                s[i, j] = nan
                continue
            if a[i, j] > 0:
                s[i, j] = b[i, j]
            else:
                s[i, j] = c[i, j]
    return s


def condition_3d(double[:, :, :] a, double[:, :, :] b, double[:, :, :] c):
    cdef Py_ssize_t dim_1 = a.shape[0]
    cdef Py_ssize_t dim_2 = a.shape[1]
    cdef Py_ssize_t dim_3 = a.shape[2]
    cdef Py_ssize_t i, j, k
    s = np.zeros((dim_1, dim_2, dim_3))
    cdef double[:, :, :] s_view = s
    cdef nan = np.nan

    for i in range(dim_1):
        for j in range(dim_2):
            for k in range(dim_3):
                if isnan(a[i, j, k]):
                    s[i, j, k] = nan
                    continue
                if a[i, j, k] > 0:
                    s[i, j, k] = b[i, j, k]
                else:
                    s[i, j, k] = c[i, j, k]
    return s
