# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:57:20 2024

@author: yutin
"""

import cupy as cp


def fillna_cpwhere_njit(array, values=0):
    if cp.isnan(array.sum()):
        print('found')
        array = cp.where(cp.isnan(array), values, array)
    return array