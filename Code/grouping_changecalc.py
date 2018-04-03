#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 02:47:16 2018

@author: jithin
"""

import os
import math
import random
import constants as cns
import numpy as np
import pandas as pd


def load_data(filename):
    current_directory = os.getcwd()
    input_folder = current_directory + '/Data'
    bitcoin_data = pd.read_csv(input_folder + '/' + filename + '.csv').sort_values(['Timestamp'])
    bitcoin_matrix = bitcoin_data.as_matrix()
    return bitcoin_matrix


def group_data(matrix, period):
    i, j, k = period - 1, 0, 0
    new_matrix = np.zeros((int(matrix.shape[0] / period), matrix.shape[1] + 1))
    while i < len(matrix):
        new_matrix[k][0] = matrix[i][0]
        new_matrix[k][1] = matrix[j][1]
        new_matrix[k][2] = matrix[j:i, 2].max()
        new_matrix[k][3] = matrix[j:i, 3].min()
        new_matrix[k][4] = matrix[i][4]
        new_matrix[k][5] = matrix[j:i, 5].sum()
        new_matrix[k][6] = matrix[j:i, 6].sum()
        new_matrix[k][7] = matrix[i][7]
        new_matrix[k][8] = ((matrix[i][4] - matrix[j][1]) / matrix[j][4]) * 100
        j = i
        k += 1
        i += period
    return new_matrix


def total_change(matrix):
    return (matrix[matrix.shape[0] - 1][4] - matrix[0][1]) / matrix[0][1] * 100


def save_file(data, filename):
    current_directory = os.getcwd()
    print(current_directory)
    input_folder = current_directory + '/Data'
    pd.DataFrame(data).to_csv(input_folder + '/' + filename + '.csv',
                              header=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',
                                      'Weighted_Price', 'Change'], index=False)


def randomSample(seed, data, train_periods, test_periods):
    if not seed == -1:
        random.seed(seed)
    beginIndex = random.randint(0, data.shape[0] - (train_periods + test_periods + 1))
    endIndex = beginIndex + train_periods
    return data[beginIndex:endIndex], data[endIndex:endIndex + test_periods]

# u_inp="Processed"
# original = load_data(u_inp)
# grp_data = group_data(original)
# print(grp_data[0])
# save_file(grp_data, 'Grouped')
