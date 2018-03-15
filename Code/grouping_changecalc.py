#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 02:47:16 2018

@author: jithin
"""

import os
import numpy as np
import pandas as pd

def load_data(filename):
    current_directory = os.getcwd()
    input_folder = current_directory + '/Data'
    bitcoin_data = pd.read_csv(input_folder + '/' + filename + '.csv').sort_values(['Timestamp'])
    bitcoin_matrix = bitcoin_data.as_matrix()
    return bitcoin_matrix

def group_data(matrix):
    i,j=10,0
    new_matrix=[]
    print(new_matrix)
    for i in range(len(matrix)):
        change=((matrix[i][4]-matrix[j][4])/matrix[j][4])*100
        new_matrix.append((np.append(matrix[i],change)))
        j=i
        i+=10
    return new_matrix
  
def save_file(data, filename):
    current_directory = os.getcwd()
    print(current_directory)
    input_folder = current_directory + '/Data'
    pd.DataFrame(data).to_csv(input_folder + '/' + filename + '.csv',
                              header=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',
                                      'Weighted_Price','Change'], index=False)
        
        
u_inp="Processed"
original = load_data(u_inp)
grp_data = group_data(original)
print(grp_data[0])
save_file(grp_data, 'Grouped')
