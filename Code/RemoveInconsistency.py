import os
import numpy as np
import pandas as pd


def fill_data(prev, curr):
    timestamp_delta = curr[0] - prev[0]
    open_delta = curr[1] - prev[1]
    high_delta = curr[2] - prev[2]
    low_delta = curr[3] - prev[3]
    close_delta = curr[4] - prev[4]
    volume_btc_delta = curr[5] - prev[5]
    volume_currency_delta = curr[6] - prev[6]
    wprice_delta = curr[7] - prev[7]

    t_delta = 60
    matrix = []
    prev_temp = prev
    while t_delta != timestamp_delta:
        row_len = timestamp_delta / 60
        matrix.append([prev_temp[0] + 60,
                       float(open_delta) / row_len + prev_temp[1],
                       float(high_delta) / row_len + prev_temp[2],
                       float(low_delta) / row_len + prev_temp[3],
                       float(close_delta) / row_len + prev_temp[4],
                       float(volume_btc_delta) / row_len + prev_temp[5],
                       float(volume_currency_delta) / row_len + prev_temp[6],
                       float(wprice_delta) / row_len + prev_temp[7]])
        prev_temp = matrix[len(matrix) - 1]
        t_delta += 60

    matrix = np.array(matrix)
    matrix[:, 1].round(str(curr[1])[::-1].find('.'))
    matrix[:, 2].round(str(curr[2])[::-1].find('.'))
    matrix[:, 3].round(str(curr[3])[::-1].find('.'))
    matrix[:, 4].round(str(curr[4])[::-1].find('.'))
    matrix[:, 5].round(str(curr[5])[::-1].find('.'))
    matrix[:, 6].round(str(curr[6])[::-1].find('.'))
    matrix[:, 7].round(str(curr[7])[::-1].find('.'))

    return matrix


def load_data(filename):
    current_directory = os.getcwd()
    input_folder = current_directory + '/../Data'
    bitcoin_data = pd.read_csv(input_folder + '/' + filename + '.csv').sort_values(['Timestamp'])
    bitcoin_matrix = bitcoin_data.as_matrix()

    return bitcoin_matrix


def process_data(matrix):
    prev = None
    new_matrix = []
    for row in matrix:
        if prev is not None:
            if row[0] - prev[0] != 60:
                generated_data = fill_data(prev, row)
                for nr in generated_data:
                    new_matrix.append(nr)
        new_matrix.append(row)
        prev = row

    return new_matrix


def check_data(matrix):
    prev = None

    for row in matrix:
        if prev is not None:
            if row[0] - prev[0] != 60:
                print 'Data not consistent..!!'
                return False
        prev = row

    return True


def save_file(data, filename):
    current_directory = os.getcwd()
    input_folder = current_directory + '/../Data'
    pd.DataFrame(data).to_csv(input_folder + '/' + filename + '.csv',
                              header=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',
                                      'Weighted_Price'], index=False)


u_inp = raw_input("\nEnter file name to load\n")

original = load_data(u_inp)
processed = process_data(original)
if check_data(processed):
    save_file(processed, 'Processed')
