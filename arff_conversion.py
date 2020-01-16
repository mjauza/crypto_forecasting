# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:04:07 2020

@author: david
"""

import re

def trans_arff2csv(file_in, file_out):
    """trans *.arff file to *.csv file."""
    columns = []
    data = []
    with open(file_in, 'r') as f:
        data_flag = 0
        for line in f:
            if line[:2] == '@A':
                # find indices
                indices = [i for i, x in enumerate(line) if x == ' ']
                columns.append(re.sub(r'^[\'\"]|[\'\"]$|\\+', '', line[indices[0] + 1:indices[-1]]))
            elif line[:2] == '@D':
                data_flag = 1
            elif data_flag == 1:
                data.append(line)

    content = ','.join(columns) + '\n' + ''.join(data)

    # save to file
    with open(file_out, 'w') as f:
        f.write(content)


if __name__ == '__main__':
        # setting arff file path
    file_attr_in = 'C:/Users/david/OneDrive/Dokumenti/crypto_data/cryptoTimeSeries1.test.pred.arff'
    # setting output csv file path
    file_csv_out = "C:/Users/david/OneDrive/Dokumenti/crypto_data/cryptoTimeSeries1_test_predictions.csv"
    # trans
    trans_arff2csv(file_attr_in, file_csv_out)