#usr/bin/env python3
"""
Creates the tables to the paper from json file
"""
import json
import numpy as np

def json2file(filename):
    """
    Json file from experiment to latex table format.
    """
    with open(filename, 'r') as f:
        experiments = json.load(f)

    def formatting(data, evaluation):
        if evaluation[0] == 'c':
            value = round(100 * data, 2)
        elif evaluation == 'bias':
            value = round(100 * data, 2)
        elif evaluation == 'mse':
            value = round(100 * data, 2)
        else:
            value = round(10 * data, 2)
        return value

    def evaluation_name(evaluation):
        if evaluation == 'bias':
            return 'Bias ($10^{-2}$)'
        elif evaluation == 'mse':
            return 'MSE ($10^{-2}$)'
        elif evaluation == 'mae':
            return 'MAE ($10^{-1}$)'
        elif evaluation == 'coverage':
            return 'Coverage (\%)'

    table = ''
    for id, method in enumerate(['MM1', 'MM2', 'MM3', 'MM4', 'MML']):
        table += "\multirow{4}{*}{"+method+"}"
        for evaluation in ['bias', 'mse', 'mae', 'coverage']:
            data = tuple(experiments[evaluation][id] + [0,0,0,0])
            table += '& '+ evaluation_name(evaluation) + ' '
            for i in range(8):
                table += ' & {}'.format(formatting(data[i], evaluation))
            table += ' \\\ \n'

    with open('teste.txt', 'w') as f:
        f.write(table)

if __name__ == '__main__':

    json2file('Documents/GitHub/bivariate-beta/experiments/exp_1_1_1_1_50_1000_500_8392.json')