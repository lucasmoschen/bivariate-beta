#usr/bin/env python3
"""
Creates the tables to the paper from json file
"""
import json
import numpy as np

def json2file_bivbeta(filename1, filename2):
    """
    Json file from experiment to latex table format.
    """
    with open(filename1, 'r') as f:
        experiment1 = json.load(f)
    with open(filename2, 'r') as f:
        experiment2 = json.load(f)

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
        elif evaluation == 'mape':
            return 'MAPE ($10^{-1}$)'
        elif evaluation == 'coverage':
            return 'Coverage (\%)'

    table = ''
    for id, method in enumerate(['MM1', 'MM2', 'MM3', 'MM4', 'MML']):
        table += "\multirow{4}{*}{"+method+"}"
        for evaluation in ['bias', 'mse', 'mape', 'coverage']:
            data = tuple(experiment1[evaluation][id] + experiment2[evaluation][id])
            table += '& '+ evaluation_name(evaluation) + ' '
            for i in range(8):
                table += ' & {}'.format(formatting(data[i], evaluation))
            table += ' \\\ \n'

    with open('teste.txt', 'w') as f:
        f.write(table)

def json2file_logit_normal(filename):
    """
    Json file from experiment to latex table format.
    """
    with open(filename, 'r') as f:
        experiments = json.load(f)

    def formatting(data, evaluation):
        if evaluation == 'bias':
            return round(100 * data, 3)
        elif evaluation == 'mse':
            return round(100 * data, 3)
        elif evaluation == 'mape':
            return round(10 * data, 4)

    def evaluation_name(evaluation):
        if evaluation == 'bias':
            return 'Bias ($10^{-2}$)'
        elif evaluation == 'mse':
            return 'MSE ($10^{-2}$)'
        elif evaluation == 'mape':
            return 'MAPE ($10^{-1}$)'

    table = ''
    for id, method in enumerate(['MM1', 'MM2', 'MM3', 'MM4', 'MML']):
        table += "\hline \multirow{3}{*}{"+method+"}"
        for evaluation in ['bias', 'mse', 'mape']:
            data = tuple(experiments[evaluation][id])
            table += '& '+ evaluation_name(evaluation) + ' '
            for i in range(5):
                table += ' & {}'.format(formatting(data[i], evaluation))
            table += ' \\\ \n'

    with open('teste.txt', 'w') as f:
        f.write(table)

if __name__ == '__main__':

    #json2file_logit_normal('Documents/GitHub/bivariate-beta/experiments/exp_logit_0_0_1.0_0.1_0.1_1.0_50_1000_63127371.json')
    json2file_logit_normal('Documents/GitHub/bivariate-beta/experiments/exp_logit_-1_-1_1.0_-0.8_-0.8_1.0_50_1000_63127371.json')
    