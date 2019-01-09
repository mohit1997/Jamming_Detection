import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from utils import *
import os.path
import pandas as pd
import csv
import argparse


n_epochs = 10000

def get_threshold(l1, l2):
    m1 = np.mean(l1)
    v1 = np.std(l1)

    m2 = np.mean(l2)
    v2 = np.std(l2)

    thresh = (v2*m1 + v1*m2)/(v1+v2)

    return thresh

def get_min_threshold(l1, l2):
    l1 = np.array(l1)
    l1 = l1.reshape(-1)
    lis = np.sort(l1)
    num = int(len(l1)*0.001) + 1
    print("Threshold Index ", num)
    thresh = lis[-num]

    return thresh

def main(args):
    ### Generate Data
    prob = args.prob
    snr = args.snr
    window = args.window
    fname = args.fname
    h1 = 1.0
    h2 = 1.0
    test_points = 10000
    X, Y1, Y2 = gen_data(p=prob, SNR=snr, N=100000, A=1, h1=h1, h2=h2)

    ### Get MI
    mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
    print(mi_numerical, " during attack")
    mia = mi_numerical

    X, Y1, Y2 = gen_data(p=prob, SNR=snr, N=100000, A=0, h1=h1, h2=h2)

    ### Get MI
    mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
    print(mi_numerical, " during no attack")
    mina = mi_numerical

    # train
    MIAs = []
    MINAs = []
    for epoch in range(n_epochs):
        
        # generate the data
        _, Y1, Y2 = gen_data(p=prob, SNR=snr, N=window, A=1, h1=h1, h2=h2)
        
        # Calculate MI
        mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
        # save MI
        MIAs.append(mi_numerical)

        _, Y1, Y2 = gen_data(p=prob, SNR=snr, N=window, A=0, h1=h1, h2=h2)
        
        # Calculate MI
        mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
        # save the loss
        MINAs.append(mi_numerical)

    A_list = MIAs
    NA_list = MINAs

    thresh = get_min_threshold(A_list, NA_list)
        
    fig, ax = plt.subplots()
    ax.plot(range(len(MIAs)), MIAs, label='kNN estimate - Attack')
    ax.plot(range(len(MINAs)), MINAs, label='kNN estimate - NoAttack')
    ax.plot([0, len(MIAs)], [mia,mia], label='True Mutual Information - Attack')
    ax.plot([0, len(MIAs)], [thresh, thresh], label='Threshold for classification')
    ax.plot([0, len(MINAs)], [mina,mina], label='True Mutual Information - NoAttack')
    ax.set_xlabel('training steps')
    ax.legend(loc='best')
    title = "SNR=" + str(snr) + "window=" + str(window) + "prob=" + str(prob)
    ax.set_title(title)
    path = "figs_md/MINE_shuffled" + title + ".png"
    fig.savefig(path)
    fig.show()

    misdetections = 0
    false_alrams = 0

    for i in range(test_points):
        _, valY1, valY2 = gen_data(p=prob, SNR=snr, N=window, A=0, h1=h1, h2=h2)
        MI = mi_numerical = mutual_info_regression(valY1.reshape(-1, 1), valY2.ravel())[0]
        MINAs.append(MI)
        if MI < thresh:
            false_alrams += 1

        _, valY1, valY2 = gen_data(p=prob, SNR=snr, N=window, A=1, h1=h1, h2=h2)
        MI = mi_numerical = mutual_info_regression(valY1.reshape(-1, 1), valY2.ravel())[0]
        MIAs.append(MI)
        if MI > thresh:
            misdetections += 1

    MR = misdetections*1.0 / test_points
    FAR = false_alrams*1.0 / test_points

    name = "gen_MI/MIA-snr" + str(snr) + "window=" + str(window) + "prob=" + str(prob)
    np.save(name, np.array(MIAs))
    name = "gen_MI/MINA-snr" + str(snr) + "window=" + str(window) + "prob=" + str(prob)
    np.save(name, np.array(MINAs))

    output = "SNR=" + str(snr) + " window=" + str(window) + " prob=" + str(prob) + " MD=" + str(MR) + " FAR" + str(FAR)

    print(output)
    res = [snr, window, prob, MR, FAR]

    if not os.path.isfile(fname):
        create_csv(fname)

    with open(fname, 'a') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-snr', action='store', type=int, default=0,
                        dest='snr',
                        help='choose sequence file')
    parser.add_argument('-p', action='store', type=int, default=0.5,
                        dest='prob',
                        help='choose input probability')
    parser.add_argument('-w', action='store', type=int, default=100,
                        dest='window',
                        help='window size')
    parser.add_argument('-csv', action='store',
                        dest='fname',
                        help='Name of CSV to save results')
    arguments = parser.parse_args()
    print(arguments)
    main(arguments)

