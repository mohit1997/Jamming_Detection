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


n_epochs = 1000

def lr(epoch):
    if epoch <300:
        return 1e-3
    if epoch <500:
        return 1e-3
    else:
        return 1e-4

def get_threshold(l1, l2):
    m1 = np.mean(l1)
    v1 = np.std(l1)

    m2 = np.mean(l2)
    v2 = np.std(l2)

    thresh = (v2*m1 + v1*m2)/(v1+v2)

    return thresh

def gaussian_noise_layer(input_layer, std, is_train):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    output = tf.cond(is_train, lambda: input_layer + noise, lambda: input_layer)
    return output

def MINE(H):

    x_in = tf.placeholder(tf.float32, [None, 1], name='x_in')
    y_in = tf.placeholder(tf.float32, [None, 1], name='y_in')
    is_train = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)
    
    # shuffle and concatenate
    y_shuffle = tf.random_shuffle(y_in)
    print(len([x_in]*10))
    x_conc = tf.concat([x_in, x_in], axis=0)
    y_conc = tf.concat([y_in, y_shuffle], axis=0)
    
    # propagate the forward pass
    x_conc = gaussian_noise_layer(x_conc, 0.1, is_train)
    layerx = layers.linear(x_conc, H)
    y_conc = gaussian_noise_layer(y_conc, 0.1, is_train)
    layery = layers.linear(y_conc, H)
    layer2 = tf.nn.relu(layerx + layery)
    layer2 = gaussian_noise_layer(layer2, 0.1, is_train)
    output = layers.linear(layer2, 1)
    
    # split in T_xy and T_x_y predictions
    N_samples = tf.shape(x_in)[0]
    T_xy = output[:N_samples]
    T_x_y = output[N_samples:]
    
    # compute the negative loss (maximise loss == minimise -loss)
    neg_loss = -(tf.reduce_mean(T_xy, axis=0) - tf.log(tf.reduce_mean(tf.exp(T_x_y))))
    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(neg_loss)

    return x_in, y_in, is_train, lr, neg_loss, opt

def main(args):
    ### Generate Data
    prob = args.prob
    snr = args.snr
    window = args.window
    test_points = 1000
    X, Y1, Y2 = gen_data(p=prob, SNR=snr, N=100000, A=1)

    ### Get MI
    mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
    print(mi_numerical, " during attack")
    mia = mi_numerical

    X, Y1, Y2 = gen_data(p=prob, SNR=snr, N=100000, A=0)

    ### Get MI
    mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
    print(mi_numerical, " during no attack")
    mina = mi_numerical

    # prepare the placeholders for inputs
    

    # make the loss and optimisation graphs
    x_in, y_in, is_train, learning_rate, neg_loss, opt = MINE(20)

    # start the session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train
    MIAs = []
    MINAs = []

    _, Y1, Y2 = gen_data(p=prob, SNR=snr, N=window, A=1)
    Y1 = Y1.reshape(-1, 1)
    Y2 = Y2.reshape(-1, 1)

    for epoch in range(n_epochs):

        # perform the training step
        feed_dict = {x_in:Y1, y_in:Y2, is_train: True, learning_rate: lr(epoch)}
        _, neg_l = sess.run([opt, neg_loss], feed_dict=feed_dict)

        # save the loss
        MIAs.append(-neg_l)

    # x_in, y_in, is_train, learning_rate, neg_loss, opt = MINE(20)
    sess.run(tf.global_variables_initializer())

    _, Y1, Y2 = gen_data(p=prob, SNR=snr, N=window, A=0)
    Y1 = Y1.reshape(-1, 1)
    Y2 = Y2.reshape(-1, 1)

    for epoch in range(n_epochs):

        # perform the training step
        feed_dict = {x_in:Y1, y_in:Y2, is_train: True, learning_rate: lr(epoch)}
        _, neg_l, = sess.run([opt, neg_loss], feed_dict=feed_dict)

        # save the loss
        MINAs.append(-neg_l)


    A_list = MIAs[-400:]
    NA_list = MINAs[-400:]

    thresh = get_threshold(A_list, NA_list)

    print(-neg_l)
        
    fig, ax = plt.subplots()
    ax.plot(range(len(MIAs)), MIAs, label='MINE estimate - Attack')
    ax.plot(range(len(MINAs)), MINAs, label='MINE estimate - NoAttack')
    ax.plot([0, len(MIAs)], [mia,mia], label='True Mutual Information - Attack')
    ax.plot([0, len(MIAs)], [thresh, thresh], label='Threshold for classification')
    ax.plot([0, len(MINAs)], [mina,mina], label='True Mutual Information - NoAttack')
    ax.set_xlabel('training steps')
    ax.legend(loc='best')
    title = "SNR=" + str(snr) + "window=" + str(window) + "prob=" + str(prob)
    ax.set_title(title)
    path = "figs_slow/MINE_" + title + ".png"
    fig.savefig(path)
    # fig.show()

    misdetections = 0
    false_alrams = 0

    for i in range(test_points):
        _, valY1, valY2 = gen_data(p=prob, SNR=snr, N=window, A=0)
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            feed_dict = {x_in:valY1, y_in:valY2, is_train: True, learning_rate: lr(i)}
            _, neg_l, = sess.run([opt, neg_loss], feed_dict=feed_dict)
        MI = -neg_l
        if MI < thresh:
            misdetections += 1

        _, valY1, valY2 = gen_data(p=prob, SNR=snr, N=window, A=1)
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            feed_dict = {x_in:valY1, y_in:valY2, is_train: True, learning_rate: lr(i)}
            _, neg_l, = sess.run([opt, neg_loss], feed_dict=feed_dict)
        MI = -neg_l
        if MI > thresh:
            false_alrams += 1

        print(misdetections, false_alrams)

    MR = misdetections*1.0 / test_points
    FAR = false_alrams*1.0 / test_points

    output = "SNR=" + str(snr) + " window=" + str(window) + " prob=" + str(prob) + " MR=" + str(MR) + " FAR" + str(FAR)

    print(output)
    res = [snr, window, prob, MR, FAR]

     
    # with open('results.csv', 'a') as myFile:
    #     writer = csv.writer(myFile)
    #     writer.writerow(res)

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-snr', action='store', type=int, default=0,
                        dest='snr',
                        help='choose sequence file')
    parser.add_argument('-p', action='store', type=int, default=0.5,
                        dest='prob',
                        help='choose input probability')
    parser.add_argument('-w', action='store', type=int, default=50,
                        dest='window',
                        help='window size')
    arguments = parser.parse_args()
    print(arguments)
    main(arguments)

