import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from utils import *

n_epochs = 1000

def lr(epoch):
    if epoch <300:
        return 1e-3
    if epoch <500:
        return 1e-3
    else:
        return 5e-4

def gaussian_noise_layer(input_layer, std, is_train):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    output = tf.cond(is_train, lambda: input_layer + noise, lambda: input_layer)
    return output

def MINE(x_in, y_in, is_train, H, lr):
    
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

    return neg_loss, opt

def main():
    ### Generate Data
    X, Y1, Y2 = gen_data(p=0.5, SNR=0.0, N=100000, A=1)

    ### Get MI
    mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
    print(mi_numerical, " during attack")
    mia = mi_numerical

    X, Y1, Y2 = gen_data(p=0.5, SNR=0.0, N=100000, A=0)

    ### Get MI
    mi_numerical = mutual_info_regression(Y1.reshape(-1, 1), Y2.ravel())[0]
    print(mi_numerical, " during no attack")
    mina = mi_numerical

    Y1 = Y1.reshape(-1, 1)
    Y2 = Y2.reshape(-1, 1)

    # prepare the placeholders for inputs
    x_in = tf.placeholder(tf.float32, [None, 1], name='x_in')
    y_in = tf.placeholder(tf.float32, [None, 1], name='y_in')
    is_train = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32)

    # make the loss and optimisation graphs
    neg_loss, opt = MINE(x_in, y_in, is_train, 20, learning_rate)

    # start the session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # train
    MIAs = []
    MINAs = []
    for epoch in range(n_epochs):
        
        # generate the data
        # x_sample=gen_x()
        # y_sample=gen_y(x_sample)
        _, Y1, Y2 = gen_data(p=0.5, SNR=0.0, N=20, A=1)
        Y1 = Y1.reshape(-1, 1)
        Y2 = Y2.reshape(-1, 1)
        # perform the training step
        feed_dict = {x_in:Y1, y_in:Y2, is_train: True, learning_rate: lr(epoch)}
        neg_l, = sess.run([neg_loss], feed_dict=feed_dict)
        
        # save the loss
        MIAs.append(-neg_l)

        _, Y1, Y2 = gen_data(p=0.5, SNR=0.0, N=20, A=0)
        Y1 = Y1.reshape(-1, 1)
        Y2 = Y2.reshape(-1, 1)
        # perform the training step
        feed_dict = {x_in:Y1, y_in:Y2, is_train: True, learning_rate: lr(epoch)}
        _, neg_l = sess.run([opt, neg_loss], feed_dict=feed_dict)
        
        # save the loss
        MINAs.append(-neg_l)

    feed_dict = {x_in:Y1, y_in:Y2, is_train: False}
    neg_l, = sess.run([neg_loss], feed_dict=feed_dict)

    print(-neg_l)
        
    fig, ax = plt.subplots()
    ax.plot(range(len(MIAs)), MIAs, label='MINE estimate - Attack')
    ax.plot(range(len(MINAs)), MINAs, label='MINE estimate - NoAttack')
    ax.plot([0, len(MIAs)], [mia,mia], label='True Mutual Information - Attack')
    ax.plot([0, len(MINAs)], [mina,mina], label='True Mutual Information - NoAttack')
    ax.set_xlabel('training steps')
    ax.legend(loc='best')
    fig.savefig('MINE_both.png')
    fig.show()

if __name__ == "__main__":
    main()
