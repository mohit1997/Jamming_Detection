import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Dropout, TimeDistributed, BatchNormalization, GaussianNoise
from keras.layers import LSTM, GRU, Flatten, Conv1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import regularizers
from utils import *
from keras import backend as K
from keras.engine.topology import Layer

import matplotlib.pyplot as plt
import tensorflow as tf

window = 40

def loss_fn(y_true, y_pred):
    return 1/np.log(2) * keras.losses.binary_crossentropy(y_true, y_pred)

class Shuffle(Layer):

    def __init__(self, **kwargs):
        
        super(Shuffle, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(Shuffle, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, training=None):
        if training in {0, False}:
            return x
        else:
            y1, y2 = tf.unstack(x, num=self.shape[-1], axis=2)
            indices = tf.range(0, self.shape[1])
            indices = tf.random_shuffle(indices)
            y1 = tf.gather(y1, indices, axis=1)
            y2 = tf.gather(y2, indices, axis=1)
            out = tf.stack([y1, y2], axis=2)
            print(out)
            return out

    def compute_output_shape(self, input_shape):
        return input_shape



def CONV_Model(time_steps):
    model = Sequential()
    init = keras.initializers.lecun_uniform(seed=0)
    # , kernel_regularizer=regularizers.l2(0.1)
    # model.add(Conv1D(10, 10, strides=1, activation='relu', input_shape=(time_steps, 2)))
    # model.add(Conv1D(5, 5, strides=1, activation='relu'))
    # model.add(Conv1D(1, 3, strides=1, activation='relu'))
    model.add(Shuffle(input_shape=(time_steps, 2)))
    model.add(Flatten(input_shape=(time_steps, 2)))
    # model.add(Dropout(rate=0.3))
    model.add(GaussianNoise(stddev=0.5))
    model.add(Dense(256, activation='relu'))
    model.add(GaussianNoise(stddev=0.1))
    # model.add(Dense(64, activation='relu')
    # model.add(BatchNormalization())
    # model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(rate=0.3))
    # model.add(Dense(16, activation='relu', kernel_initializer=init))
    # model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def fit_model(X, Y, X_val, Y_val, bs, nb_epoch, model):
    y = Y

    scale = 2
    it = len(X)*1.0/bs * scale
    decay = 1/it

    class_weight = {0: 2.0, 1: 1.}


    optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=decay, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc', fpr, fnr])
    # checkpoint = ModelCheckpoint("wts", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("log_shuffle", append=True, separator=';')
    # early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

    # callbacks_list = [checkpoint, csv_logger]#, early_stopping]
    callbacks_list = [csv_logger]

    hist = model.fit(X, y, epochs=nb_epoch, batch_size=bs, class_weight=class_weight, verbose=1, validation_split=0.33, shuffle=True, callbacks=callbacks_list)


    thresh = get_threshold(model, X, Y, bs)

    p = model.predict(X_val, batch_size=bs)

    p_0 = p[Y_val==0]
    md = np.sum(p_0>thresh)*1.0/(np.sum(Y_val==0))
    
    p_1 = p[Y_val==1]
    fa = np.sum(p_1<thresh)*1.0/(np.sum(Y_val==1))
    print("false_alarms -> ", fa, "mis_det -> ", md)
    return fa, md


def plot_SNR(fname, p):
    create_csv(fname)
    SNRlist = [5.0]
    windowlist = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for snr in SNRlist:
        fprlist = []
        fnrlist = []
        for w in windowlist:
            X, Y, A = gen_and_process(p=p, SNR=snr, N=100000, window=w)
            X_val, Y_val, A_val = gen_and_process(p=p, SNR=snr, N=50000, window=w)
            Y = Y[:, -1:]
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            X = X[indices]
            Y = Y[indices]
            A = A[indices]

            model = CONV_Model(time_steps=w)

            fa, md = fit_model(X, A, X_val, A_val, bs=512, nb_epoch=8, model=model)
            
            fprlist.append(md)
            fnrlist.append(fa)

            res = [snr, w, p, md, fa]

            with open(fname, 'a') as myFile:
                writer = csv.writer(myFile)
                writer.writerow(res)

        lab_md = "MD SNR = " + str(snr)
        lab_fa = "FAR SNR = " + str(snr)
        plt.plot(windowlist, fprlist, label=lab_md)
        plt.plot(windowlist, fnrlist, label=lab_fa)

    plt.xlabel("Window Sizes")
    plt.ylabel("Attack Detection Accuracy")
    plt.legend()
    plt.savefig("acc_plot.png")
    plt.show()

def main():
    X, Y, A = gen_and_process(p=0.5, SNR=1.0, N=100000, window=window)
    X_val, Y_val, A_val = gen_and_process(p=0.5, SNR=1.0, N=50000, window=window)
    Y = Y[:, -1:]
    # X = X[:, :, 0:1]

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    A = A[indices]

    print(X.shape, Y.shape)
    print(X_val.shape, A_val.shape)

    model = CONV_Model(time_steps=window)

    print(np.sum(A)*1.0 /len(A))



    ##LSTM
    # X = np.expand_dims(X, -1)
    # model = LSTM(window_size)

    ## For Decoding
    # fit_model(X, Y, bs=512, nb_epoch=10, model=model)

    ## For Attack Detection
    fit_model(X, A, X_val, A_val, bs=512, nb_epoch=10, model=model)

    # np.save('input_symbols', Y)
    # Y_cap = model.predict(X, batch_size=1024)
    # np.save('predictions', Y_cap)



if __name__ == "__main__":
	# main()
    plot_SNR(fname='results_nn0.4.csv', p=0.4)
    plot_SNR(fname='results_nn0.3.csv', p=0.3)
    plot_SNR(fname='results_nn0.2.csv', p=0.2)
    plot_SNR(fname='results_nn0.1.csv', p=0.1)