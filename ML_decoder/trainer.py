import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Dropout, TimeDistributed, BatchNormalization, GaussianNoise
from keras.layers import LSTM, GRU, Flatten, Conv1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import regularizers
from utils import *
from keras import backend as K

import matplotlib.pyplot as plt

window = 100

def loss_fn(y_true, y_pred):
    return 1/np.log(2) * keras.losses.binary_crossentropy(y_true, y_pred)


def CONV_Model(time_steps):
    model = Sequential()
    init = keras.initializers.lecun_uniform(seed=0)
    # , kernel_regularizer=regularizers.l2(0.1)
    # model.add(Conv1D(10, 10, strides=1, activation='relu', input_shape=(time_steps, 2)))
    # model.add(Conv1D(5, 5, strides=1, activation='relu'))
    # model.add(Conv1D(1, 3, strides=1, activation='relu'))
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

def fit_model(X, Y, bs, nb_epoch, model):
    y = Y

    scale = 2
    it = len(X)*1.0/bs * scale
    decay = 1/it

    class_weight = {0: 2., 1: 0.5}


    optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=decay, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc', fpr, fnr])
    # checkpoint = ModelCheckpoint("wts", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("log", append=True, separator=';')
    # early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

    # callbacks_list = [checkpoint, csv_logger]#, early_stopping]
    callbacks_list = [csv_logger]

    hist = model.fit(X, y, epochs=nb_epoch, batch_size=bs, class_weight=class_weight, verbose=1, validation_split=0.33, shuffle=True, callbacks=callbacks_list)
    return hist


def plot_SNR():
    SNRlist = [0.5, 1.0, 2.0, 3.0, 4.0]
    windowlist = [50, 100, 150, 500]
    for snr in SNRlist:
        acclist = []
        for w in windowlist:
            X, Y, A = gen_and_process(p=0.5, SNR=snr, N=100000, window=w)
            Y = Y[:, -1:]
            model = CONV_Model(time_steps=w)
            h = fit_model(X, A, bs=512, nb_epoch=5, model=model)
            val_max = np.max(h.history['val_acc'])
            acclist.append(val_max)
        lab = "SNR = " + str(snr)
        plt.plot(windowlist, acclist, label=lab)
    plt.xlabel("window sizes")
    plt.ylabel("Attack Detection Accuracy")
    plt.legend()
    plt.show()

def main():
    X, Y, A = gen_and_process(p=0.5, SNR=1.0, N=100000, window=window)
    Y = Y[:, -1:]
    # X = X[:, :, 0:1]

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    A = A[indices]

    print(X.shape, Y.shape)

    model = CONV_Model(time_steps=window)

    print(np.sum(A)*1.0 /len(A))



    ##LSTM
    # X = np.expand_dims(X, -1)
    # model = LSTM(window_size)

    ## For Decoding
    # fit_model(X, Y, bs=512, nb_epoch=10, model=model)

    ## For Attack Detection
    fit_model(X, A, bs=512, nb_epoch=10, model=model)

    # np.save('input_symbols', Y)
    # Y_cap = model.predict(X, batch_size=1024)
    # np.save('predictions', Y_cap)



if __name__ == "__main__":
	main()
    # plot_SNR()