import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Dropout, TimeDistributed, BatchNormalization, GaussianNoise
from keras.layers import LSTM, GRU, Flatten, Conv1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import regularizers
from utils import *

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
    model.add(Dense(128, activation='relu'))
    model.add(GaussianNoise(stddev=0.1))
    # model.add(Dense(64, activation='relu')
    # model.add(BatchNormalization())
    # model.add(Dropout(rate=0.3))
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(rate=0.3))
    # model.add(Dense(16, activation='relu', kernel_initializer=init))
    # model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def fit_model(X, Y, bs, nb_epoch, model):
    y = Y
    optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc'])
    # checkpoint = ModelCheckpoint("wts", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("log", append=True, separator=';')
    # early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

    # callbacks_list = [checkpoint, csv_logger]#, early_stopping]
    callbacks_list = [csv_logger]
    model.fit(X, y, epochs=nb_epoch, batch_size=bs, verbose=1, validation_split=0.3, shuffle=True, callbacks=callbacks_list)

def main():
    X, Y, A = gen_and_process(p=0.5, SNR=1.0, N=100000, window=window)
    Y = Y[:, -1:]
    # X = X[:, :, 0:1]

    print(X.shape, Y.shape)

    model = CONV_Model(time_steps=window)

    ##LSTM
    # X = np.expand_dims(X, -1)
    # model = LSTM(window_size)

    ## For Decoding
    # fit_model(X, Y, bs=512, nb_epoch=10, model=model)

    ## For Attack Detection
    fit_model(X, A, bs=512, nb_epoch=10, model=model)

    np.save('input_symbols', Y)
    Y_cap = model.predict(X, batch_size=1024)
    np.save('predictions', Y_cap)

if __name__ == "__main__":
	main()