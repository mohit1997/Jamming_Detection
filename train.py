import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, Dropout, TimeDistributed
from keras.layers import LSTM, GRU, Flatten, Conv1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import regularizers

window_size = 300

def loss_fn(y_true, y_pred):
    return 1/np.log(2) * keras.losses.binary_crossentropy(y_true, y_pred)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def FC(time_steps):
    model = Sequential()
    init = keras.initializers.lecun_uniform(seed=0)
    model.add(Dense(64, input_shape=(time_steps,), activation='relu', kernel_initializer=init, kernel_regularizer=regularizers.l2(0.1)))
    # model.add(Dropout(rate=0.3))
    model.add(Dense(32, activation='relu', kernel_initializer=init, kernel_regularizer=regularizers.l2(0.1)))
    # model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model

def LSTM(time_steps):
    model = Sequential()
    model.add(TimeDistributed(Dense(4), input_shape=(time_steps,1)))
    # model.add(keras.layers.Conv1D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(time_steps, 1)))
    model.add(Bidirectional(CuDNNLSTM(8, stateful=False, return_sequences=False, kernel_regularizer=regularizers.l2(0.1))))
    # model.add(Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=False, kernel_regularizer=regularizers.l2(0.1))))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def fit_model(X, Y, bs, nb_epoch, model):
    y = Y
    optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0, amsgrad=False)
    model.compile(loss=loss_fn, optimizer=optim, metrics=['acc'])
    # checkpoint = ModelCheckpoint("wts", monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    csv_logger = CSVLogger("log", append=True, separator=';')
    # early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

    # callbacks_list = [checkpoint, csv_logger]#, early_stopping]
    callbacks_list = [csv_logger]
    model.fit(X, y, epochs=nb_epoch, batch_size=bs, verbose=1, validation_split=0.3, shuffle=True, callbacks=callbacks_list)

def gen_data(data):
    true_signal = data[:, 0]
    false_signal = data[:, 1]

    true_X = strided_app(true_signal, L=window_size, S=1)
    false_X = strided_app(false_signal, L=window_size, S=1)

    true_Y = np.ones((len(true_X), 1))
    false_Y = np.zeros((len(false_X), 1))

    X = np.concatenate([true_X, false_X], axis=0)
    Y = np.concatenate([true_Y, false_Y], axis=0)

    return X, Y




def main():
    data = np.load('data.npy')
    X, Y = gen_data(data)

    print(X.shape, Y.shape)

    model = FC(window_size)

    ##LSTM
    X = np.expand_dims(X, -1)
    model = LSTM(window_size)
    fit_model(X, Y, bs=64, nb_epoch=10, model=model)






if __name__ == "__main__":
	main()