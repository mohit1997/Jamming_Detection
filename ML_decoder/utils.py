import numpy as np
from keras import backend as K
import csv

def create_csv(fname, header=None):
    if header is None:
        header = ["SNR", "Window", "Prob", "MD", "FAR"]

    with open(fname, 'w') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(header)

def gen_data(p, SNR, N, A=1, h1=1.0, h2=1.0):
	R = 10**(0.05*SNR)
	x = np.random.choice([-1, 1], size=(N, 1), p=[p, 1-p])

	b = np.random.choice([-1, 1], size=(N, 1), p=[0.5, 0.5])

	unscaled_z1 = np.random.normal(size=(N, 1))
	unscaled_z2 = np.random.normal(size=(N, 1))

	pwrx = np.sqrt(np.sum(x**2))
	pwrz1 = np.sqrt(np.sum(unscaled_z1**2))
	pwrz2 = np.sqrt(np.sum(unscaled_z2**2))
	scalefactor1 = pwrx/pwrz1/R
	scalefactor2 = pwrx/pwrz2/R

	y1 = x + unscaled_z1*scalefactor1
	if A==1:
		y2 = np.multiply(x, b) + unscaled_z2*scalefactor2/h1
	else:
		y2 = x + unscaled_z2*scalefactor2/h1

	y = np.concatenate([y1, y2], axis=1)/h2

	return x, y

def strided_2d(a, L, S):
	# a: 2d Array
	nrows = ((len(a) - L) // S) + 1
	return np.lib.stride_tricks.as_strided(a, shape=(nrows, L, a.shape[1]), strides=(S*8*a.shape[1], 8*a.shape[1], 8), writeable=False)

def strided_1d(a, L, S):
	# a: 2d Array
	nrows = ((len(a) - L) // S) + 1
	return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*8*a.shape[1], 8*a.shape[1]), writeable=False)

def preprocess(inp, out, window):

	inp[inp==-1.0] = 0.0
	inp[inp==1.0] = 1.0

	X = strided_2d(out, window, S=1)
	Y = strided_1d(inp, window, S=1)

	return X, Y

def gen_and_process(p, SNR, N, window):
	inp_symbols, out_symbols = gen_data(p=p, SNR=SNR, N=N, A=1)
	X_attack, Y_attack = preprocess(inp_symbols, out_symbols, window=window)

	inp_symbols, out_symbols = gen_data(p=p, SNR=SNR, N=N, A=0)
	X_noattack, Y_noattack = preprocess(inp_symbols, out_symbols, window=window)

	zeros = np.zeros((len(X_attack), 1))
	ones = np.ones((len(X_noattack), 1))

	X = np.concatenate([X_attack, X_noattack], axis=0)
	Y = np.concatenate([Y_attack, Y_noattack], axis=0)
	A = np.concatenate([zeros, ones], axis=0)

	return X, Y, A

def fpr(y_pred, y_true):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    """
    y_pred = K.round(y_pred)
    y_true = K.round(y_true)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    tp = K.sum(y_true * y_pred)
    fpr = fp/(tn + fp + K.epsilon())*100.0
    # fpr = (fp + fn)/(tn + fp + tp + fn)*100
    return fpr

def fnr(y_pred, y_true):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    """
    y_pred = K.round(y_pred)
    y_true = K.round(y_true)
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    tp = K.sum(y_true * y_pred)
    fnr = fn/(fn + tp + K.epsilon())*100
    return fnr



def main():
	X, Y = gen_and_process(p=0.4, SNR=1.0, N=10000, window=100)


	print(X.shape)
	print(Y)


if __name__ == "__main__":
	main()