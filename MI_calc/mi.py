import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

window_size = 500

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def gen_data(p, SNR, N, A=1):
	x = np.random.choice([-1, 1], size=(N, 1), p=[p, 1-p])

	b = np.random.choice([-1, 1], size=(N, 1), p=[0.5, 0.5])

	unscaled_z1 = np.random.normal(size=(N, 1))
	unscaled_z2 = np.random.normal(size=(N, 1))

	pwrx = np.sqrt(np.sum(x**2))
	pwrz1 = np.sqrt(np.sum(unscaled_z1**2))
	pwrz2 = np.sqrt(np.sum(unscaled_z2**2))
	scalefactor1 = pwrx/pwrz1/SNR
	scalefactor2 = pwrx/pwrz2/SNR

	y1 = x + unscaled_z1*scalefactor1
	if A==1:
		y2 = np.multiply(x, b) + unscaled_z2*scalefactor2
	else:
		y2 = x + unscaled_z2*scalefactor2

	return x, y1, y2

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
	nrows = ((a.size - L) // S) + 1
	n = a.strides[0]
	return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def preprocess(data):
	data[data<0] = -1.0
	data[data>0] = 1.0

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
	bin_size = 20
	p_list = [0.5]
	for snr in [0.3, 0.5, 0.7, 0.9]:
		acc_list = []
		for p in p_list:
			x, y1, y2 = gen_data(p=p, SNR=snr, N=1000000, A=0)
			x = x.reshape(-1)
			y1 = y1.reshape(-1)
			y2 = y2.reshape(-1)
			Ixy1 = calc_MI(x, y1, bin_size)
			Ixy2 = calc_MI(x, y2, bin_size)
			Iy1y2 = calc_MI(y1, y2, bin_size)

			MI = (Ixy1 + Ixy2 - Iy1y2)/np.log(2)
			print(MI, p, snr)

			






if __name__ == "__main__":
	main()