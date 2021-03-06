import numpy as np
import matplotlib.pyplot as plt

window_size = 500

def gen_data(p, SNR, N):
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
	y2 = np.multiply(x, b) + unscaled_z2*scalefactor2

	y = np.hstack([y1, y2])

	return y

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
	p_list = [0.1]
	for snr in [0.3, 0.5, 0.7, 0.9]:
		snr = snr*100
		acc_list = []
		for p in p_list:
			data = gen_data(p=p, SNR=snr, N=1000000)
			X, Y = preprocess(data)
			l = int(len(X)/2.0)
			sm = np.mean(X, axis=1)
			hist, bins = np.histogram(sm.reshape(-1), bins=10)
			plt.hist(sm.reshape(-1)[l:], bins=30)
			plt.show()
			plt.hist(sm.reshape(-1)[:l], bins=30)
			plt.show()






if __name__ == "__main__":
	main()