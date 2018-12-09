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
	p_list = np.linspace(0+1e-8, 0.5, num=20)
	for snr in [1, 2, 4, 8, 16]:
		acc_list = []
		for p in p_list:
			data = gen_data(p=p, SNR=snr, N=100000)

			X, Y = preprocess(data)
			tmp = X
			tmp[X==-1] = 0
			prob = np.sum(X, axis=1)*1.0/window_size
			print(prob)
			tmp = prob
			thresh = (0.5 + (1-p))/2.0
			tmp[prob>thresh] = 1
			tmp[prob<thresh] = 0
			print(tmp)
			err = np.sum(np.abs(Y.reshape(-1)-tmp))/len(Y)
			print(err)
			acc_list.append(1 - err)

		lab = "SNR = " + str(snr) 
		plt.plot(p_list, acc_list, label=lab)

	plt.xlabel('Input probability distribution')
	plt.ylabel('Attack detection accuracy')
	plt.title('Input distribution vs Detection')
	plt.legend()
	plt.savefig('Accuracy.png')






if __name__ == "__main__":
	main()