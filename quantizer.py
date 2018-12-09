import numpy as np

window_size = 500

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
	data = np.load('data.npy')

	X, Y = preprocess(data)
	tmp = X
	tmp[X==-1] = 0
	p = np.sum(X, axis=1)*1.0/window_size
	print(p)
	tmp = p
	tmp[p>0.55] = 1
	tmp[p<0.55] = 0
	print(tmp)
	acc = np.sum(np.abs(Y.reshape(-1)-tmp))/len(Y)
	print(acc)






if __name__ == "__main__":
	main()