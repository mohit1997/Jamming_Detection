import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

	return y, np.std(unscaled_z1*scalefactor1), np.std(unscaled_z2*scalefactor2)

def variance(p):
	var = 4*p*(1-p)
	return var
def divide(p, gvar1, gvar2, u=0.5):
	m1 = 1 - 2*p
	m2 = 0
	var1 = variance(p) + gvar1
	var2 = variance(u) + gvar2
	thresh = (m2*var1 + m1*var2)/(var1 + var2)
	return thresh

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
	nrows = ((a.size - L) // S) + 1
	n = a.strides[0]
	return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

def preprocess(data):
	true_signal = data[:, 0]
	false_signal = data[:, 1]
	true_X = strided_app(true_signal, L=window_size, S=1)
	false_X = strided_app(false_signal, L=window_size, S=1)

	true_Y = np.ones((len(true_X), 1))
	false_Y = np.zeros((len(false_X), 1))

	X = np.concatenate([true_X, false_X], axis=0)
	Y = np.concatenate([true_Y, false_Y], axis=0)

	return X, Y

def likelihood(x, p, dev):
	var = dev**2
	dist1 = norm(1, 1.0)
	dist2 = norm(-1, 1.0)
	
	prob = dev * (p*dist1.pdf(x) + (1-p)*dist2.pdf(x))
	print(prob.shape)
	print(prob)
	
	likelihood = np.mean(-np.log2(prob), axis=1)

	return likelihood

def probability(x, p, dev):
	var = dev**2
	dist1 = norm(1, dev)
	dist2 = norm(-1, dev)
	print("Var", var)
	# prob = p*np.exp(-((x)**2)/2*var)/np.sqrt(2*np.pi)/dev \
	#+ (1-p)*np.exp(-((x-1)**2)/2*var)/np.sqrt(2*np.pi)/dev
	prob = dev * (p*dist1.pdf(x) + (1-p)*dist2.pdf(x))
	return prob

def main():
	p_list = np.linspace(0+1e-8, 0.5, num=20)
	p_list = [0.1]
	for snr in [2]:
		acc_list = []
		for p in p_list:
			data, v1, v2 = gen_data(p=p, SNR=snr, N=10000)
			X, Y = preprocess(data)
			tmp = X
			print(v1, v2)
			l1 = likelihood(X, p, v1)
			l2 = likelihood(X, 0.5, v1)
			print(l1, l2)
			a = (l1 < l2)*1.0
			pred = np.mean((a == Y.reshape(-1))*1.0)
			print(pred)
			acc_list.append(np.mean(pred))
			print(a.shape, Y.shape)
			
			

		lab = "SNR = " + str(snr) 
		plt.plot(p_list, acc_list, label=lab)

	plt.xlabel('Input probability distribution')
	plt.ylabel('Attack detection accuracy')
	plt.title('Input distribution vs Detection')
	plt.legend()
	plt.savefig('Accuracy.png')






if __name__ == "__main__":
	main()