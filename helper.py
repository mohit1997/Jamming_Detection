import numpy as np

def entropy(p):
	h = - np.sum(np.log2(p)*p)
	return h

def main():
	p = 0.99
	p_list = np.array([p, 1-p])
	print(entropy(p_list))

if __name__ == "__main__":
	main()