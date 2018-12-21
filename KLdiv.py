import numpy as np


def KL(p, q):
	div = -np.sum(p*np.log2(q/p))
	return div

def distance(p, q):
	return KL(p, q) + KL(q, p)

def main():
	p = 0.5
	q = 0.9
	p_dis = np.array([p, 1-p])
	q_dis = np.array([q, 1-q])

	# print(distance(p_dis, q_dis))
	theta = (p+q)/2.0
	p_cap = np.array([theta, 1-theta])
	measure = np.abs(distance(p_dis, p_cap) - distance(q_dis, p_cap))
	print("Average", measure)

	for theta in np.linspace(q, p, 100):
		p_cap = np.array([theta, 1-theta])
		measure = np.abs(distance(p_dis, p_cap) - distance(q_dis, p_cap))
		print(measure, theta)



if __name__ == "__main__":
	main()