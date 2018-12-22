import numpy as np

np.random.seed(0)

p = 0.4 # Input Probability Distribution Followed
SNR = 2 # SNR of the both the channels
N = 100000 # Number of samples transmitted from each channel
error = np.random.normal(size=(N, 1))

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
y2 = np.multiply(x, b)*(1) + unscaled_z2*scalefactor2

y = np.hstack([y1, y2])

np.save('data/data', y)