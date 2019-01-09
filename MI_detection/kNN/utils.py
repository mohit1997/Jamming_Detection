import numpy as np
import csv

def create_csv(fname, header=None):
    if header is None:
        header = ["SNR", "window", "p", "MD", "FAR"]

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

    y1 = h1*x + unscaled_z1*scalefactor1
    if A==1:
        y2 = h2*np.multiply(x, b) + unscaled_z2*scalefactor2
    else:
        y2 = h2*x + unscaled_z2*scalefactor2

    return x, y1, y2

