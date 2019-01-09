import numpy as np
import csv
import matplotlib.pyplot as plt
from utils import *

def Iy1(snr, p, h1=1, N=10000):
    std = np.sqrt(4*p*(1-p)) / 10**(0.05*snr)

    def Py1_na(p, std, h1):
        return lambda x: p * gaussian(-h1, std)(x) + (1-p) * gaussian(h1, std)(x)

    def Py1_na_cond(std, h1, x):
        if x==1:
            return gaussian(h1, std)
        else:
            return gaussian(-h1, std)

    inp = np.random.choice([-1, 1], size=(N, 1), p=[p, 1-p])
    z1 = std * np.random.normal(size=(N, 1))
    
    y1 = h1*inp + z1

    py1 = Py1_na(p, std, h1)(y1)
    hy1 = -np.mean(np.log2(py1))

    y1_xpos = y1[inp==1]
    y1_xneg = y1[inp==-1]

    py1_xpos = Py1_na_cond(std, h1, x=1)(y1_xpos)
    py1_xneg = Py1_na_cond(std, h1, x=-1)(y1_xneg)

    hy1_xpos = -np.mean(np.log2(py1_xpos))
    hy1_xneg = -np.mean(np.log2(py1_xneg))

    MI = hy1 - p*hy1_xneg - (1-p)*hy1_xpos

    return MI

def Iy1y2(snr, p, h1=1, h2=1, N=10000):
    std1 = np.sqrt(4*p*(1-p)) / 10**(0.05*snr)
    std2 = np.sqrt(4*p*(1-p)) / 10**(0.05*snr)

    std_eff = np.sqrt(h1**2 * std1**2 + h2**2 * std2**2)

    def Py_na(p, std, h1, h2):
        return lambda x: p * gaussian(-(h1**2 + h2**2), std)(x) + (1-p) * gaussian((h1**2 + h2**2), std)(x)

    def Py_na_cond(std, h1, h2, x):
        if x==1:
            return gaussian((h1**2 + h2**2), std)
        else:
            return gaussian(-(h1**2 + h2**2), std)

    inp = np.random.choice([-1, 1], size=(N, 1), p=[p, 1-p])
    z_eff = std_eff * np.random.normal(size=(N, 1))
    
    y = (h1**2 + h2**2)*inp + z_eff

    py = Py_na(p, std_eff, h1, h2)(y)
    hy = -np.mean(np.log2(py))

    y_xpos = y[inp==1]
    y_xneg = y[inp==-1]

    py_xpos = Py_na_cond(std_eff, h1, h2, x=1)(y_xpos)
    py_xneg = Py_na_cond(std_eff, h1, h2, x=-1)(y_xneg)

    hy_xpos = -np.mean(np.log2(py_xpos))
    hy_xneg = -np.mean(np.log2(py_xneg))

    MI = hy - p*hy_xneg - (1-p)*hy_xpos

    return MI

if __name__ == "__main__":
    create_csv('MIs.csv')
    for snr in [0, 5.0, 10.0, 15.0]:
        for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            MI1 = Iy1(snr=snr, p=p, h1=1, N=100000)
            MI2 = Iy1y2(snr=snr, p=p, h1=1, h2=1, N=100000)

            res = [snr, p, MI1, MI2]

            with open('MIs.csv', 'a') as myFile:
                writer = csv.writer(myFile)
                writer.writerow(res)