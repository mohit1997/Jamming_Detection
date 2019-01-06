import csv
import numpy as np

def create_csv(fname, header=None):
    if header is None:
        header = ["SNR", "Prob", "Iy1", "Iy1y2"]

    with open(fname, 'w') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(header)

def gaussian(mean, std):
    return lambda x: np.exp(-(x-mean)**2/2.0/std**2) * 1.0/np.sqrt(2*np.pi*std**2)