import pandas as pd
import numpy as np

df = pd.read_csv("getmumy.csv")

MDs = df['MD_x'].values
FARs = df['FAR_x'].values
mus = MDs/(1 - FARs + MDs)
print(np.mean(mus), np.std(mus))