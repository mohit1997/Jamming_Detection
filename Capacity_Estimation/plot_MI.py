import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('MIs.csv')
df = df.sort_values(['Prob'])
print(df)
snrs = np.unique(df['SNR'].values)
for i in snrs:
	temp = df.loc[df['SNR'] == i]
	lab = "True MI - SNR=" + str(i)
	plt.plot(temp['Prob'], 0.5*temp['Iy1'] + 0.5*temp['Iy1y2'], label=lab)

plt.xlabel('Input Probability')
plt.ylabel('True MI')
plt.legend()
plt.savefig("MIideal_curve.png")
plt.show()