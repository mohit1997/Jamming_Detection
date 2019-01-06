import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('MIs.csv')
df = df.sort_values(['Prob'])
print(df)
snrs = np.unique(df['SNR'].values)
for i in snrs:
	temp = df.loc[df['SNR'] == i]
	lab = "Iy1 - SNR=" + str(i)
	plt.plot(temp['Prob'], temp['Iy1'], label=lab)
	lab = "Iy1y2 - SNR=" + str(i)
	plt.plot(temp['Prob'], temp['Iy1y2'], label=lab)

plt.xlabel('Input Probability')
plt.ylabel('Iy1 and Iy1y2')
plt.legend()
plt.savefig("MIideal_curve.png")
plt.show()