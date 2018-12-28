import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df = df.sort_values(['window'])
print(df)
snrs = np.unique(df['SNR'].values)
for i in snrs:
	temp = df.loc[df['SNR'] == i]
	lab = "MR - SNR=" + str(i)
	plt.plot(temp['window'], temp['MR'], label=lab)
	lab = "FAR - SNR=" + str(i)
	plt.plot(temp['window'], temp['FAR'], label=lab)

plt.xlabel('Frame Lengths')
plt.ylabel('MR and FAR')
plt.legend()
plt.show()