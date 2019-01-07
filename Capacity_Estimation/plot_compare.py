import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_MIs(df, snr, p):
	temp = df.loc[(df['SNR'] == snr) & (df['Prob'] == p)]
	return temp.iloc[0]['Iy1'], temp.iloc[0]['Iy1y2']

df_MINE = pd.read_csv('results_mine.csv')
df_MINE = df_MINE.sort_values(['window'])

df_nn = pd.read_csv('results_nn.csv')
df_nn = df_nn.sort_values(['window'])

true_df = pd.read_csv('MIs.csv')

snrs = np.unique(df_MINE['SNR'].values)

for i in snrs:
	I1, I2 = get_MIs(true_df, i, 0.5)

	temp = df_nn.loc[df_nn['SNR'] == i]
	lab = "MI NN - SNR=" + str(i)
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	plt.plot(temp['window'], acheived_mi, label=lab)

	temp = df_MINE.loc[df_MINE['SNR'] == i]
	lab = "MI MINE - SNR=" + str(i)
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	plt.plot(temp['window'], acheived_mi, label=lab)

	lab = "MI True - SNR=" + str(i)
	true_mi = (I1*(1)/2.0 + I2*(1)/2.0) * np.ones(len(temp['window']))
	plt.plot(temp['window'], true_mi, label=lab)

plt.xlabel('Frame Lengths')
plt.ylabel('MI Comparision')
plt.legend()
plt.savefig('Comparision.png')
plt.show()