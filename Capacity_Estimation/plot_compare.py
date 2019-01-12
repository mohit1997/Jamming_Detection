import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_MIs(df, snr, p):
	temp = df.loc[(df['SNR'] == snr) & (df['Prob'] == p)]
	return temp.iloc[0]['Iy1'], temp.iloc[0]['Iy1y2']

df_knn3 = pd.read_csv('results_knn3.csv')
df_knn3 = df_knn3.sort_values(['window'])

df_knn4 = pd.read_csv('results_knn4.csv')
df_knn4 = df_knn4.sort_values(['window'])

df_nn = pd.read_csv('results_nn.csv')
df_nn = df_nn.sort_values(['window'])

true_df = pd.read_csv('MIs.csv')

snrs = np.unique(df_nn['SNR'].values)

markers=['.','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
styles = ['--', '-.', ':']

index = 0
for i in [0, 5]:
	I1, I2 = get_MIs(true_df, i, 0.5)

	temp = df_knn3.loc[df_knn3['SNR'] == i]
	lab = "KNN (k=3) - SNR=" + str(i) + "dB"
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	plt.plot(temp['window'], acheived_mi, label=lab, marker=markers[3*index])

	temp = df_knn4.loc[df_knn4['SNR'] == i]
	lab = "KNN (k=4)- SNR=" + str(i) + "dB"
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	plt.plot(temp['window'], acheived_mi, label=lab, marker=markers[3*index+1])

	temp = df_nn.loc[df_nn['SNR'] == i]
	lab = "NN - SNR=" + str(i) + "dB"
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	plt.plot(temp['window'], acheived_mi, label=lab, marker=markers[3*index+2])

	lab = "Genie - SNR=" + str(i) + "dB"
	true_mi = (I1*(1)/2.0 + I2*(1)/2.0) * np.ones(len(temp['window']))
	plt.plot(temp['window'], true_mi, label=lab, linestyle=styles[index])
	index += 1

plt.xlabel('Frame Lengths ($n\'$)', fontsize=12)
plt.ylabel('Achievable Rate', fontsize=12)
plt.legend(loc='right', fontsize=10)
plt.grid()
plt.savefig('ComparisionFinal.png')
plt.show()