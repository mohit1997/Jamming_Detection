import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

def get_MIs(df, snr, p):
	temp = df.loc[(df['SNR'] == snr) & (df['Prob'] == p)]
	return temp.iloc[0]['Iy1'], temp.iloc[0]['Iy1y2']

df_MINE = pd.read_csv('results_mine.csv')
df_MINE = df_MINE.sort_values(['window'])

df_nn = pd.read_csv('results_nn0.4.csv')
df_nn = df_nn.sort_values(['window'])
print(df_nn)

true_df = pd.read_csv('MIs.csv')

snrs = np.unique(df_nn['SNR'].values)

MINE_mi_list = []
NN_mi_list = []

for i in snrs:
	I1, I2 = get_MIs(true_df, i, 0.4)

	temp = df_nn.loc[df_nn['SNR'] == i]
	lab = "MI NN - SNR=" + str(i)
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	NN_mi_list.append(acheived_mi)
	plt.plot(temp['window'], acheived_mi, label=lab)

	temp = df_MINE.loc[df_MINE['SNR'] == i]
	lab = "MI MINE - SNR=" + str(i)
	pfa = temp['FAR']
	acheived_mi = I1*(1+pfa)/2.0 + I2*(1-pfa)/2.0
	MINE_mi_list.append(acheived_mi)
	plt.plot(temp['window'], acheived_mi, label=lab)

	lab = "MI True - SNR=" + str(i)
	true_mi = (I1*(1)/2.0 + I2*(1)/2.0) * np.ones(len(temp['window']))
	plt.plot(temp['window'], true_mi, label=lab)

nn = pd.concat(NN_mi_list)
mine = pd.concat(MINE_mi_list)

print(df_nn)
df_nn['MI'] = nn
df_MINE['MI'] = mine
df_nn = df_nn.round(5)
df_MINE = df_MINE.round(5)

# if os.path.isfile("MINE_MIs.csv"):
# 	frame = pd.read_csv("MINE_MIs.csv")
# 	frame = frame.round(5)
# 	s1 = pd.merge(frame, df_MINE, how='outer', on=['SNR', 'window', 'p', 'MD', 'FAR', 'MI'], sort=True)
# 	s1.to_csv("MINE_MIs.csv", index=False)

# else:
# 	df_MINE.to_csv("MINE_MIs.csv", index=False)

if os.path.isfile("NN_MIs.csv"):
	frame = pd.read_csv("NN_MIs.csv")
	frame = frame.round(5)
	s1 = pd.merge(frame, df_nn, how='outer', on=['SNR', 'window', 'p', 'MD', 'FAR', 'MI'], sort=True)
	s1.to_csv("NN_MIs.csv", index=False)
else:
	df_nn.to_csv("NN_MIs.csv", index=False)


# plt.xlabel('Frame Lengths')
# plt.ylabel('MI Comparision')
# plt.legend()
# plt.savefig('Comparision.png')
# plt.show()