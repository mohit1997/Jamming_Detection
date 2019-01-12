import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

framelength = 40
SNR = 0

def get_MIs(df, snr, window):
	temp = df.loc[(df['SNR'] == snr) & (df['Prob'] == p)]
	return temp.iloc[0]['Iy1'], temp.iloc[0]['Iy1y2']

df_nn = pd.read_csv('NN_MIs.csv')
true_df = pd.read_csv('MIs.csv')

prob_list = np.unique(df_nn['p'].values)

markers=['.','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
styles = ['--', '-.', ':']

index = 0
temp = df_nn.loc[(df_nn['SNR'] == SNR) & (df_nn['window'] == framelength)]
lab = "Rate NN"
acheived_mi = temp['MI']
plt.plot(temp['p'], acheived_mi, label=lab, marker=markers[index])
pfa = temp['FAR']
lab = "$P_{fa}$ NN"
plt.plot(temp['p'], pfa, label=lab, marker=markers[index+1])


temp = true_df[true_df['SNR']==SNR]
lab = "$\mathregular{I(x; y_1)}$"
true_iy1 = temp['Iy1']
plt.plot(temp['Prob'], true_iy1, label=lab, linestyle=styles[index])
lab = "$\mathregular{I(x; y_1, y_2|A=0)}$"
true_iy1y2 = temp['Iy1y2']
plt.plot(temp['Prob'], true_iy1y2, label=lab, linestyle=styles[index+1])
index += 1

plt.xlabel('Input Probability Mass Function', fontsize=12)
plt.ylabel('Metric of Interest', fontsize=12)
title = "Analysis at SNR=" + str(SNR) +" Frame Length=" + str(framelength)
plt.title(title)
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.savefig('Skew.png')
plt.show()