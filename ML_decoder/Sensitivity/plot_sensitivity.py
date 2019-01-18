import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('try5snr40.csv')
# df = df.sort_values(['window'])

window = int(df.loc[df['emd'] == 0.01].loc[0]['window'])
snr = int(df.loc[df['emd'] == 0.01].loc[0]['SNR'])

print(window)
print(df)

emds = df['emd'].as_matrix()
emds = np.log10(emds)
fas = df['FAR']

title = "Sensitivity - SNR=" + str(snr) +"dB - Framelength=" + str(window)

plt.plot(emds, fas)
plt.title(title)
plt.xlabel("Precision in log10 scale", fontsize=12)
plt.ylabel("FAR", fontsize=12)
plt.savefig("sensitivity" + str(snr)+str(window))
plt.grid()
plt.show()

