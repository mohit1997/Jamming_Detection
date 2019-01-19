import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

x = np.linspace(0, 1, 100)
y = 0.3*x

x_1 = np.random.uniform(0, 0.5, 8)
y_1 = np.random.uniform(0, 0.15, 8)

x_2 = np.random.uniform(0, 0.5, 8)
y_2 = np.random.uniform(0, 0.25*x_2, 8)

plt.ylim((-0.1, 1))
plt.xlim((-0.1, 1))
plt.plot(x, y)
plt.plot(x_1, y_1, 'o', color='red')
plt.plot(x_2, y_2, 'x', color='green')
plt.text(0.7, 0.1, 'Slope = $\\frac{\\mu}{1-\\mu}$', fontsize=15)
plt.arrow(0.7,0.21,0.05,-0.03, shape='full', lw=1, length_includes_head=True, head_width=.02, fc='k', ec='k')

plt.ylabel('($p_{md|\\bar{x}}$)', fontsize=12)
plt.xlabel('$1 - p_{fa|\\bar{x}}$', fontsize=12)
plt.title("Approximation Check for N=8 codewords")

plt.grid()
plt.savefig("Approx")
plt.show()