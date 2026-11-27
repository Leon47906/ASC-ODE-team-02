import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('output_demo_legendre.txt', skiprows=1)

xs = data[:, 0]

for i in range(6):
    plt.plot(xs, data[:, 2*i+1], label=f'L{i}')
    plt.plot(xs, data[:, 2*(i+1)], label=f'L{i}\'')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.title(f'{i}\'th Legendre polynomial and its derivative')
    plt.legend()
    plt.grid()
    plt.show()
