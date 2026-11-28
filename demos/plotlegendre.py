import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('output_demo_legendre.txt', skiprows=1)

xs = data[:, 0]

for i in range(6):
    idx = 1 + 3*i
    p = data[:, idx]
    pprime = data[:, idx+1]
    pdprime = data[:, idx+2]
    plt.plot(xs, p, label=f'L{i}')
    plt.plot(xs, pprime, label=f'L{i}\'')
    plt.plot(xs, pdprime, label=f'L{i}\'\'')
    plt.plot(xs, (1-xs**2)*pdprime-2*xs*pprime+i*(i+1)*p, label="zero")
    plt.xlabel('x')
    plt.ylabel('value')
    plt.title(f'{i}\'th Legendre polynomial and its derivative')
    plt.legend()
    plt.grid()
    plt.show()
