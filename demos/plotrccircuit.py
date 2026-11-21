import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('output_test_ode_rc.txt', usecols=(0, 1))


plt.plot(data[:, 0], data[:, 1], label='voltage over capacitor')
plt.xlabel('time')
plt.ylabel('U_C')
plt.title('RC-Circuit System Time Evolution')
plt.legend()
plt.grid()
plt.show()
