import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('output_demo_pendulum.txt', usecols=(0, 1, 2))


plt.plot(data[:, 0], data[:, 1], label='angle')
plt.plot(data[:, 0], data[:, 2], label='angular velocity')
plt.xlabel('time')
plt.ylabel('value')
plt.title('Pendulum System Time Evolution')
plt.legend()
plt.grid()
plt.show()


plt.plot(data[:, 1], data[:, 2], label='phase plot')
plt.xlabel('angle')
plt.ylabel('angular velocity')
plt.title('Pendulum System Phase Plot')
plt.legend()
plt.grid()
plt.show()
