import numpy as np
import matplotlib.pyplot as plt

step_set = [-1, 0, 1]
step_size = 100

steps = np.random.choice(a=step_set, size=step_size)
path = steps.cumsum()

y = path[1:]-path[:-1]
x = np.arange(y.shape[0])

plt.plot(path)
plt.show()
plt.bar(x, y)
plt.show()
