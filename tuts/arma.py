'''
Source:
https://stats.stackexchange.com/questions/437497/difference-between-ma-and-ar

Some numbers are wrong though
'''
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

b1 = 0.5
b2 = 0.2
observed = np.array([0, 0, -2, -2, -1, 2, 2, 1, 0])
observed = np.random.random(100)

AR_1 = observed[1:]*b1
Err_1 = AR_1[:-1] - observed[2:]
MA_1 = np.insert(Err_1,0, 0, axis=0)*b1
observed

## 
table_array = np.hstack((observed[2:].reshape(-1, 1), AR_1[:-1].reshape(-1, 1),
                         Err_1.reshape(-1, 1), MA_1[:-1].reshape(-1, 1)))
headers = ["Ob", "AR", "Er", "MA"]
table = tabulate(table_array, headers, tablefmt="fancy_grid")
print(table)

for col in range(4):
    plt.plot(table_array[:, col], label=headers[col])

plt.legend()
plt.grid()
plt.show()

