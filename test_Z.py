import numpy as np


Q = np.loadtxt('Q_stochastic_3x3.txt')
Z = np.loadtxt('optimal_z.txt')

for i in range(Z.shape[0]):
    print(i, np.nanmax(Q[i, :]), np.log(Z[i]))