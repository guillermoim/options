from partitions_tracker import _id_room
from taxi_domain import TaxiDomain
import numpy as np

env = TaxiDomain(5)

Q_flat = np.loadtxt('results/TAXI_Flat_Q_softmax.txt')
Q_h = np.loadtxt('results/TAXI_H_Q_softmax.txt')

V_flat = np.nanmax(Q_flat, axis=1)
V_h = np.nanmax(Q_h, axis=1)


for i, s in enumerate(env.states):
     print(s, V_flat[i], V_h[i])