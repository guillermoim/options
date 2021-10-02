from partitions_tracker import _id_room
from taxi_domain import TaxiDomain
import numpy as np



env = TaxiDomain(10)
Q_flat = np.loadtxt('results/TAXI_Flat_Q_10x10.txt')

print(len(env.states), Q_flat.shape)

V_flat = np.nanmax(Q_flat, axis=1)


for i, s in enumerate(env.states):
     print(s, V_flat[i])