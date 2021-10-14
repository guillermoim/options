from partitions_tracker import _id_room
from taxi_domain import TaxiDomain
import numpy as np
from itertools import product

dim= 5

env = TaxiDomain(dim)
Q_flat = np.loadtxt('results/TAXI_Flat_Q_5x5.txt')
Q_H = np.loadtxt('results/TAXI_H_Q_5x5.txt')

options = np.load('results/TAXI_options_Q_5x5.npy')

options_corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]


exit_states = set()
starting_states = set()
E1 = [(a, 'TAXI', b) for a in options_corners for b in options_corners]
E2 = [(a, b, c) for a in options_corners for b in options_corners for c in options_corners if b!=c]

exit_states.update(E1)
exit_states.update(E2)
E_set = list(exit_states)


starting_states = list(starting_states)

print(((4, 4), (0,0), (4, 4)) in starting_states)

states = env.states

V_flat = np.nanmax(Q_flat, axis=1)
V_H = np.nanmax(Q_H, axis=1)
for i, s in enumerate(sorted(E_set,key=lambda x: x[0])):
     print(i, s, V_flat[states.index(s)], V_H[states.index(s)])