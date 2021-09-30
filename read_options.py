import numpy as np
from rooms_domain import NRoomDomain

dims = (2,2)
room_size = 3
goal_pos = (1,1)
goal_rooms = [(0,0)]

abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states
abs_room

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)

Q = np.load('results/options_Q_softmax.npy')

names = ['T', 'L', 'R', 'B', 'G']

a = 3

O_policies = np.exp(Q) / np.nansum(np.exp(Q), axis=2, keepdims=1) 

print(names[a])

for i, s in enumerate(abs_states):
     print(s, names[np.nanargmax(Q[a, i, :])])
