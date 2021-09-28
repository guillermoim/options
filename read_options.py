import numpy as np
from rooms_domain import NRoomDomain

dims = (2,2)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(1,1)]

abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states
abs_room

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)

Q = np.load('results/options_Q_softmax.npy')

names = ['T', 'L', 'R', 'B', 'G']

a = 0

O_policies = np.exp(Q) / np.exp(Q).sum(axis=2, keepdims=1) 

print(names[a])

for i, s in enumerate(abs_room.states):
    print(s, np.nanmax(Q[a, i, :]), names[np.nanargmax(Q[a, i, :])], env._compute_next_state(s, np.nanargmax(Q[a, i, :])))
    if i > 23:
            break
