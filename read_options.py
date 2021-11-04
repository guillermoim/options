import numpy as np
from rooms_domain import NRoomDomain

dims = (1,1)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,0)]

abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states
abs_room

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)

Q = np.load('results/rooms_options_Q_3x3.npy')

names = ['T', 'L', 'R', 'B', 'G', 'NoOp']

o = 0

print(Q.shape)

for i, s in enumerate(abs_states):
     print(s, names[np.nanargmax(Q[o, i, :])], np.nanmax(Q[o, i, :]))
