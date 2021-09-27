from partitions_tracker import _id_room
from rooms_domain import NRoomDomain
import numpy as np


dims = (2,2)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(1,1)]

infty = 1e6

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states

Qo = np.loadtxt('Hierarchical_Q.txt')
Q = np.loadtxt('Flat_Q.txt')

names = ['T', 'L', 'R', 'B', 'G']

for i, s in enumerate(env.states):
    print(s, _id_room(s), f'Best option ({np.nanmax(Qo[i, :])})', names[np.nanargmax(Qo[i, :])], f'Best action ({np.nanmax(Q[i, :])})', names[np.nanargmax(Q[i, :])] )
    