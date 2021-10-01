from partitions_tracker import _id_room
from rooms_domain import NRoomDomain
import numpy as np

dims = (5,5)
room_size = 3
goal_pos = (1,1)
goal_rooms = [(0,0)]

infty = 1e6

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states

Q_flat = np.loadtxt('results/Flat_Q_softmax.txt')
policy_flat = np.loadtxt('results/Flat_Policy_softmax.txt')

Q_h = np.loadtxt('results/H_Q_softmax.txt')
policy_h = np.loadtxt('results/H_Policy_softmax.txt')

V_flat = np.nanmax(Q_flat, axis=1)
V_h = np.nanmax(Q_h, axis=1)

print(Q_h.shape, Q_flat.shape)

for i, s in enumerate(env.interior_states):
     print(s, V_flat[i], V_h[i])
    