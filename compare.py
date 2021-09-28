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

Q_flat = np.loadtxt('results/Flat_Q_softmax.txt')
policy_flat = np.loadtxt('results/Flat_Policy_softmax.txt')

Q_h = np.loadtxt('results/H_Q_softmax.txt')
policy_h = np.loadtxt('results/H_Policy_softmax.txt')

V_flat = np.nansum(np.multiply(Q_flat, policy_flat), axis=1)
V_h = np.nansum(np.multiply(Q_h, policy_h), axis=1)

for i, s in enumerate(env.interior_states):
     print(s, V_flat[i], V_h[i])
    