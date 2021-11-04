from partitions_tracker import _id_room
from rooms_domain import NRoomDomain
import numpy as np



dims = (8,8)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,7)]

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, True)
Q_flat = np.loadtxt('results/rooms_Flat_Q_8x8.txt')
Q_g = np.load('results/rooms_stochastic_options_Q_3x3.npy')
print(len(env.states), Q_flat.shape)

V_flat = np.nanmax(Q_flat, axis=1)

print(Q_g)

'''for i, s in enumerate(env.interior_states):
     print(s, V_flat[i])'''