from partitions_tracker import _id_room
from rooms_domain import NRoomDomain
import numpy as np
from itertools import product

dims = (3,3)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(2,0)]

infty = 1e6

def _get_exit_states(dims, room_size, states):
    
    exit_states = []
    rooms = set([room for room in product(range(dims[0]), range(dims[1]))])
   
    for room in rooms:
        X, Y = room[0]*room_size, room[1]*room_size
        ts = [(0, X-1, (Y+room_size)//2), (0, (X+room_size)//2, Y-1), (0, (X+room_size)//2, Y+room_size), (0, X+room_size, (Y+room_size)//2)]
        local = [t for t in ts if t in states]
        exit_states.extend(local)
    return exit_states

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states

E_set = _get_exit_states(dims, room_size, env.states)
E_set_idx = [env.states.index(x) for x in E_set]

Q_flat = np.loadtxt('results/rooms_Flat_Q_3x3.txt')

Q_h = np.loadtxt('results/rooms_H_Q_3x3.txt')

V_flat = np.nanmax(Q_flat[:len(env.interior_states)], axis=1)
V_h = np.nanmax(Q_h[:len(env.interior_states)], axis=1)

print(Q_h)


for s in E_set:
     i = env.states.index(s)
     print(s, _id_room(s), V_flat[i], V_h[i],  np.abs(V_flat[i] - V_h[i]))

