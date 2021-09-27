from rooms_domain import NRoomDomain
import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from itertools import product

dims = (2,2)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(1,1)]


env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, goal_reward=1, non_goal_reward=0, penalty=0)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)], goal_reward=1, non_goal_reward=0, penalty=0)
abs_states = abs_room.states

def _effect_option_at(room, option):
    if option == 0:
        return room[0]-1, room[1]
    elif option == 1:
        return room[0], room[1]-1
    elif option == 2:
        return room[0], room[1]+1
    elif option == 3:
        return room[0]+1, room[1]
    elif option == 4:
        return room

def _option_is_applicable(room, option, dims, goal_rooms):
    X_,Y_ = _effect_option_at(room, option)

    res = True

    if dims[0]-1<X_ or X_<0:
        res =  False
    elif dims[1]-1<Y_ or Y_<0:
        res = False
    if room == (X_, Y_) and room not in goal_rooms:
        res = False

    return res

def _applicable_options(room, dims, goal_rooms):
    return [o for o in range(5) if _option_is_applicable(room, o, dims, goal_rooms)]

# Q-Learning
actions = [0,1,2,3,4]

No = 5

Q = np.full((env.Ns, No), np.NaN)
for i, x in enumerate(env.interior_states):
    Q[i, _applicable_options(_id_room(x), dims, goal_rooms)] = 1


K = np.nansum(Q, axis=1, keepdims=1)

print(np.nansum(np.exp(Q), axis=1, keepdims=1).shape, Q.shape)

print(Q / K)