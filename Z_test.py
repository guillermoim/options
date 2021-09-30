from rooms_domain import NRoomDomain
import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from tqdm import tqdm

dims = (2,2)
room_size = 3
goal_pos = (1,1)
goal_rooms = [(0,0)]

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, goal_reward=0, non_goal_reward=-1)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)], goal_reward=0, non_goal_reward=-1)
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

o_terminals = [(0, -1, room_size//2), (0, room_size//2, -1), (0, room_size//2, room_size), (0, room_size, room_size//2), (1, *goal_pos)]
No = len(o_terminals)

# High-level Q
Q = np.full((len(env.states), No), np.NaN)
policy = np.full((len(env.states), No), np.NaN)
for i, x in enumerate(env.states):
    if x in env.terminal_states:
        ntn = env.P[:, env.states.index(x)].nonzero()[0][0]
        stn = env.states[ntn]
        p_options = _applicable_options(_id_room(stn, room_size), dims, goal_rooms)
        Q[i, p_options] = 0
        policy[i, p_options] = 1 / len(p_options)
        continue
    Q[i, _applicable_options(_id_room(x, room_size), dims, goal_rooms)] = 0
    policy[i, _applicable_options(_id_room(x, room_size), dims, goal_rooms)] = 1 / len(_applicable_options(_id_room(x, room_size), dims, goal_rooms))

# Q_o for options
Qg = np.full((No, len(abs_room.interior_states), env.Na), np.NaN)
O_policies = np.full((No, len(abs_room.interior_states), env.Na), np.NaN)
for i, x in enumerate(abs_room.interior_states):
    Qg[:, i, abs_room.applicable_actions(x)] = 0
    O_policies[:, i, abs_room.applicable_actions(x)] = 1 / len(abs_room.applicable_actions(x))

Qg = np.load('results/options_Q_softmax.npy')
O_policies = np.exp(Qg) / np.nansum(np.exp(Qg), keepdims=1, axis=2)

gamma = 1

env.current_state = (0, 0, 1)
print(env.applicable_actions(env.current_state))
