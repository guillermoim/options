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

#O_policies = np.exp(Qg) / np.nansum(np.exp(Qg), keepdims=1, axis=2)

gamma = 1


for k in tqdm(range(10000)):
    
    env.reset(1)
    alpha = 0.1     

    while env.current_state not in env.terminal_states:
        
        init_state = env.current_state

        t = 0

        # Select option from Softmax option
        
        p_options = _applicable_options(_id_room(env.current_state, room_size), dims, goal_rooms)
        
        o = np.random.choice(p_options, p=policy[env.states.index(env.current_state), p_options])

        # Retrieve option's policy
        policy_o = O_policies[o, :, :]

        leaving_state = None

        os = None
        room = None

        if env.current_state in env.terminal_states:
            print(env.current_state)
        
        while True: # while option not finished, follow policy for o until termination
            # Normalize state
            i_n_s = abs_states.index(_normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size))
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            print(Qg[o, i_n_s, p_actions])
            weights = (np.exp(Qg[o, i_n_s, p_actions]) / np.nansum(np.exp(Qg[o, i_n_s, p_actions]), keepdims=1, axis=1))
            print(env.current_state, p_actions, weights.shape)
            action = np.random.choice(p_actions, p=weights)

            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            n_ns = _normalize_cell(ns, _id_room(os, room_size), room_size)
            i_n_ns = abs_states.index(n_ns)

            room = _id_room(os, room_size)

            t+=1

            # Update Qg accordingly and leave if state is an option's terminal state
            if n_ns in o_terminals:
                leaving_state = ns
                sel = o_terminals.index(n_ns)
                G = np.full(No, env.penalty)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha * (r + gamma * G - Qg[:, i_n_s, action])
                break
            else:
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha * (r + gamma * np.nanmax(Qg[:, i_n_ns, :], axis=1) - Qg[:, i_n_s, action])

        # Update high-level Q function
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)
        
        if leaving_state in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + np.nanmax(Q[i_ls, :]) - Q[i_is, o]) 
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + env.r[leaving_state] - Q[i_is, o])

    # Derive new high level Softmax policy
    policy = np.exp(Q[:len(env.interior_states)]) / np.nansum(np.exp(Q[:len(env.interior_states)]), axis=1, keepdims=1)

np.savetxt('results/H_Q_softmax.txt', Q)
np.savetxt('results/H_Policy_softmax.txt', policy)
np.save('results/options_Q_softmax', Qg)
