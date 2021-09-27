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
    Q[i, _applicable_options(_id_room(x), dims, goal_rooms)] = 0

# Q_o for options
Qg = np.full((No, room_size**2, env.Na), np.NaN)
for i, x in enumerate(abs_room.interior_states):
    Qg[:, i, abs_room.applicable_actions(x)] = 0
O_policies = np.full((No, room_size**2, env.Na), 0.2)

o_terminals = [(0, -1, 2), (0, 2, -1), (0, 2, 5), (0, 5, 2), (1, 2, 3)]

gamma = 0.9

p = 0.3

#print(env.terminal_states)

for k in range(200000):
    
    env.reset()
    alpha = 0.1      # c / (c + k + 1)
    alpha_2 = 0.1
    accumulated_reward = 0

    current_room = None

    while env.current_state not in env.terminal_states:
        # select option
        p_options = _applicable_options(_id_room(env.current_state), dims, goal_rooms)
        state_idx = env.states.index(env.current_state)
        if np.random.random() > p:
            o = np.nanargmax(Q[state_idx, :])
        else:
            o = np.random.choice(p_options)
        #o = np.random.choice(range(No))
        policy_o = O_policies[o, :, :]

        acc_reward_option = 0
        init_state = tuple(env.current_state)
        leaving_state = None

        t = 0
        #print('At', env.current_state, 'executing', o)
        while True: # while option not finished
            # Follow policy for o until termination
            
            # Normalize state
            i_n_s = abs_states.index(_normalize_cell(env.current_state, _id_room(env.current_state), room_size))
            # I take the action for the 'projected' option
            p_actions = env.applicable_actions(env.current_state)
            if np.random.random() > p:
                action = np.nanargmax(Qg[o, i_n_s, :])
            else:
                action = np.random.choice(p_actions)
            # I applt the action and normalize the new state
            os, ns, r = env.apply_action(action)
            n_ns = _normalize_cell(ns, _id_room(os), room_size)
            i_n_ns = abs_states.index(n_ns)

            acc_reward_option += gamma**t * r
            
            if n_ns in o_terminals:
                leaving_state = ns
                sel = o_terminals.index(n_ns)
                G = np.full(No, 0)
                G[sel] = 1
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * G - Qg[:, i_n_s, action])
                break
            else:
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * np.nanmax(Qg[:, i_n_ns, :], axis=1) - Qg[:, i_n_s, action])
            
            t+=1

        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)
        
        if leaving_state not in env.terminal_states:
            print('Ending option', o, 'from', init_state, 'to', leaving_state, 'with acc. reward', acc_reward_option, np.nanmax(Q[i_ls, :]))
            Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward_option + gamma**t * np.nanmax(Q[i_ls, :]) - Q[i_is, o])
        else:
            print('Ending option', o, 'from', init_state, 'to', leaving_state, 'with acc. reward', acc_reward_option, env.r[leaving_state])
            Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward_option + gamma**t * env.r[leaving_state] - Q[i_is, o])
            break

    # Derive new policy
    # sum_Q = np.sum(np.exp(Q[:len(env.terminal_states)]), axis=1, keepdims=1)
    # policy[:len(env.terminal_states), :] = np.exp(Q[:len(env.terminal_states)]) / sum_Q[:len(env.terminal_states)]


# simulation
options_names = ['T', 'L', 'R', 'B', 'G']

greedy_policy = np.argmax(Q, axis=1).tolist()

print(env.goal_states)

np.savetxt('results/H_Q_eps_dicounted.txt', Q)
np.save('results/options_Q_eps_dicounted', Qg)
