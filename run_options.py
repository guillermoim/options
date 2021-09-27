from networkx.utils.heaps import PairingHeap
from rooms_domain import NRoomDomain
from partitions_tracker import HierarchyTracker
import numpy as np
from partitions_tracker import _id_room, _normalize_cell, _de_normalize_cell

dims = (2,2)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(1,1)]

infty = 1e6

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)])
abs_states = abs_room.states

# Q-Learning
actions = [0,1,2,3,4]

No = 5

Q = np.zeros((env.Ns, No))
policy = np.full((env.Ns, No), 0.2)

# Q_o for options
Qg = np.full((No, room_size**2, env.Na), np.NaN)
O_policies = np.full((No, room_size**2, env.Na), 0.2)

o_terminals = [(0, -1, 2), (0, 2, -1), (0, 2, 5), (0, 5, 2), (1, 2, 3)]

gamma = 1

for k in range(10000):
    
    env.reset()
    alpha = 0.25      #c / (c + k + 1)
    alpha_2 = 0.5
    accumulated_reward = 0

    current_room = None

    while env.current_state not in env.terminal_states:
        # select option 
        o = np.random.choice(range(No), p=policy[env.states.index(env.current_state), :])

        policy_o = O_policies[o, :, :]

        acc_reward_option = 0
        init_state = env.current_state
        leaving_state = None

        t = 0

        while True: # while option not finished
            # Follow policy for o until termination
            
            # Normalize state
            i_n_s = abs_states.index(_normalize_cell(env.current_state, _id_room(env.current_state), room_size))
            # I take the action for the 'projected' option
            action = np.random.choice(range(env.Na), p=policy_o[i_n_s, :])
            # I applt the action and normalize the new state
            os, ns, r = env.apply_action(action)
            i_n_ns = abs_states.index(_normalize_cell(ns, _id_room(os), room_size))

            acc_reward_option += gamma ** t * r

            if abs_states[i_n_ns] not in o_terminals:
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * np.max(Qg[:, i_n_ns, :]) - Qg[:, i_n_s, action])

            else:
                leaving_state = env.current_state
                sel = o_terminals.index(_normalize_cell(ns, _id_room(os), room_size))
                G = np.full(No, -infty)
                G[sel] = 0
                print(leaving_state, G)
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * G - Qg[:, i_n_s, action])
                # last state so update policy
                O_policies = np.exp(Qg) / np.exp(Qg).sum(axis=2, keepdims=1) 
                break

            t+=1

        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)
        
        if leaving_state not in env.terminal_states:
            #print('updating', init_state, _id_room(init_state), ['T', 'L', 'R', 'B', 'G'][o], leaving_state, 'with total reward', acc_reward_option + np.max(Q[i_ls, :]))
            Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward_option + gamma ** t * np.max(Q[i_ls, :]) - Q[i_is, o])
        else:
            #print('updating', init_state, _id_room(init_state), ['T', 'L', 'R', 'B', 'G'][o], leaving_state, 'with total reward', acc_reward_option + env.r[leaving_state])
            Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward_option + gamma ** t * env.r[leaving_state] - Q[i_is, o])

    # Derive new policy
    sum_Q = np.sum(np.exp(Q[:len(env.terminal_states), ]), axis=1, keepdims=1)
    policy[:len(env.terminal_states), :] = np.exp(Q[:len(env.terminal_states)]) / sum_Q[:len(env.terminal_states)]


# simulation
options_names = ['T', 'L', 'R', 'B', 'G']

greedy_policy = np.argmax(Q, axis=1).tolist()

print(env.goal_states)

np.savetxt('Hierarchical_Q.txt', Q)
np.save('Options_Q', Qg)
