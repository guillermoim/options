from NRoomDomainStochastic import NRoomDomainStochastic
from NRoomStochasticOptions import NRoomStochasticOptions

import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from tqdm import tqdm
from itertools import product

dims = (3,3)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,0)]

states_path = 'solutions_rooms3x3_size5x5/abs_states_5x5.pkl'
passive_dyn = 'solutions_rooms3x3_size5x5/passive_dynamics_goal@2-3_5x5.txt'
optimal_policy = 'solutions_rooms3x3_size5x5/optimal_policy_options_goal@2-3_5x5.npy'

env = NRoomDomainStochastic(dims, room_size, goal_pos, goal_rooms, path='solutions_lmdps/policy_3x3_goal@0-0_rooms5x5.txt')
options_env = NRoomStochasticOptions(states_path, passive_dyn, optimal_policy)

abs_states = options_env.states

def option_is_applicable(room, room_size, option, states, goal_pos):
    X, Y = room[0]*room_size, room[1]*room_size
    goal_pos = X+goal_pos[0], Y+goal_pos[1]
    ts = [(0, X-1, (Y+room_size)//2), (0, (X+room_size)//2, Y-1), (0, (X+room_size)//2, Y+room_size), (0, X+room_size, (Y+room_size)//2), (1, *goal_pos)]
    return ts[option] in states
   
def _applicable_options(room, room_size, states, goal_pos):
    return [o for o in range(5) if option_is_applicable(room, room_size, o, states, goal_pos)]

def _get_exit_states(dims, room_size, states):
    
    exit_states = []
    rooms = set([room for room in product(range(dims[0]), range(dims[1]))])
   
    for room in rooms:
        X, Y = room[0]*room_size, room[1]*room_size
        ts = [(0, X-1, (Y+room_size)//2), (0, (X+room_size)//2, Y-1), (0, (X+room_size)//2, Y+room_size), (0, X+room_size, (Y+room_size)//2)]
        local = [t for t in ts if t in states]
        exit_states.extend(local)
    return exit_states

E_set = _get_exit_states(dims, room_size, env.states)
E_set_idx = [env.states.index(x) for x in E_set]


# Q-Learning
actions = [0,1,2,3,4]

o_terminals = [(0, -1, room_size//2), (0, room_size//2, -1), (0, room_size//2, room_size), (0, room_size, room_size//2), (1, *goal_pos)]
No = len(o_terminals)

# High-level Q
Q = np.full((len(env.states), No), np.NaN, dtype=np.float64)
policy = np.full((len(env.states), No), np.NaN)
for i, x in enumerate(env.states):
    if x in env.terminal_states:
        ntn = env.P[:, env.states.index(x)].nonzero()[0][0]
        stn = env.states[ntn]
        p_options = _applicable_options(_id_room(stn, room_size), room_size, env.states, goal_pos)
        Q[i, p_options] = 0
        policy[i, p_options] = 1 / len(p_options)
        continue
    p_options = _applicable_options(_id_room(x, room_size), room_size, env.states, goal_pos)
    Q[i, p_options] = 0
    policy[i, p_options] = 1 / len(p_options)

# Q_o for options
Qg = np.full((No, len(options_env.states[:-5]), env.Na), np.NaN, dtype=np.float64)
O_policies = np.full((No, len(options_env.states[:-5]), env.Na), np.NaN)
for i, x in enumerate(options_env.states[:-5]):
    Qg[:, i, options_env.applicable_actions(x)] = 0
    O_policies[:, i, options_env.applicable_actions(x)] = 1 / len(options_env.applicable_actions(x))

Q_flat = np.loadtxt('Q_stochastic_3x3.txt')

gamma = 1

errors = []

eps0 = 0.3
eps1 = 0.4

c1 = 1000
c2 = 3000

for k in tqdm(range(5000)):
    
    env.current_state = E_set[np.random.choice(len(E_set))]
    options_env.current_state = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)
    alpha = c1 / (c1+k+1)

    while env.current_state not in env.goal_states:
        
        init_state = env.current_state
        c_idx = env.states.index(init_state)

        t = 0

        # Select option from Softmax option
        
        p_options = _applicable_options(_id_room(env.current_state, room_size), room_size, env.states, goal_pos)
        o = np.random.choice(p_options, p = policy[c_idx, p_options])

        # Retrieve option's policy
        policy_o = O_policies[o, :, :]

        leaving_state = None

        os = None

        alpha_2 = c2 / (c2+k+1)

        while True: # while option not finished, follow policy for o until termination
                        
            # Normalize state
            n_s = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)
            i_n_s = abs_states.index(n_s)
            p_o_actions = options_env.applicable_actions(n_s)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            
            if np.random.random() < 1-eps1:
                action = np.nanargmax(Qg[o, i_n_s, :])
            else:
                action = np.random.choice(p_o_actions)

            error = np.mean(np.abs(np.nanmax(Q_flat[E_set_idx, :], axis=1) - np.nanmax(Q[E_set_idx, :], axis=1)))
            errors.append(error)

            if action not in p_actions:
                
                # Update (learn) and select new action for the high-level
                options_env.current_state = n_s
                x_os, x_ns, x_r = options_env.apply_action(o, action)
                G = options_env.r[options_env.states.index(x_ns),:]
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (x_r + gamma * G - Qg[:, i_n_s, action])
                # Terminate option
                leaving_state = env.current_state
                action = np.random.choice(p_actions)
                
                options_env.current_state = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)

            # Apply action and project new state
            os, ns, R = env.apply_action(action)
            n_ns = _normalize_cell(ns, _id_room(os, room_size), room_size)
            i_n_ns = abs_states.index(n_ns)

            t+=R

            # Update Qg accordingly and leave if state is an option's terminal state
            if n_ns in o_terminals:
                leaving_state = ns
                x_os, x_ns, x_r = options_env.apply_action(o, action)
                G = options_env.r[options_env.states.index(x_ns),:]
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (x_r + gamma * G - Qg[:, i_n_s, action])
                options_env.current_state = _normalize_cell(env.current_state, _id_room(env.current_state, room_size))

                #policy = np.exp(Qg[:, :-5]) / np.nansum(np.exp(Q[:len(env.interior_states)]), axis=1, keepdims=True)

                break
            else:
                x_os, x_ns, x_r = options_env.apply_action(o, action)
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * ( x_r + gamma * np.nanmax(Qg[:, i_n_ns, :], axis=1) - Qg[:, i_n_s, action])
                options_env.current_state = n_ns

        
        # Update high-level Q function
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)

        if leaving_state not in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + np.nanmax(Q[i_ls, :]) - Q[i_is, o])    
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + env.r[leaving_state] - Q[i_is, o])

    # Derive new high level Softmax policy
    policy = np.exp(Q[:len(env.interior_states)]) / np.nansum(np.exp(Q[:len(env.interior_states)]), axis=1, keepdims=True)


np.savetxt('results/rooms_stochastic_H_Q_3x3.txt', Q)
np.save('results/rooms_stochastic_options_Q_3x3', Qg)
np.savetxt('results/rooms_stochastic_H_errors_3x3.txt', np.array(errors))
