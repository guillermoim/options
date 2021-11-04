from NRoomDomainStochastic import NRoomDomainStochastic
from NRoomStochasticOptions import NRoomStochasticOptions

import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from tqdm import tqdm
from itertools import product

import warnings

dims = (3,3)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,0)]

states_path = 'solutions_rooms3x3_size5x5/abs_states_5x5.pkl'
passive_dyn = 'PROV_SOLUTIONS/passive_dynamics_NROOM_5x5.npy'
optimal_policy = 'PROV_SOLUTIONS/optimal_policies_NROOM_5x5.npy'

optimal_controls = 'PROV_SOLUTIONS/optimal_policy_3x3.npy'

env = NRoomDomainStochastic(dims, room_size, goal_pos, goal_rooms, high_level=False, path=optimal_controls)
options_env = NRoomStochasticOptions(states_path, passive_dyn, optimal_policy)

abs_states = options_env.states

def option_is_applicable(room, room_size, option, states, goal_pos):
    X, Y = room[0]*room_size, room[1]*room_size
    goal_pos = X+goal_pos[0], Y+goal_pos[1]
    ts = [(0, X-1, Y+room_size//2), (0, X+room_size, Y+room_size//2), (0, X+room_size//2, Y-1), (0, X+room_size//2, Y+room_size), (1, *goal_pos)]
    return ts[option] in states
   
def _applicable_options(room, room_size, states, goal_pos):
    return [o for o in range(5) if option_is_applicable(room, room_size, o, states, goal_pos)]

def _get_exit_states(dims, room_size, states):
    
    exit_states = []
    rooms = set([room for room in product(range(dims[0]), range(dims[1]))])

    rooms = sorted(list(rooms), key=lambda x: x[0])
   
    for room in rooms:
        X, Y = room[0]*room_size, room[1]*room_size
        ts = [(0, X-1, Y+room_size//2), (0, X+room_size, Y+room_size//2), (0, X+room_size//2, Y-1), (0, X+room_size//2, Y+room_size)]
        local = []
        for t in ts:
            if t in states:
                exit_states.append(t)
                local.append(t)
    return exit_states

E_set = _get_exit_states(dims, room_size, env.states)
E_set_idx = [env.states.index(x) for x in E_set]


# Q-Learning
actions = [0,1,2,3,4]

o_terminals = [(0, -1, room_size//2), (0, room_size, room_size//2), (0, room_size//2, -1), (0, room_size//2, room_size), (1, *goal_pos)]
No = len(o_terminals)

# High-level Q
Q = np.full((len(env.states), No), np.NaN, dtype=np.float64)
policy = np.full((len(env.states) - len(env.terminal_states), No), np.NaN, dtype=np.float64)

for i, x in enumerate(env.states):
    if x in env.terminal_states:
       continue
    p_options = _applicable_options(_id_room(x, room_size), room_size, env.states, goal_pos)
    Q[i, p_options] = 0
    policy[i, p_options] = 1 / len(p_options)

# Q_o for options
Qg = np.full((No, len(options_env.states), env.Na), np.NaN, dtype=np.float64)
intra_option_policy = np.full((No, len(options_env.states), env.Na), np.NaN)
for o in range(No):
    for i, x in enumerate(options_env.states[:-5]):
        poss_actions = options_env.applicable_actions(o, x)
        Qg[o, i, poss_actions] = 0
        intra_option_policy[o, i, poss_actions] = 1 / len(poss_actions)

Q_flat = Q.copy()# np.loadtxt('Q_stochastic_1x1.txt')

Qg_initial = np.copy(Qg)

gamma = 1

errors = []

c1 = 1000
c2 = 3000

eps0 = 0.3
eps1 = 0.3

warnings.filterwarnings("error")

for k in tqdm(range(1000)):
    
    #env.current_state = E_set[np.random.choice(range(len(E_set)))]
    env.reset()
    options_env.current_state = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)
    alpha = c1 / (c1+k+1)

    while env.current_state not in env.terminal_states:
        
        init_state = env.current_state
        c_idx = env.states.index(init_state)

        t = 0

        # Select option from Softmax option
        p_options = _applicable_options(_id_room(env.current_state, room_size), room_size, env.states, goal_pos)
        
        if np.random.random() < 1-eps0:
            o = np.nanargmax(Q[c_idx, :])
        else:
            o = np.random.choice(p_options)

        o = np.random.choice(p_options)

        leaving_state = None

        os = None

        alpha_2 = c2 / (c2+k+1)

        current_room = _id_room(env.current_state, room_size)

        while True: # while option not finished, follow policy for o until termination
                        
            # Normalize state
            proj_state = _normalize_cell(env.current_state, current_room, room_size)
            idx_proj_state = abs_states.index(proj_state)
            p_o_actions = options_env.applicable_actions(o, proj_state)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            if np.random.random() < 1-eps1:
                action = np.nanargmax(Qg[o, idx_proj_state, :])
            else:
                action = np.random.choice(p_o_actions)

            p_actions = env.applicable_actions(env.current_state)

            # Apply action and project new state
            old_state, next_state, T = env.apply_action(action)
            proj_next_state = _normalize_cell(next_state, current_room, room_size)
            idx_proj_next_state = abs_states.index(proj_next_state)
            
            t+=T

            # Update Qg accordingly and leave if state is an option's terminal state
            if proj_next_state in o_terminals:
                #x_os, x_ns, x_r = options_env.apply_action(o, action)
                G = options_env.r[idx_proj_next_state, :]
                #old = Qg[o, idx_proj_state, action]
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (options_env.r[idx_proj_state, :] + gamma * G - Qg[:, idx_proj_state, action])
                options_env.current_state = proj_next_state
                #eps1*=.99
                leaving_state = next_state
                break
            else:
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (options_env.r[idx_proj_state, :] + gamma * np.nanmax(Qg[:, idx_proj_next_state, :], axis=1) - Qg[:, idx_proj_state, action])
                options_env.current_state = proj_next_state
        
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)

        if leaving_state not in env.terminal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (t + np.nanmax(Q[i_ls, :]) - Q[i_is, o])    
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (t + env.r[leaving_state] - Q[i_is, o])

        error = np.mean(np.abs(np.nanmax(Q_flat[:25, :], axis=1) - np.nanmax(Q[:25, :], axis=1)))
        errors.append(error)

    #eps0*=.99

np.savetxt('results/rooms_stochastic_H_Q_3x3.txt', Q)
np.save('results/rooms_stochastic_options_Q_3x3', Qg)
np.savetxt('results/rooms_stochastic_H_errors_3x3.txt', np.array(errors))
