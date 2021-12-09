import numpy as np
import pickle
from tqdm import tqdm 
from utilities.utils_nroom import *

def Q_options(config_filepath, eps0, eps1, c1, c2):
    
    # Read configuration path
    CONFIG = pickle.load(open(f'{config_filepath}', 'rb'))

    # For each key in CONFIG, create a variable with same name (ignore warnings)
    for key in CONFIG:
        exec(f"{key} = CONFIG['{key}']")

    # Initialize Value Functions
    Q = np.full((len(S), No), np.NaN, dtype=np.float64)
    for i, x in enumerate(S):
        available_options = applicable_options(id_room(x, ROOM_SIZE), ROOM_SIZE, F, GOAL_POS)
        Q[i, available_options] = 0

    Qg = np.full((No, len(abs_int_states), Na), np.NaN, dtype=np.float64)
    for i, x in enumerate(abs_int_states):
        Qg[:, i, applicable_actions(x, abs_states, P_)] = 0


    # Declare epsilons
    eps0 = 0.15
    eps1 = 0.3

    errors = []

    c1 = 1000
    c2 = 3000

    for k in tqdm(range(N_MAX_SAMPLES)):
        
        state = E_set[np.random.choice(len(E_set))]

        alpha = c1 / (c1 + k + 1)

        while state not in T:
            init_state = state
            init_idx = F.index(init_state)

            acc_reward = 0

            possible_options = applicable_options(id_room(state, ROOM_SIZE), ROOM_SIZE, F, GOAL_POS)

            if np.random.random() < 1-eps0:
                o = np.nanargmax(Q[init_idx, :])
            else:
                o = np.random.choice(possible_options)
            
            leaving_state = None
            alpha_2 = c2 / (c2+k+1)

            while True:
                proj_state = project_state(state, id_room(state, ROOM_SIZE), ROOM_SIZE)
                idx_proj_state = abs_states.index(proj_state)
                LL_actions = applicable_actions(proj_state, abs_states, P_)
                HL_actions = applicable_actions(state, F, P)

                # I take the action for the 'projected' state, sampling from Softmax policy
                if np.random.random() < 1-eps1:
                    action = np.nanargmax(Qg[o, idx_proj_state, :])
                else:
                    action = np.random.choice(LL_actions)

                acc_reward -= 1

                V = np.nanmax(Q[E_set_idx, :], axis=1)

                error = np.mean(np.abs(V_flat - V))
                errors.append(error)

                # In this case, the only inconsistency between the HL and LL happens when the option selects an action
                # leading to a terminal state which does not exist in the HL. We opted for a design in which there is only
                # goal, terminal states. Any other terminal state is discarded from the MDP.
                if action not in HL_actions:
                    
                    proj_next_state = compute_next_state(proj_state, action)
                    sel = options_terminals.index(proj_next_state)
                    G = np.full(No, -1e6)
                    G[sel] = 0
                    Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (-1 + G - Qg[:, idx_proj_state, action])
                    # Terminate option
                    leaving_state = state
                    action = np.random.choice(HL_actions)
                    eps1*=0.99

                # Apply action and project new state
                next_state = compute_next_state(state, action)
                r = -1
                proj_next_state = project_state(next_state, id_room(state, ROOM_SIZE), ROOM_SIZE)
                idx_proj_next_state = abs_states.index(proj_next_state)

                state = next_state

                # Update Qg accordingly and leave if state is an option's terminal state
                if proj_next_state in options_terminals:
                    leaving_state = next_state
                    sel = options_terminals.index(proj_next_state)
                    G = np.full(No, -1e6)
                    G[sel] = 0
                    Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (r + G - Qg[:, idx_proj_state, action])
                    eps1*=0.99
                    break
                else:
                    Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (r + np.nanmax(Qg[:, idx_proj_next_state, :], axis=1) - Qg[:, idx_proj_state, action])

            # Update high-level Q function
            i_is = F.index(init_state)
            i_ls = F.index(leaving_state)

            if leaving_state not in T:
                Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward + np.nanmax(Q[i_ls, :]) - Q[i_is, o])
            else:
                Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward + R[i_ls] - Q[i_is, o])


    return errors