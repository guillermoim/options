import numpy as np
import pickle
from tqdm import tqdm
from utilities.utils_taxi import *

def Q_options(config_filepath, eps0, eps1, c1, c2):
    
    # Read configuration path
    CONFIG = pickle.load(open(f'{config_filepath}', 'rb'))

    # For each key in CONFIG, create a variable with same name (ignore warnings)
    for key in CONFIG:
        exec(f"global {key}; {key} = CONFIG['{key}']")

    # Initialize Value Functions
    Q = np.full((len(S), No), np.NaN)
    for i, x in enumerate(S):
        Q[i, applicable_options(x, DIM)] = 0

    Qo = np.full((No, len(abs_int_states), Na), np.NaN)
    for i, x_s in enumerate(abs_int_states):
        Qo[:, i, abs_applicable_actions(x_s, abs_states, P_)] = 0


    errors = []

    eps0 = 0.3
    eps1 = 0.15

    c1 = 1000
    c2 = 3000

    for k in tqdm(range(3000)):
        
        state = sample_states[np.random.choice(len(sample_states))]

        alpha = c1 / (c1+k+1)

        while state not in T:
            
            init_state = state
            init_idx = F.index(init_state)

            acc_reward = 0
            
            possible_options = applicable_options(state, DIM)

            # Select option with Softmax policy
            if np.random.random() < 1-eps0:
                o = np.nanargmax(Q[init_idx, :])
            else:
                o = np.random.choice(possible_options)
            
            leaving_state = None
        
            alpha_2 = c2 / (c2+k+1)
            
            while True: # while option not finished, follow policy for o until termination
                # Normalize state
                proj_state = (0, *state[0])
                idx_proj_state = abs_states.index(proj_state)
                LL_actions = abs_applicable_actions(proj_state, abs_states, P_)
                HL_actions = applicable_actions(state, F, P)
                
                if np.random.random() < 1 - eps1:
                    action = np.nanargmax(Qo[o, idx_proj_state, :])
                else:
                    action = np.random.choice(LL_actions)


                error = np.mean(np.abs(V_flat - np.nanmax(Q[E_set_idx, :], axis=1)))
                errors.append(error)

                # In this case, the only inconsistency between the HL and LL happens when the option selects an action
                # leading to a terminal state which does not exist in the HL. We opted for a design in which there is only
                # goal, terminal states. Any other terminal state is discarded from the MDP.

                # Thus, the option must terminate when a 'wrong' action is chosen.
                if action not in HL_actions:
                    leaving_state = state
                    # Update (learn) and select new action for the high-level
                    proj_next_state = compute_next_abs_state(proj_state, action)
                    #print(proj_state, action, proj_next_state)
                    sel = options_terminals.index(proj_next_state)
                    G = np.full(No, -1e6)
                    G[sel] = 0
                    Qo[:, idx_proj_state, action] = Qo[:, idx_proj_state, action] + alpha_2 * (-1 + G - Qo[:, idx_proj_state, action])
                    # Terminate option
                    eps1*=.99
                    break
                
                acc_reward-=1

                # Apply action and project new state
                next_state = compute_next_state(state, action)
                proj_next_state = (0 if state[1]==next_state[1] else 1, *next_state[0])
                idx_proj_next_state = abs_states.index(proj_next_state)

                state = next_state

                # Update Qo accordingly and leave if state is an option's terminal state
                if proj_next_state[0] == 1:
                    leaving_state = state
                    sel = options_terminals.index(proj_next_state)
                    G = np.full(No, -1e6)
                    G[sel] = 0
                    Qo[:, idx_proj_state, action] = Qo[:, idx_proj_state, action] + alpha_2 * (-1 + G - Qo[:, idx_proj_state, action])
                    eps1*=.99
                    break
                else:
                    Qo[:, idx_proj_state, action] = Qo[:, idx_proj_state, action] + alpha_2 * (-1 + np.nanmax(Qo[:, idx_proj_next_state, :], axis=1) - Qo[:, idx_proj_state, action])
            
            # Update high-level Q function
            idx_is = F.index(init_state)
            idx_ls = F.index(leaving_state)
            
            if leaving_state not in T:
                Q[idx_is, o] = Q[idx_is, o] + alpha * (acc_reward + np.nanmax(Q[idx_ls, :]) - Q[idx_is, o])    
            else:
                Q[idx_is, o] = Q[idx_is, o] + alpha * (acc_reward + R[idx_ls] - Q[idx_is, o])
    
    return errors