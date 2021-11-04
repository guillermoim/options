from taxi_domain import TaxiDomain, _create_taxi_room
import numpy as np
from tqdm import tqdm
from itertools import product

## Auxiliary
def _compute_next_abs_state(state, action):
    next_state = None
    if action == 0: # TOP
        next_state=(state[0], state[1]-1, state[2])
    elif action==1: # LEFT
        next_state=(state[0], state[1], state[2]-1)
    elif action==2: # RIGHT
        next_state=(state[0], state[1], state[2]+1)
    elif action==3: # BOTTOM
        next_state=(state[0], state[1]+1, state[2])
    elif action==4: # PICKUP
        if state[0] == 0:
            next_state = (1, state[1], state[2])    
    elif action==5: # NoOP
        next_state = state
    
    return next_state


def _is_legal_move(states, P, state, next_state):
    p1 = next_state in states
    p2 = P[states.index(state), states.index(next_state)] > 0 if p1 else False
    return p1 and p2

def _applicable_actions(abs_state, states, P):
    
    actions = []
    
    for a in range(6):
        abs_next_state = _compute_next_abs_state(abs_state, a)
        p1 = abs_next_state in states
        p2 = P[states.index(abs_state), states.index(abs_next_state)] > 0 if p1 else False
        if p1 and p2: actions.append(a)
    
    return actions

dim = 5

env = TaxiDomain(dim)

options_corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]
options_terminals = [(1, 0, 0), (1, dim-1, 0), (1, 0, dim-1), (1, dim-1, dim-1)]

# Compute exit states
E_set = [(a, 'TAXI', b) for a in options_corners for b in options_corners if a!=b]
E_set_idx = [env.states.index(e) for e in E_set]


def _applicable_options(state, options_corners):
    
    # compute applicable options at each state
    options = []

    # if passenger in at pickup location, then applicable option is to go to pickup location
    if state[1] != 'TAXI':
        options.append(options_corners.index(state[1]))
    # otherwise, the passenger is at the taxi, thus the taxi can go to any location
    elif state[1] == 'TAXI':
        #options.append(options_corners.index(state[2]))
        options.append(options_corners.index(state[2]))

    return options  

# Q-Learning
# actions: T, L, R, B, CHANGE, NoOP
actions = list(range(5))

No = 4

# High-level Q
Q = np.full((len(env.states), No), np.NaN)
policy = np.full((len(env.states), No), np.NaN)

for i, x in enumerate(env.states):
    if not x in env.terminal_states:
        possible_options = _applicable_options(x, options_corners)
        Q[i, possible_options] = 0
        policy[i, possible_options] = 1 / len(possible_options)

# Q_o for options
abs_states, abs_P = _create_taxi_room(dim)

Qg = np.full((No, len(abs_states), env.Na), np.NaN)
O_policies = np.full((No, len(abs_states), env.Na), np.NaN)

for i, x in enumerate(abs_states):
    if x[0]: continue
    x_actions = []
    
    for a in actions:
        xs = _compute_next_abs_state(x, a)
        if _is_legal_move(abs_states, abs_P, x, xs):
            x_actions.append(a)
    
    Qg[:, i, x_actions] = 0
    O_policies[:, i, x_actions] = 1 / len(x_actions)

Q_flat = np.loadtxt('results/TAXI_Flat_Q_5x5.txt')

gamma = 1

errors = []

eps0 = 0.3
eps1 = 0.15

c1 = 1000
c2 = 3000

for k in tqdm(range(3000)):
    
    env.reset()

    alpha = c1 / (c1+k+1)

    while env.current_state not in env.terminal_states:
        
        init_state = env.current_state
        c_idx = env.states.index(init_state)

        t = 0
        
        poss_options = _applicable_options(env.current_state, options_corners)

        # Select option with Softmax policy
        if np.random.random() < 1-eps0:
            o = np.nanargmax(Q[c_idx, :])
        else:
            o = np.random.choice(poss_options)
        
        leaving_state = None
        os = None
       
        alpha_2 = c2 / (c2+k+1)
        
        while True: # while option not finished, follow policy for o until termination
            # Normalize state
            proj_state = (0, *env.current_state[0])
            idx_proj_state = abs_states.index(proj_state)
            poss_actions_option = _applicable_actions(proj_state, abs_states, abs_P)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            
            if np.random.random() < 1 - eps1:
                action = np.nanargmax(Qg[o, idx_proj_state, :])

            else:
                action = np.random.choice(poss_actions_option)

            error = np.mean(np.abs(np.nanmax(Q_flat[E_set_idx, :], axis=1) - np.nanmax(Q[E_set_idx, :], axis=1)))
            errors.append(error)

            if action not in p_actions:
                leaving_state = env.current_state
                # Update (learn) and select new action for the high-level
                proj_next_state = _compute_next_abs_state(proj_state, action)
                #print(proj_state, action, proj_next_state)
                sel = options_terminals.index(proj_next_state)
                G = np.full(No, -1e6)
                G[sel] = 0
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (-1 + gamma * G - Qg[:, idx_proj_state, action])
                # Terminate option
                action = np.random.choice(p_actions)
                eps1*=.99
            
            t+=1

            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            proj_next_state = (0 if os[1]==ns[1] else 1, *ns[0])
            idx_proj_next_state = abs_states.index(proj_next_state)

            # Update Qg accordingly and leave if state is an option's terminal state
            if proj_next_state[0] == 1:
                leaving_state = env.current_state
                sel = options_terminals.index(proj_next_state)
                G = np.full(No, -1e6)
                G[sel] = 0
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (r + gamma * G - Qg[:, idx_proj_state, action])
                eps1*=.99
                break
            else:
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (r + gamma * np.nanmax(Qg[:, idx_proj_next_state, :], axis=1) - Qg[:, idx_proj_state, action])
        
        # Update high-level Q function
        idx_init_state = env.states.index(init_state)
        idx_leaving_state = env.states.index(leaving_state)
        
        if leaving_state not in env.goal_states:
            Q[idx_init_state, o] = Q[idx_init_state, o] + alpha * (-t + np.nanmax(Q[idx_leaving_state, :]) - Q[idx_init_state, o])    
        else:
            Q[idx_init_state, o] = Q[idx_init_state, o] + alpha * (-t + env.r[leaving_state]- Q[idx_init_state, o])


np.savetxt('results/TAXI_H_Q_5x5_T.txt', Q)
np.save('results/TAXI_options_Q_5x5_T', Qg)
np.savetxt('results/TAXI_H_errors_5x5_T.txt', np.array(errors))

