from operator import index
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
            next_state = 1, state[1], state[2]    
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

options_names = ['TL', 'TR', 'BL', 'BR']
options_corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]
options_terminals = [(1, 0,0), (1, dim-1,0), (1, 0,dim-1), (1, dim-1,dim-1)]

# Compute exit states
exit_states = set()
starting_states = set()
E1 = [(a, 'TAXI', b) for a in options_corners for b in options_corners]
E2 = [(a, b, c) for a in options_corners for b in options_corners for c in options_corners if b!=c]

exit_states.update(E1)
exit_states.update(E2)

E_set_idx = [env.states.index(e) for e in exit_states]
E_set = list(exit_states)
starting_states = list(starting_states)

def _applicable_options(state, options_corners):
    
    options = []

    # if passenger in at pickup location, then applicable option is to go to pickup location
    if state[1] in options_corners:
        options.append(options_corners.index(state[1]))
    # otherwise, the passenger is at the taxi, thus the taxi can go to any location
    elif state[1] == 'TAXI':
        #options.append(options_corners.index(state[2]))
        options.extend(list(range(4)))

    return options  

# Q-Learning
# actions: T, L, R, B, CHANGE, NoOP
actions = [0,1,2,3,4]

# options: 'TL', 'TR', 'BL', 'BR'
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

eps0 = 0.15
eps1 = 0.15

c1 = 1000
c2 = 3000

print(len(starting_states))

for k in tqdm(range(3000)):
    
    env.current_state = E_set[np.random.choice(len(E_set))]

    alpha = c1 / (c1+k+1)

    while env.current_state not in env.terminal_states:
        
        init_state = env.current_state
        c_idx = env.states.index(init_state)

        t = 0
        
        # Select option with Softmax policy
        p_o = _applicable_options(env.current_state, options_corners)
        o = np.random.choice(p_o)
        leaving_state = None
        os = None
       
        alpha_2 = c2 / (c2+k+1)
        
        while True: # while option not finished, follow policy for o until termination
            # Normalize state
            n_s = (0, *env.current_state[0])
            i_n_s = abs_states.index(n_s)
            p_o_actions = _applicable_actions(n_s, abs_states, abs_P)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            
            if np.random.random() < 1 - eps1:
                action = np.nanargmax(Qg[o, i_n_s, :])
            else:
                action = np.random.choice(p_o_actions)

            error = np.mean(np.abs(np.nanmax(Q_flat[E_set_idx, :], axis=1) - np.nanmax(Q[E_set_idx, :], axis=1)))
            errors.append(error)

            if action not in p_actions:
                
                # Update (learn) and select new action for the high-level
                x_s = n_s
                x_ns = _compute_next_abs_state(n_s, action)
                sel = options_terminals.index(x_ns)
                G = np.full(No, -1e10)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (-1 + gamma * G - Qg[:, i_n_s, action])
                # Terminate option
                leaving_state = env.current_state
                action = np.random.choice(p_actions)
                break
            
            t+=1


            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            n_ns = (0 if os[1]==ns[1] else 1, *env.current_state[0])
            i_n_ns = abs_states.index(n_ns)

            # Update Qg accordingly and leave if state is an option's terminal state
            if n_ns[0] == 1:
                leaving_state = env.current_state
                sel = options_terminals.index(n_ns)
                G = np.full(No, -1e10)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * G - Qg[:, i_n_s, action])
                break
            else:
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * np.nanmax(Qg[:, i_n_ns, :], axis=1) - Qg[:, i_n_s, action])
        
        # Update high-level Q function
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)
        
        if leaving_state not in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + np.nanmax(Q[i_ls, :]) - Q[i_is, o])    
        else:
            if leaving_state == ((4, 4), 'TAXI', (4, 4)):
                print('asass')
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + env.r[leaving_state]- Q[i_is, o])
    
    # Derive new softmax policy
    # policy = np.exp(Q[:-len(env.goal_states)]) / np.nansum(np.exp(Q[:-len(env.goal_states)]), axis=1, keepdims=True)


np.savetxt('results/TAXI_H_Q_5x5.txt', Q)
np.save('results/TAXI_options_Q_5x5', Qg)
np.savetxt('results/TAXI_H_errors_5x5.txt', np.array(errors))

