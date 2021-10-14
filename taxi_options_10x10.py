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


## Auxiliary

dim = 5

env = TaxiDomain(dim)

No = 4
names = ['TL', 'TR', 'BL', 'BR']
options_corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]
options_terminals = [(1, 0,0), (1, dim-1,0), (1, 0,dim-1), (1, dim-1,dim-1)]

exit_states = set()
for pair in product(options_corners, options_corners):
    exit_states.add((pair[0], 'TAXI', pair[1]))

for c0 in options_corners:
    for c1 in options_corners:
        if c0!=c1:
            exit_states.add((c0, c0, c1))

E_set = [env.states.index(e) for e in exit_states]


def _applicable_options(state):
    options = []

    if state[1] == 'TAXI':
        options.append(options_corners.index(state[2]))
        for c in options_corners:
            if c != state[2]:
                options.append(options_corners.index(c))

    elif state[1] != 'D':
        options.append(options_corners.index(state[1]))

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
        Q[i, _applicable_options(x)] = 0
        policy[i, _applicable_options(x)] = 1 / len(_applicable_options(x))

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

#o_terminals = [(0, -1, 2), (0, 2, -1), (0, 2, 5), (0, 5, 2), (1, 2, 3)]

Q_flat = np.loadtxt('results/TAXI_Flat_Q_10x10.txt')


gamma = 1

errors = []

eps0 = 0.15
eps1 = 0.15

c1 = 1000
c2 = 3000

for k in tqdm(range(10000)):
    
    env.current_state = E_set[np.random.choice(len(E_set))]
    alpha = c1 / (c1+k+1)

    while env.current_state not in env.goal_states:
        
        init_state = env.current_state

        t = 0

        # Select option from Softmax option
        
        p_options = _applicable_options(env.current_state)
        
        if np.random.random() < 1-eps0:
            o = np.nanargmax(Q[env.states.index(init_state), :])
        else:
            o = np.random.choice(p_options)

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
            
            if np.random.random() < 1-eps1:
                action = np.nanargmax(Qg[o, i_n_s, :])
            else:
                action = np.random.choice(p_o_actions)
            t+=1

            error = np.mean(np.abs(np.nanmax(Q_flat[exit_states, :], axis=1) - np.nanmax(Q[exit_states, :], axis=1)))
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
                break

            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            n_ns = (0 if os[1]==ns[1] else 1, *env.current_state[0])
            i_n_ns = abs_states.index(n_ns)

            # Update Qg accordingly and leave if state is an option's terminal state
            alpha_2 = c2 / (c2+k)
            if n_ns[0] == 1:
                leaving_state = ns
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
        
        alpha = c1 / (c1+k)
        if leaving_state not in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + np.nanmax(Q[i_ls, :]) - Q[i_is, o])    
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + env.r[leaving_state] - Q[i_is, o])
        

np.savetxt('results/TAXI_H_Q_10x0.txt', Q)
np.save('results/TAXI_options_Q_10x10', Qg)
np.savetxt('results/TAXI_H_errors_10x10.txt', np.array(errors))
