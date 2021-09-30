from taxi_domain import TaxiDomain, _create_taxi_room
import numpy as np
from tqdm import tqdm

## Auxiliary
def _compute_next_state(corners, state, action):
    next_state = None
    if action == 0: # TOP
        next_state=(state[0], state[1]-1, state[2])
    elif action==1: # LEFT
        next_state=(state[0], state[1], state[2]-1)
    elif action==2: # RIGHT
        next_state=(state[0], state[1]-1, state[2]+1)
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

## Auxiliary

dim = 5

env = TaxiDomain(dim)

No = 4
names = ['TL', 'TR', 'BL', 'BR']
options_corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]


def _applicable_options(state):
    options = []

    if state[1] == 'TAXI':
        options.append(options_corners.index(state[2]))
    elif state[1] not in ('Forbidden', 'D'):
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
        xs = _compute_next_state(options_corners, x, a)
        if _is_legal_move(abs_states, abs_P, x, xs):
            x_actions.append(a)
    Qg[:, i, x_actions] = 0
    O_policies[:, i, x_actions] = 1 / len(x_actions)

#o_terminals = [(0, -1, 2), (0, 2, -1), (0, 2, 5), (0, 5, 2), (1, 2, 3)]

gamma = 1

for k in tqdm(range(10000)):
    
    env.reset()
    alpha = 0.2     
    alpha_2 = 0.2
    
    while env.current_state not in env.terminal_states:
        
        acc_reward_option = 0
        init_state = env.current_state

        # Select option from Softmax option
        #if env.current_state in env.terminal_states:
        #    print(k, 'A', env.current_state)
        #    ntn = env.P[:, env.states.index(env.current_state)].nonzero()[0][0]
        #    p_options = _applicable_options(_id_room(env.states[ntn]), dims, goal_rooms)
        #    env.current_state = env.states[ntn]
        #    print(k, 'B', env.current_state)

        #    acc_reward_option+=-1
        
        p_options = _applicable_options(env.current_state)

        o = np.random.choice(p_options, p=policy[env.states.index(env.current_state), p_options])

        # Retrieve option's policy
        policy_o = O_policies[o, :, :]

        leaving_state = None

        t = 0

        os = None
        
        while True: # while option not finished, follow policy for o until termination
            # Normalize state
            n_s = 0, *env.current_state[0]
            i_n_s = abs_states.index(n_s)

            print(k, 'inside', env.current_state)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            print(env.current_state, p_actions, O_policies[o, i_n_s, p_actions])
            action = np.random.choice(p_actions, p=O_policies[o, i_n_s, p_actions])

            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            n_ns = 0 if ns[1] == os[1] else 1, ns[0]
            i_n_ns = abs_states.index(n_ns)
            
            # Accumulate reward for the executing option
            acc_reward_option += gamma**t * r
            
            # Update Qg accordingly and leave if state is an option's terminal state
            if n_ns[0] != n_s[0]:
                leaving_state = ns
                sel = options_corners.index((n_ns[1], n_ns[2]))
                G = np.full(No, env.penalty)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * G - Qg[:, i_n_s, action])
                break
            else:
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * np.nanmax(Qg[:, i_n_ns, :], axis=1) - Qg[:, i_n_s, action])
            
            t+=1

        # New estimate of the softmax policy for the options
        O_policies = np.exp(Qg) / np.nansum(np.exp(Qg), keepdims=1, axis=2)

        # Update high-level Q function
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)
        
        if leaving_state not in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward_option + gamma**t * np.nanmax(Q[i_ls, :]) - Q[i_is, o])
            if leaving_state in env.terminal_states:
                break
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (acc_reward_option + gamma**t * env.r[leaving_state] - Q[i_is, o])
            break
    
    # Derive new high level Softmax policy
    policy = np.exp(Q) / np.nansum(np.exp(Q), axis=1, keepdims=1)

np.savetxt('results/Taxi_H_Q_softmax.txt', Q)
np.savetxt('results/Taxi_H_Policy_softmax.txt', policy)
np.save('results/Taxi_options_Q', Qg)
