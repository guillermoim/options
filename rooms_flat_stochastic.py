from NRoomDomainStochastic import NRoomDomainStochastic
import numpy as np
from tqdm import tqdm


dims = (3,3)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,0)]

env = NRoomDomainStochastic(dims, room_size, goal_pos, goal_rooms, path='PROV_SOLUTIONS/optimal_policy_3x3.npy')

# Q-Learning
actions = [0,1,2,3,4,5]

Q = np.full((env.Ns, env.Na), np.NaN)
policy = np.full((env.Ns, env.Na), np.NaN)

for i, x in enumerate(env.interior_states):
    Q[i, env.applicable_actions(x)] = 0
    policy[i, env.applicable_actions(x)] = 1 / len(env.applicable_actions(x))

gamma = 1

V_ref = np.loadtxt('PROV_SOLUTIONS/optimal_VF_3x3.txt')

c = 3000
eps = 0.2

errors = []

for k in tqdm(range(50000)):
    
    env.reset()
    alpha = c / (c + k + 1)

    while env.current_state not in env.terminal_states:
        
        p_actions = env.applicable_actions(env.current_state)
        
        if np.random.random() < 1-eps:
            action = np.nanargmax(Q[env.states.index(env.current_state), :])
        else:
            action = np.random.choice(p_actions)
        
        os, ns, r = env.apply_action(action)

        i_s = env.states.index(os)
        i_ns = env.states.index(ns)
    
            
        if ns in env.terminal_states:
            Q[i_s, action] = Q[i_s, action] + alpha * (r + gamma * env.r[ns] - Q[i_s, action])
        else:
            Q[i_s, action] = Q[i_s, action] + alpha * (r + gamma * np.nanmax(Q[i_ns, :]) - Q[i_s, action])
    
        error = np.mean(V_ref - np.nanmax(Q[:-len(env.terminal_states), :], axis=1))
        errors.append(error)
    
np.savetxt('PROV_SOLUTIONS/Q_flat_stochastic_3x3.txt', Q)
np.savetxt('PROV_SOLUTIONS/erro_flat_stochastic_3x3.txt.txt', errors)