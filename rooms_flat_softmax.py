from rooms_domain import NRoomDomain
import numpy as np
from tqdm import tqdm

dims = (5,5)
room_size = 3
goal_pos = (1,1)
goal_rooms = [(0,0)]

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, True)

# Q-Learning
actions = [0,1,2,3,4,5]

Q = np.full((env.Ns, env.Na), np.NaN)
policy = np.full((env.Ns, env.Na), np.NaN)

for i, x in enumerate(env.interior_states):
    Q[i, env.applicable_actions(x)] = 0
    policy[i, env.applicable_actions(x)] = 1 / len(env.applicable_actions(x))

gamma = 1

for k in tqdm(range(100000)):
    
    env.reset()
    alpha = 0.1 #c / (c + k + 1)

    while env.current_state not in env.terminal_states:
        
        p_actions = env.applicable_actions(env.current_state)
        
        action = np.random.choice(p_actions, p=policy[env.states.index(env.current_state), p_actions])

        os, ns, r = env.apply_action(action)
        
        i_s = env.states.index(os)
        i_ns = env.states.index(ns)
            
        if ns in env.terminal_states:
            Q[i_s, action] = Q[i_s, action] + alpha * (r + gamma * env.r[ns] - Q[i_s, action])
        else:
            Q[i_s, action] = Q[i_s, action] + alpha * (r + gamma * np.nanmax(Q[i_ns, :]) - Q[i_s, action])

    policy = np.exp(Q) / np.nansum(np.exp(Q), keepdims=1, axis=1)
    
np.savetxt('results/Flat_Q_softmax.txt', Q)
np.savetxt('results/Flat_Policy_softmax.txt', policy)
