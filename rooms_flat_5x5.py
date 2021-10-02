from rooms_domain import NRoomDomain
import numpy as np
from tqdm import tqdm

dims = (5,5)
room_size = 3
goal_pos = (1,1)
goal_rooms = [(0,4)]

env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, True)

# Q-Learning
actions = [0,1,2,3,4,5]

Q = np.full((env.Ns, env.Na), np.NaN)
policy = np.full((env.Ns, env.Na), np.NaN)

for i, x in enumerate(env.interior_states):
    Q[i, env.applicable_actions(x)] = 0
    policy[i, env.applicable_actions(x)] = 1 / len(env.applicable_actions(x))


Q_ref = np.loadtxt('results/rooms_Flat_Q_5x5.txt')
errors = []

gamma = 1

c = 10000
eps = 0.3

for k in tqdm(range(10000)):
    
    env.reset()
    alpha = c / (c + k + 1)

    while env.current_state not in env.goal_states:
        
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
        
        error = np.mean(np.abs(np.nanmax(Q_ref[:-len(env.terminal_states), :], axis=1) - np.nanmax(Q[:-len(env.terminal_states), :], axis=1)))
        errors.append(error)
        
    eps = eps * .99
    
np.savetxt('results/rooms_Flat_Q_5x5.txt', Q)
np.savetxt('results/rooms_Flat_errors_5x5.txt', errors)
