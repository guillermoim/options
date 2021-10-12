from NRoomDomainStochastic import NRoomDomainStochastic
import numpy as np
from tqdm import tqdm


dims = (3,3)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,0)]

env = NRoomDomainStochastic(dims, room_size, goal_pos, goal_rooms, path='solutions_lmdps/policy_3x3_goal@0-0_rooms5x5.txt')

# Q-Learning
actions = [0,1,2,3,4,5]

Q = np.full((env.Ns, env.Na), np.NaN)
policy = np.full((env.Ns, env.Na), np.NaN)

for i, x in enumerate(env.interior_states):
    Q[i, env.applicable_actions(x)] = 0
    policy[i, env.applicable_actions(x)] = 1 / len(env.applicable_actions(x))

gamma = 1

#Q_ref = np.loadtxt('results/rooms_Flat_Q_3x3.txt')
#errors = []

c = 30000
eps = 0.5


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
    
        #error = np.mean(np.abs(np.nanmax(Q_ref[:-len(env.terminal_states), :], axis=1) - np.nanmax(Q[:-len(env.terminal_states), :], axis=1)))
        #errors.append(error)
    
np.savetxt('Q_stochastic_3x3.txt', Q)