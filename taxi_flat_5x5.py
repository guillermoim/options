from taxi_domain import TaxiDomain
import numpy as np
from tqdm import tqdm

env = TaxiDomain(5)

# Q-Learning
actions = list(range(env.Na))

Q = np.full((env.Ns, env.Na), np.NaN)
policy = np.full((env.Ns, env.Na), np.NaN)

for i, x in enumerate(env.states):
    if x not in env.terminal_states:
        Q[i, env.applicable_actions(x)] = 0
        policy[i, env.applicable_actions(x)] = 1 / len(env.applicable_actions(x))

Q_ref = np.loadtxt('results/TAXI_Flat_Q_5x5.txt')
errors = []

gamma = 1

c = 10000
eps = 0.2

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
    
np.savetxt('results/TAXI_Flat_Q_5x5.txt', Q)
np.savetxt('results/TAXI_Flat_errors_5x5.txt', errors)
