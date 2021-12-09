from tqdm import tqdm
import numpy as np

def QLearning(init_states, S, T, P, R, Na, applicable_actions, compute_next_state, c, eps):

    F = S+T

    Q = np.full((len(S), Na), np.NaN, dtype=np.float64)

    for i, x in enumerate(S):
        Q[i, applicable_actions(x, F, P)] = 0

    for k in tqdm(range(5000)):
        
        state = init_states[np.random.choice(range(len(init_states)))]
        alpha = c / (c + k + 1)

        while state not in T:
            
            p_actions = applicable_actions(state, F, P)
            
            if np.random.random() < 1-eps:
                action = np.nanargmax(Q[S.index(state), :])
            else:
                action = np.random.choice(p_actions)

            next_state, r = compute_next_state(state, action), R[S.index(state)]
            
            i_s = F.index(state)
            i_ns = F.index(next_state)
                
            if next_state in T:
                Q[i_s, action] = Q[i_s, action] + alpha * (r + R[F.index(next_state)] - Q[i_s, action])
            else:
                Q[i_s, action] = Q[i_s, action] + alpha * (r + np.nanmax(Q[i_ns, :]) - Q[i_s, action])
            
            state = next_state
    

    return Q