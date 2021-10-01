from rooms_domain import NRoomDomain
import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from tqdm import tqdm

dims = (2,2)
room_size = 3
goal_pos = (1,1)
goal_rooms = [(0,0)]


env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, high_level=True)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)], high_level=False)
abs_states = abs_room.states

def option_is_applicable(room, room_size, option, states, goal_pos):
    X, Y = room[0]*room_size, room[1]*room_size
    goal_pos = X+goal_pos[0], Y+goal_pos[1]
    ts = [(0, X-1, (Y+room_size)//2), (0, (X+room_size)//2, Y-1), (0, (X+room_size)//2, Y+room_size), (0, X+room_size, (Y+room_size)//2), (1, *goal_pos)]
    return ts[option] in states
   


def _applicable_options(room, room_size, states, goal_pos):
    return [o for o in range(5) if option_is_applicable(room, room_size, o, states, goal_pos)]

# Q-Learning
actions = [0,1,2,3,4]

o_terminals = [(0, -1, room_size//2), (0, room_size//2, -1), (0, room_size//2, room_size), (0, room_size, room_size//2), (1, *goal_pos)]
No = len(o_terminals)

# High-level Q
Q = np.full((len(env.states), No), np.NaN)
policy = np.full((len(env.states), No), np.NaN)
for i, x in enumerate(env.states):
    if x in env.terminal_states:
        ntn = env.P[:, env.states.index(x)].nonzero()[0][0]
        stn = env.states[ntn]
        p_options = _applicable_options(_id_room(stn, room_size), room_size, env.states, goal_pos)
        Q[i, p_options] = 0
        policy[i, p_options] = 1 / len(p_options)
        continue
    p_options = _applicable_options(_id_room(x, room_size), room_size, env.states, goal_pos)
    Q[i, p_options] = 0
    policy[i, p_options] = 1 / len(p_options)

# Q_o for options
Qg = np.full((No, len(abs_room.interior_states), env.Na), np.NaN)
O_policies = np.full((No, len(abs_room.interior_states), env.Na), np.NaN)
for i, x in enumerate(abs_room.interior_states):
    Qg[:, i, abs_room.applicable_actions(x)] = 0
    O_policies[:, i, abs_room.applicable_actions(x)] = 1 / len(abs_room.applicable_actions(x))

Q_flat = np.loadtxt('results/txt')

gamma = 1

for k in tqdm(range(100000)):
    
    env.reset(1)
    alpha = 0.1
    alpha_2 = 0.3     

    while env.current_state not in env.goal_states:
        
        init_state = env.current_state

        t = 0

        # Select option from Softmax option
        
        p_options = _applicable_options(_id_room(env.current_state, room_size), room_size, env.states, goal_pos)
        
        o = np.random.choice(p_options, p=policy[env.states.index(env.current_state), p_options])

        # Retrieve option's policy
        policy_o = O_policies[o, :, :]

        leaving_state = None

        os = None
        room = None
        
        while True: # while option not finished, follow policy for o until termination
            # Normalize state
            n_s = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)
            i_n_s = abs_states.index(n_s)
            p_o_actions = abs_room.applicable_actions(n_s)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            
            action = np.nanargmax(Qg[o, i_n_s, :])
            
            t+=1

            if action not in p_actions:
                
                # Update (learn) and select new action for the high-level
                abs_room.current_state = n_s
                x_os, x_ns, x_r = abs_room.apply_action(action)
                sel = o_terminals.index(x_ns)
                G = np.full(No, abs_room.penalty)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (-1 + gamma * G - Qg[:, i_n_s, action])
                # Terminate option
                leaving_state = env.current_state
                break

            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            n_ns = _normalize_cell(ns, _id_room(os, room_size), room_size)
            i_n_ns = abs_states.index(n_ns)

            # Update Qg accordingly and leave if state is an option's terminal state
            if n_ns in o_terminals:
                leaving_state = ns
                sel = o_terminals.index(n_ns)
                G = np.full(No, env.penalty)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * G - Qg[:, i_n_s, action])
                break
            else:
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * np.nanmax(Qg[:, i_n_ns, :], axis=1) - Qg[:, i_n_s, action])

        policy_o = np.exp(Qg) / np.nansum(np.exp(Qg), axis=2, keepdims=1)
        
        # Update high-level Q function
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)
        
        if leaving_state not in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + np.nanmax(Q[i_ls, :]) - Q[i_is, o])
            
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (-t + env.r[leaving_state] - Q[i_is, o])
            break

    # Derive new high level Softmax policy
    policy = np.exp(Q) / np.nansum(np.exp(Q), axis=1, keepdims=1)

np.savetxt('results/H_Q_softmax.txt', Q)
np.savetxt('results/H_Policy_softmax.txt', policy)
np.save('results/options_Q_softmax', Qg)
