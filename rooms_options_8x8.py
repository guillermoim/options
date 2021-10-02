from rooms_domain import NRoomDomain
import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from tqdm import tqdm
from itertools import product

dims = (5,5)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,7)]


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

def _get_exit_states(dims, room_size, states, goal_pos):
    
    exit_states = []
    rooms = set([room for room in product(range(dims[0]), range(dims[1]))])
   
    for room in rooms:
        X, Y = room[0]*room_size, room[1]*room_size
        ts = [(0, X-1, (Y+room_size)//2), (0, (X+room_size)//2, Y-1), (0, (X+room_size)//2, Y+room_size), (0, X+room_size, (Y+room_size)//2)]
        local = []
        for t in ts:
            if t is (1,1,1):
                print(t)
            if t in states:
                exit_states.append(t)
                local.append(t)
        #print(room, local)
    return exit_states

E_set = _get_exit_states(dims, room_size, env.states, (1,1))
E_set_idx = [env.states.index(x) for x in E_set]

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

Q_flat = np.loadtxt('results/rooms_Flat_Q_8x8.txt')

gamma = 1

errors = []

eps0 = 0.1
eps1 = 0.4

c1 = 1000000
c2 = 50000

for k in tqdm(range(10000)):
    
    env.reset()

    lenghts_episode = 0

    while env.current_state not in env.goal_states and lenghts_episode < 200:
        
        init_state = env.current_state

        t = 0

        # Select option from Softmax option
        
        p_options = _applicable_options(_id_room(env.current_state, room_size), room_size, env.states, goal_pos)
        
        o = np.random.choice(p_options)

        # Retrieve option's policy
        policy_o = O_policies[o, :, :]

        leaving_state = None

        os = None
        room = None

        while True: # while option not finished, follow policy for o until termination
            lenghts_episode+=1
            # Normalize state
            n_s = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)
            i_n_s = abs_states.index(n_s)
            p_o_actions = abs_room.applicable_actions(n_s)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            
                
            action = np.nanargmax(Qg[o, i_n_s, :])
            
            t+=1

            error = np.mean(np.abs(np.nanmax(Q_flat[E_set_idx, :], axis=1) - np.nanmax(Q[E_set_idx, :], axis=1)))
            errors.append(error)

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
                eps1 = eps1*0.99

                break

            # Apply action and project new state
            os, ns, r = env.apply_action(action)
            n_ns = _normalize_cell(ns, _id_room(os, room_size), room_size)
            i_n_ns = abs_states.index(n_ns)

            # Update Qg accordingly and leave if state is an option's terminal state
            alpha_2 = c2 / (c2+k)
            if n_ns in o_terminals:
                leaving_state = ns
                sel = o_terminals.index(n_ns)
                G = np.full(No, env.penalty)
                G[sel] = 0
                Qg[:, i_n_s, action] = Qg[:, i_n_s, action] + alpha_2 * (r + gamma * G - Qg[:, i_n_s, action])
                eps1 = eps1*0.99
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
        
    # Derive new high level Softmax policy
    eps0 = eps0*0.99

np.savetxt('results/rooms_H_Q_8x8.txt', Q)
np.save('results/rooms_options_Q_8x8', Qg)
np.savetxt('results/rooms_H_errors_8x8.txt', np.array(errors))
