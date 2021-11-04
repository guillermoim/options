from rooms_domain import NRoomDomain
import numpy as np
from partitions_tracker import _id_room, _normalize_cell
from tqdm import tqdm
from itertools import product

dims = (8,8)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,7)]


env = NRoomDomain(dims, room_size, goal_pos, goal_rooms, high_level=True)
abs_room = NRoomDomain((1,1), room_size, goal_pos, [(0,0)], high_level=False)
abs_states = abs_room.states

def option_is_applicable(room, room_size, option, states, goal_pos):
    X, Y = room[0]*room_size, room[1]*room_size
    goal_pos = X+goal_pos[0], Y+goal_pos[1]
    ts = [(0, X-1, Y+room_size//2), (0, X+room_size//2, Y-1), (0, X+room_size//2, Y+room_size), (0, X+room_size, Y+room_size//2), (1, *goal_pos)]
    return ts[option] in states
   
def _applicable_options(room, room_size, states, goal_pos):
    return [o for o in range(5) if option_is_applicable(room, room_size, o, states, goal_pos)]

def _get_exit_states(dims, room_size, states):
    
    exit_states = []
    rooms = set([room for room in product(range(dims[0]), range(dims[1]))])
   
    for room in rooms:
        X, Y = room[0]*room_size, room[1]*room_size
        ts = [(0, X-1, Y+room_size//2), (0, X+room_size//2, Y-1), (0, X+room_size//2, Y+room_size), (0, X+room_size, Y+room_size//2)]
        local = []
        for t in ts:
            if t in states:
                exit_states.append(t)
                local.append(t)
    return exit_states

E_set = _get_exit_states(dims, room_size, env.states)
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
V_ref = np.nanmax(Q_flat[E_set_idx, :], axis=1)

gamma = 1

errors = []

eps0 = 0.3
eps1 = 0.3

c1 = 1000
c2 = 3000


for k in tqdm(range(5000)):
    
    # Reset the option at one of the exit states
    env.current_state = E_set[np.random.choice(len(E_set))]
    alpha = c1 / (c1+k+1)

    while env.current_state not in env.goal_states:
        
        init_state = env.current_state
        current_idx = env.states.index(init_state)

        acc_reward = 0

        # Select option from Softmax option
        possible_ops = _applicable_options(_id_room(env.current_state, room_size), room_size, env.states, goal_pos)

        if np.random.random() < 1-eps0:
            o = np.nanargmax(Q[current_idx, :])
        else:
            o = np.random.choice(possible_ops)

        # Retrieve option's policy
        policy_o = O_policies[o, :, :]

        leaving_state = None

        os = None

        alpha_2 = c2 / (c2+k+1)

        while True: # while option not finished, follow policy for o until termination
                        
            # Normalize state
            proj_state = _normalize_cell(env.current_state, _id_room(env.current_state, room_size), room_size)
            idx_proj_state = abs_states.index(proj_state)
            poss_actions_in_option = abs_room.applicable_actions(proj_state)
            
            # I take the action for the 'projected' state, sampling from Softmax policy
            p_actions = env.applicable_actions(env.current_state)
            
            if np.random.random() < 1-eps1:
                action = np.nanargmax(Qg[o, idx_proj_state, :])
            else:
                action = np.random.choice(poss_actions_in_option)
            
            acc_reward += 1

            V = np.nanmax(Q[E_set_idx, :], axis=1)

            error = np.mean(np.abs(V_ref - V))
            errors.append(error)

            if action not in p_actions:
                # Update (learn) and select new action for the high-level
                abs_room.current_state = proj_state
                x_os, x_ns, x_r = abs_room.apply_action(action)
                sel = o_terminals.index(x_ns)
                G = np.full(No, env.penalty)
                G[sel] = 0
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (-1 + gamma * G - Qg[:, idx_proj_state, action])
                # Terminate option
                leaving_state = env.current_state
                action = np.random.choice(p_actions)
                eps1*=0.99

            # Apply action and project new state
            old_state, next_state, r = env.apply_action(action)
            proj_next_state = _normalize_cell(next_state, _id_room(old_state, room_size), room_size)
            idx_proj_next_state = abs_states.index(proj_next_state)

            # Update Qg accordingly and leave if state is an option's terminal state
            if proj_next_state in o_terminals:
                leaving_state = next_state
                sel = o_terminals.index(proj_next_state)
                G = np.full(No, env.penalty)
                G[sel] = 0
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (r + gamma * G - Qg[:, idx_proj_state, action])
                eps1*=0.99
                break
            else:
                Qg[:, idx_proj_state, action] = Qg[:, idx_proj_state, action] + alpha_2 * (r + gamma * np.nanmax(Qg[:, idx_proj_next_state, :], axis=1) - Qg[:, idx_proj_state, action])
        
        # Update high-level Q function
        i_is = env.states.index(init_state)
        i_ls = env.states.index(leaving_state)

        if leaving_state not in env.goal_states:
            Q[i_is, o] = Q[i_is, o] + alpha * (-acc_reward + np.nanmax(Q[i_ls, :]) - Q[i_is, o])
        else:
            Q[i_is, o] = Q[i_is, o] + alpha * (-acc_reward + env.r[leaving_state] - Q[i_is, o])

    # Derive new high level Softmax policy
    eps0*=0.99

np.savetxt('results/rooms_H_Q_8x8.txt', Q)
np.save('results/rooms_options_Q_8x8', Qg)
np.savetxt('results/rooms_H_errors_8x8.txt', np.array(errors))
