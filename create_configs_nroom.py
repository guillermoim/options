import pickle
import numpy as np
from mdps.nroom_mdp import create_flat_mdp, create_room_hierarchical
from utilities.utils_nroom import get_exit_states, applicable_actions, compute_next_state
from q_learning import QLearning

def create_config(name, N_MAX_SAMPLES, DIMS, ROOM_SIZE, GOAL_POS, GOAL_ROOMS):
    
    N_MAX_SAMPLES = N_MAX_SAMPLES
    DIMS = DIMS
    ROOM_SIZE = ROOM_SIZE
    GOAL_POS = GOAL_POS
    GOAL_ROOMS = GOAL_ROOMS

    # Number of actions and number of options per subtask
    Na = 6 # T, L, R, B, G, NoOp
    No = 5 # T, L, R, B, G

    sample_states, S, T, P, R = create_flat_mdp(DIMS, ROOM_SIZE, GOAL_POS, GOAL_ROOMS, non_goal_terminals=False)
    F = S + T

    P_, abs_states, R_ = create_room_hierarchical(ROOM_SIZE, GOAL_POS)
    abs_int_states = abs_states[:-No]
    options_terminals = abs_states[-No:]

    E_set = get_exit_states(DIMS, ROOM_SIZE, F)
    E_set_idx = [F.index(x) for x in E_set]

    Q_flat = QLearning(sample_states, S, T, P, R, Na, applicable_actions, compute_next_state, 10000, 0.3)
    V_flat = np.nanmax(Q_flat[E_set_idx, :], axis=1)

    CONFIG = {}

    CONFIG['N_MAX_SAMPLES'] = N_MAX_SAMPLES
    CONFIG['DIMS'] = DIMS
    CONFIG['ROOM_SIZE'] = ROOM_SIZE
    CONFIG['GOAL_POS'] = GOAL_POS
    CONFIG['GOAL_ROOMS'] = GOAL_ROOMS
    CONFIG['Na'] = Na
    CONFIG['No'] = No
    CONFIG['sample_states'] = sample_states
    CONFIG['S'] = S
    CONFIG['T'] = T
    CONFIG['P'] = P
    CONFIG['R'] = R
    CONFIG['F'] = F
    CONFIG['P_'] = P_
    CONFIG['abs_states'] = abs_states
    CONFIG['R_'] = R_
    CONFIG['abs_int_states'] = abs_int_states
    CONFIG['options_terminals'] = options_terminals
    CONFIG['E_set'] = E_set
    CONFIG['E_set_idx'] = E_set_idx
    CONFIG['V_flat'] = V_flat

    pickle.dump(CONFIG, open(f'configs/{name}.pkl', 'wb'))


if __name__ == '__main__':
    create_config('nrooms_3x3', 20000, (3,3), 5, (2,3), [(0,0)])
    create_config('nrooms_5x5', 20000, (5,5), 3, (1,1), [(0,0)])
    create_config('nrooms_8x8', 50000, (8,8), 5, (2,3), [(0,0)])


