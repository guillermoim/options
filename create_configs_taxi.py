import pickle
import numpy as np
from mdps.taxi_mdps import create_flat_mdp, create_taxi_room
from utilities.utils_taxi import get_exit_states, applicable_actions, compute_next_state
from q_learning import QLearning

def create_config(name, N_MAX_SAMPLES, DIM):
    
    N_MAX_SAMPLES = N_MAX_SAMPLES
    DIM = DIM

    # Number of actions and number of options per subtask
    Na = 6 # T, L, R, B, CHANGE, NoOp
    No = 4 # One for each corner

    sample_states, S, T, P, R = create_flat_mdp(dim=DIM, terminals_non_goals=False)
    F = S + T

    P_, abs_states, R_ = create_taxi_room(DIM)
    abs_int_states = abs_states[:-No]
    options_terminals = abs_states[-No:]

    E_set = get_exit_states(DIM)
    E_set_idx = [F.index(x) for x in E_set]

    Q_flat = QLearning(sample_states, S, T, P, R, Na, applicable_actions, compute_next_state, 10000, 0.3)
    V_flat = np.nanmax(Q_flat[E_set_idx, :], axis=1)

    CONFIG = {}

    CONFIG['N_MAX_SAMPLES'] = N_MAX_SAMPLES
    CONFIG['DIM'] = DIM
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
    create_config('taxi_5', 20000, 5)
    create_config('taxi_10', 20000, 10)

