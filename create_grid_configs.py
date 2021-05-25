from grid_domain.ground_truth import ground_truth
import numpy as np
import pickle
from lmdps.hierarchical_gridworld import _create_TLRBG


def create_config(name, grid_size, goal_rooms, max_n_samples, lambda_):

    execution = {
        'grid_size': grid_size,
        'r_dim': 5,
        'goal_pos': (2, 3),
        'goal_rooms': goal_rooms,
        'lambda': lambda_,
        'max_n_samples': max_n_samples,
        'goal_reward': 0,
        'non_goal_reward': -1
    }

    [T, L, R, B, G], P = _create_TLRBG(5, (2, 3), t=execution['lambda'])
    opt_Z_i = np.array([T, L, R, B, G]).reshape(5, 30)

    Z_true = ground_truth(**execution)

    execution['Z_true'] = Z_true
    execution['Z_opt_i'] = opt_Z_i

    pickle.dump(execution, open(f'configs_grid/{name}.dict', 'wb'))


if __name__ == '__main__':

    create_config('2_2_1', (2, 2), [(1, 0)], 10000, 1)
    create_config('3_3_1', (3, 3), [(2, 0)], 10000, 1)
    create_config('4_4_1', (4, 4), [(3, 0)], 10000, 1)
    create_config('5_5_1', (5, 5), [(4, 0)], 10000, 1)
    create_config('6_6_1', (6, 6), [(5, 0)], 10000, 1)

    create_config('2_2_10', (2, 2), [(1, 0)], 10000, 10)
    create_config('3_3_10', (3, 3), [(2, 0)], 10000, 10)
    create_config('4_4_10', (4, 4), [(3, 0)], 10000, 10)
    create_config('5_5_10', (5, 5), [(4, 0)], 10000, 10)
    create_config('6_6_10', (6, 6), [(5, 0)], 10000, 10)