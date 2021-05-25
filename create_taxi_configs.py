import pickle

from taxi_domain.taxi import create_flat_taxi_MDP
from taxi_domain.ground_truth import ground_truth, ground_truth_subtasks


def create_config(name, dim, max_n_samples, lambda_=1):

    execution = {
        'dim': dim,
        'lambda_': lambda_,
        'max_n_samples': max_n_samples,
        'goal_reward': 0,
        'non_goal_reward': -1
    }

    G, P, states, init_states, non_goal_states, goal_states = create_flat_taxi_MDP(dim, lambda_)
    Z_true = ground_truth(dim, lambda_)
    Z_i_opt = ground_truth_subtasks(dim, lambda_)

    execution['P'] = P
    execution['states'] = states
    execution['init_states'] = init_states
    execution['goal_states'] = goal_states
    execution['Z_true'] = Z_true
    execution['Z_i_opt'] = Z_i_opt
    execution['non_goal_states'] = non_goal_states

    pickle.dump(execution, open(f'configs_taxi/{name}.dict', 'wb'))


if __name__ == '__main__':

    create_config('taxi_5_1', 5, 5000, 1)
    create_config('taxi_10_1', 10, 15000, 1)
    create_config('taxi_15_1', 15, 30000, 1)

    create_config('taxi_5_10', 5, 5000, 10)
    create_config('taxi_10_10', 10, 15000, 10)
    create_config('taxi_15_10', 15, 30000, 10)
