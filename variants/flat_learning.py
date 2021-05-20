import sys

sys.path.insert(0, '../..')

from .error_metric import error_metric
from super_grid import SuperGrid, GridSubtasks
from learning import ZLearning
from tqdm import tqdm

def flat_Z_learning(c, iw=True, **kargs):
    grid_size = kargs['grid_size']
    r_dim = kargs['r_dim']
    goal_pos = kargs['goal_pos']
    goal_rooms = kargs['goal_rooms']
    MAX_N_SAMPLES = kargs['max_n_samples']
    Z_true = kargs['Z_true']
    lambda_ = kargs['lambda']
    goal_reward = kargs['goal_reward']
    non_goal_reward = kargs['non_goal_reward']

    grid = SuperGrid(grid_size, r_dim, goal_pos, goal_rooms, non_goal_reward, goal_reward)
    Z = ZLearning(grid.states, grid.P, c, tau=lambda_)

    n_samples = 0

    pbar = tqdm(total=MAX_N_SAMPLES)

    errors = []

    while n_samples < MAX_N_SAMPLES:

        terminate_episode = False

        grid.reset()

        Z.update_alpha()

        pbar.set_description(f'c={Z.c} alpha={Z.get_alpha():.4f}')

        while not terminate_episode:
            # Get a sample in the super-grid
            # Update the high-level function
            (state, reward, next_state) = grid.sample(Z.Z)

            w_a = grid.get_importance_weight(state, next_state) if iw else 1

            Z.update(state, next_state, reward, weight=w_a)

            # If the agent has reached any of the exit sates
            grid.update_controls(Z.Z)

            terminate_episode = state in grid.terminal
            n_samples += 1
            pbar.update(1)

            error = error_metric(Z_true, Z.Z, len(grid.non_terminal))
            errors.append(error)

    return Z, errors
