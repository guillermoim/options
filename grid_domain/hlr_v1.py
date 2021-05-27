import sys

sys.path.insert(0, '../..')

from super_grid import SuperGrid, GridSubtasks
from hierarchical_learning import HierarchicalLearning
import numpy as np
from tqdm import tqdm
from super_grid import _id_room
from .error_metric import error_metric

def __init__():
    pass

def hlr_v1(c0, c1, **kargs):

    grid_size = kargs['grid_size']
    r_dim = kargs['r_dim']
    goal_pos = kargs['goal_pos']
    goal_rooms = kargs['goal_rooms']
    MAX_N_SAMPLES = kargs['max_n_samples']
    Z_true = kargs['Z_true']
    Z_opt_i = kargs['Z_opt_i']
    lambda_ = kargs['lambda']
    goal_reward = kargs['goal_reward']
    non_goal_reward = kargs['non_goal_reward']

    t_map = {(0, -1, r_dim // 2): (0, 0, r_dim // 2),
             (0, r_dim // 2, -1): (0, r_dim // 2, 0),
             (0, r_dim//2, r_dim): (0, r_dim//2, r_dim - 1),
             (0, r_dim, r_dim//2): (0, r_dim - 1, r_dim//2),
             (1, *goal_pos): (0, *goal_pos)}

    grid = SuperGrid(grid_size, r_dim, goal_pos, goal_rooms, non_goal_reward, goal_reward)
    subtasks = GridSubtasks(r_dim, non_goal_reward, goal_reward, c1, lambda_, t_map)

    HL = HierarchicalLearning(c0, grid_size, grid.states, grid.terminal, goal_pos, goal_rooms, r_dim)

    errors = []
    errors_i = []

    n_samples = 0
    n_episodes = 0
    pbar = tqdm(total=MAX_N_SAMPLES)

    subtasks.update_alpha()

    while n_samples < MAX_N_SAMPLES:

        terminate_episode = False
        n_episodes += 1

        init_state = grid.reset()
        current_room = _id_room(init_state)
        old_room = _id_room(init_state)

        HL.update_alpha()

        pbar.set_description(f'alphas HL={HL.alpha:.4f} LL={subtasks.alpha:.4f} episodes={n_episodes}')

        while not terminate_episode:
            # Get a sample in the super-grid
            # Update the high-level function

            (state, reward, next_state) = grid.sample(HL.get_implicit_Z(current_room, subtasks.Z_i))

            # Update the subtasks
            if state in HL.H[old_room]['exit_states']:
                subtasks.update(state, state, old_room)
                subtasks.update_alpha()
            else:
                subtasks.update(state, next_state, current_room)

            # Then I need to update the high-level Z function with the learnt subtasks
            if HL.is_exit(state) and state not in grid.terminal:
                HL.update_exit_state(state, subtasks.Z_i, subtasks.abstract_states)
                if next_state not in grid.terminal:
                    current_room = _id_room(next_state)
                    old_room = _id_room(state)

            terminate_episode = state in grid.terminal

            n_samples += 1
            pbar.update(1)

            explicit_Z = HL.get_explicit_Z(subtasks.Z_i)

            error = error_metric(Z_true, explicit_Z, len(grid.non_terminal))
            errors.append(error)

            error_i = np.abs(np.log(Z_opt_i[:, :r_dim**2]) - np.log(subtasks.Z_i[:, :r_dim**2])).mean()
            errors_i.append(error_i)

    return explicit_Z, subtasks, errors, errors_i, Z_opt_i