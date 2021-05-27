import sys
sys.path.insert(0, '../..')
from learning import ZLearning
from tqdm import tqdm
import random
import numpy as np
from .error_metric import error_metric


def __init__():
    pass


def _get_transition_prob(P, states, terminals, state, Z):
    idx = states.index(state)
    if state in terminals or Z is None:
        return P[idx, :]
    else:
        u = np.multiply(P[idx].reshape(-1, 1), Z)
        d = P[idx, :].reshape(-1, 1).T.dot(Z)
        return (u / d).flatten()


def _get_importance_weight(P, state_idx, next_state_idx, Z):
    '''
        Gets the importance weight of a transition
    '''

    values = P[state_idx, next_state_idx] * Z[next_state_idx, None]

    sums = P[state_idx, :].dot(Z)

    res = (values / sums).reshape(-1, 1)

    return P[state_idx, next_state_idx] / res

def flat_Z_learning(c, **kargs):

    P = kargs['P']
    states = kargs['states']
    init_states = kargs['init_states']
    goal_states = kargs['goal_states']
    Z_true = kargs['Z_true']
    MAX_N_SAMPLES = kargs['max_n_samples']
    lambda_ = kargs['lambda_']
    goal_reward = kargs['goal_reward']
    non_goal_reward = kargs['non_goal_reward']
    non_goal_states = kargs['non_goal_states']

    Z = ZLearning(states, P, c, tau=lambda_)

    n_samples = 0

    errors = []

    terminals = goal_states + non_goal_states

    pbar = tqdm(total=MAX_N_SAMPLES)

    while n_samples < MAX_N_SAMPLES:

        terminate_episode = False

        state = random.choice(init_states)

        Z.update_alpha()

        pbar.set_description(f'c={Z.c} alpha={Z.get_alpha():.4f}')

        while not terminate_episode:
            # Get a sample in the super-grid
            # Update the high-level function
            reward = goal_reward if state in goal_states else non_goal_reward
            state_idx = states.index(state)

            p = _get_transition_prob(P, states, terminals, state, Z.Z)

            next_state_idx = np.random.choice(np.arange(len(states)), p=p)
            next_state = states[next_state_idx]

            w_a = _get_importance_weight(P, state_idx, next_state_idx, Z.Z).flatten()[0]

            Z.update(state, next_state, reward, weight=w_a)

            terminate_episode = state in terminals

            state = next_state

            n_samples += 1
            pbar.update(1)

            error = error_metric(Z_true, Z.Z, len(states) - len(terminals))
            errors.append(error)

    return Z, errors
