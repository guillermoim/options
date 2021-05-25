from tqdm import tqdm
import numpy as np
from taxi_domain.error_metric import error_metric
from taxi_domain.taxi import TaxiSubstasks, TaxiProblem

def __init__():
    pass

def hlr_v2(c0, c1, **kargs):

    P = kargs['P']
    dim = kargs['dim']
    states = kargs['states']
    init_states = kargs['init_states']
    goal_states = kargs['goal_states']
    Z_true = kargs['Z_true']
    Z_i_opt = kargs['Z_i_opt']
    MAX_N_SAMPLES = kargs['max_n_samples']
    lambda_ = kargs['lambda_']
    goal_reward = kargs['goal_reward']
    non_goal_reward = kargs['non_goal_reward']
    non_goal_states = kargs['non_goal_states']

    terminals = goal_states + non_goal_states

    HL = TaxiSubstasks(dim, c1, lambda_)
    HL.update_alpha()

    Taxi = TaxiProblem(c0, dim, states, P, init_states, goal_states, non_goal_states, goal_reward, non_goal_reward)

    n_samples = 0

    errors = []
    errors_i = []

    pbar = tqdm(total=MAX_N_SAMPLES)
    n_episodes = 0

    while n_samples < MAX_N_SAMPLES:

        Taxi.update_alpha()
        Taxi.reset()

        terminate_episode = False

        state = Taxi.current_state

        current_partition = (state[1], state[2])

        n_episodes += 1

        while not terminate_episode:
            # Get a sample in the super-grid
            # Update the high-level function
            #(state, reward, next_state) = Taxi.sample(Taxi.get_implicit_representation(current_partition, HL.Z))
            #(state, reward, next_state) = Taxi.sample(None)
            (state, reward, next_state) = Taxi.sample(Taxi.get_implicit_representation(current_partition, HL.Z))

            pbar.set_description(f'c={Taxi.c} alpha={Taxi.alpha:.4f} alpha_1={HL.alpha:.3f} episodes={n_episodes}')

            # Learn the subtasks!
            if (next_state[1], next_state[2]) != (state[1], state[2]):
                HL.update(state, state, False)
                HL.update(next_state, next_state, True)
                HL.update_alpha()
            else:
                HL.update(state, next_state, False)

            if next_state in Taxi.partitions[current_partition]['exit_states']:
                Taxi.update_exit_states_in_partition(current_partition, HL.Z, HL.abstract_states)

            if next_state not in terminals:
                current_partition = next_state[1], next_state[2]

            terminate_episode = state in terminals

            n_samples += 1
            pbar.update(1)

            error = error_metric(Z_true, Taxi.get_explicit_Z(HL.Z), len(states) - len(terminals))
            A = np.log(HL.Z[:, :dim**2])
            B = np.log(Z_i_opt[:, :dim**2])
            error_i = np.abs(A - B).mean()
            errors_i.append(error_i)
            errors.append(error)

    return Taxi.get_explicit_Z(HL.Z), HL.Z, errors, errors_i, None