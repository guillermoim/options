import numpy
from NRoomStochasticOptions import NRoomStochasticOptions


states_path = 'solutions_rooms3x3_size5x5/abs_states_5x5.pkl'
passive_dyn = 'solutions_rooms3x3_size5x5/passive_dynamics_goal@2-3_5x5.txt'
optimal_policy = 'solutions_rooms3x3_size5x5/optimal_policy_options_goal@2-3_5x5.npy'


options_env = NRoomStochasticOptions(states_path, passive_dyn, optimal_policy)

options_env.reset()

print(options_env.apply_action(0, 1))

print(numpy.diagonal(options_env.P))