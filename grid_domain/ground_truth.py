import sys
sys.path.insert(0, '../..')
from lmdps import lmdps
from lmdps import hierarchical_gridworld as HG

def ground_truth(**kargs):

    grid_size = kargs['grid_size']
    r_dim = kargs['r_dim']
    goal_pos = kargs['goal_pos']
    goal_rooms = kargs['goal_rooms']
    lambda_ = kargs['lambda']
    graph, P, G, z = HG.create_flat_MDP(grid_size, r_dim, goal_pos, goal_rooms, t=lambda_)
    Z_true = lmdps.power_method(P, G, sparse=True, n_iter=2500)

    return Z_true