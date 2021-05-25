import sys
sys.path.insert(0, '../..')
from lmdps import lmdps
from taxi_domain.taxi import create_flat_taxi_MDP, _create_taxi_room
import numpy as np


def ground_truth(dim, lambda_=1):
    G, P, _, _, _, _ = create_flat_taxi_MDP(dim, lambda_)
    Z = lmdps.power_method(P, G, sparse=True)
    return Z


def ground_truth_subtasks(dim, lambda_=1):
    states, P = _create_taxi_room(dim)
    corners = [(1, 0, 0), (1, dim-1, 0), (1, 0, dim-1), (1, dim-1, dim-1)]

    Z_i = np.ndarray((len(corners), len(states)))

    for i, c in enumerate(corners):
        q = np.full((len(states), 1), -1)
        q[states.index(c)] = 0
        G = np.diagflat(np.exp(q/lambda_))
        Z = lmdps.power_method(P, G, True)
        Z_i[i, :, None] = Z
    return Z_i
