import networkx.linalg
import numpy as np
import networkx as nx
from lmdps.lmdps import power_method

def _create_taxi_room(r_dim):

    graph = nx.grid_graph((r_dim, r_dim)).to_directed()
    graph = nx.DiGraph(graph)
    # Change node names from (x,y) to (0,x,y) so we can have terminals @ (1,0,0) ... (1,r_dim-1, r_dim-1)
    mapping = {node:(0, *node) for node in graph.nodes}

    graph = nx.relabel_nodes(graph, mapping)

    graph.add_edge((0, 0, 0), (1, 0, 0))
    graph.add_edge((0, 0, r_dim-1), (1, 0, r_dim-1))
    graph.add_edge((0, r_dim-1, 0), (1, r_dim-1, 0))
    graph.add_edge((0, r_dim-1, r_dim-1), (1, r_dim-1, r_dim-1))

    for node in graph.nodes:
        graph.add_edge(node, node)

    A = nx.adjacency_matrix(graph).todense()
    P = A / A.sum(axis=1)

    return list(graph.nodes), P



class TaxiHL:

    def __init__(self, r_dim, c, lambda_, goal_reward = 0, non_goal_reward = -1, t=4):
        states, P = _create_taxi_room(r_dim)
        self.states = states
        self.P = P
        self.Z = np.ones((t, len(self.abstract_states)))
        self.goal_reward = goal_reward
        self.non_goal_reward = non_goal_reward
        self.c = c
        self.lambda_ = lambda_

    def update(self, state, next_state, exit):
        # States in this setting are going to be of the shape of (taxi_XY, pass_LOC, dst_LOC)
        # Thus, I reach an exit state anytime I go from
        # (taxi_XY, pass_LOC, dst_LOC) -> (taxi_XY, pass_LOC', dst_LOC)
        taxi, passenger, dst = state
        next_taxi, next_passenger, next_dst = next_state

        if next_passenger != next_passenger:
            norm_state = (0, *taxi)
            norm_next_state = (1, *next_taxi)
        else:
            if exit:
                norm_state = (1, *taxi)
                norm_next_state = (1, *taxi)
            else:
                norm_state = (0, *taxi)
                norm_next_state = (0, *taxi)

        s_idx, ns_idx = self.abstract_states.index(norm_state), self.abstract_states.index(norm_next_state)
        w_i_a = self.get_importance_weights(s_idx, ns_idx)


    def get_importance_weight(self, state_idx, next_state_idx):

        values = self.P_[state_idx, next_state_idx] * self.Z_i[:, next_state_idx]
        sums = np.einsum('j,ij->i', self.P_[state_idx, :], self.Z_i)
        res = (values / sums).reshape(-1, 1)

        return self.P_[state_idx, next_state_idx] / res

    def update_alpha(self):
        self.counter += 1
        self.alpha = self.c / (self.c + self.counter)

    def get_z_functions(self):
        return self.Z_i

def create_flat_MDP_2():
    dim = 5
    nav_locs = [(i,j) for i in range(dim) for j in range(dim)]

    corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]
    passenger_locs = corners + ['TAXI']

    # create possible transitions
    loc_and_neighbors = {}

    for loc in nav_locs:
        neighbors = []
        # UP and LEFT
        if loc[0] - 1 > -1: neighbors.append((loc[0] - 1, loc[1]))
        if loc[1] - 1 > -1: neighbors.append((loc[0], loc[1] - 1))
        # DOWN and RIGHT
        if loc[0] + 1 < dim: neighbors.append((loc[0] + 1, loc[1]))
        if loc[1] + 1 < dim: neighbors.append((loc[0], loc[1] + 1))

        loc_and_neighbors[loc] = neighbors


    transitions_1 = []
    transitions_2 = []

    # Add pickup exit transitions
    for c0 in corners:
        for c1 in corners:
            if c0 == c1: continue
            # exit state transition
            transition = (c0, c0, c1), (c0, 'TAXI', c1)
            transitions_1.append(transition)

    # Add all transitions that represent navigation in the same 'grid' (grid is defined by pass_loc x dst)
    for c0 in passenger_locs:
        for c1 in corners:
            if c0 == c1: continue
            for xy in nav_locs:
                for neighbor in loc_and_neighbors[xy]:
                    transition = (xy, c0, c1), (neighbor, c0, c1)
                    transitions_2.append(transition)

    terminal_edges = []
    # I should add terminal states
    for taxi_pos in corners:
        for dst in corners:
            transition = (taxi_pos, 'TAXI', dst), (taxi_pos, 'D', dst)
            terminal_edges.append(transition)

    goal_states = [(c, 'D', c) for c in corners]
    non_goal_states = [(c0, 'D', c1) for c0 in corners for c1 in corners if c0 != c1]

    graph = nx.DiGraph()
    graph.add_edges_from(transitions_1)
    graph.add_edges_from(transitions_2)
    graph.add_edges_from(terminal_edges)

    for node in graph.nodes():
        graph.add_edge(node, node)


    states = list(graph.nodes())
    goal_idxs = list(map(states.index, goal_states))
    print(goal_idxs)
    lambda_ = 1
    q = np.full((len(states), 1), -1 / lambda_)

    q[goal_idxs] = 0

    G = np.diagflat(np.exp(q))

    A = nx.linalg.adjacency_matrix(graph).todense()
    P = A / A.sum(axis=1)

    return G, P, states, non_goal_states, goal_states

def sample_from(states):
    return


def Z_learning_taxi():



if __name__ == '__main__':
    G, P = create_flat_MDP_2()

    # ground truth !
    Z = power_method(P, G, sparse=True, n_iter=1000)

