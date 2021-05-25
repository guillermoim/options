import networkx as nx
import numpy as np
import random

def _create_taxi_room(r_dim):

    graph = nx.grid_graph((r_dim, r_dim)).to_directed()
    graph = nx.DiGraph(graph)
    # Change node names from (x,y) to (0,x,y) so we can have terminals @ (1,0,0) ... (1,r_dim-1, r_dim-1)
    mapping = {node:(0, *node) for node in graph.nodes}

    graph = nx.relabel_nodes(graph, mapping)

    graph.add_edge((0, 0, 0), (1, 0, 0))
    graph.add_edge((0, r_dim-1, 0), (1, r_dim-1, 0))
    graph.add_edge((0, 0, r_dim-1), (1, 0, r_dim-1))
    graph.add_edge((0, r_dim-1, r_dim-1), (1, r_dim-1, r_dim-1))

    for node in graph.nodes:
        graph.add_edge(node, node)

    A = nx.adjacency_matrix(graph).todense()
    P = A / A.sum(axis=1)

    return list(graph.nodes), np.asarray(P)


def create_flat_taxi_MDP(dim=5, lambda_=1):
    '''
    This method creates the flat MDP for our taxi domain.
    TODO: Explain better how it works and the return statement.
    '''

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

    # Add pickup exit-states transitions
    for c0 in corners:
        for c1 in corners:
            if c0 == c1: continue
            # exit state transition
            transition = (c0, c0, c1), (c0, 'TAXI', c1)
            transition_reversed = (c0, 'TAXI', c1), (c0, c0, c1)
            transitions_1.append(transition)
            transitions_1.append(transition_reversed)

    # Add all transitions that represent navigation in the same 'grid' (grid is defined by pass_loc x dst)
    for c0 in passenger_locs:
        for c1 in corners:
            if c0 == c1: continue
            for xy in nav_locs:
                for neighbor in loc_and_neighbors[xy]:
                    transition = (xy, c0, c1), (neighbor, c0, c1)
                    transitions_2.append(transition)

    terminal_edges = []
    # I should add a signle terminal state that happens when taxi_pos

    for corner in corners:
        transition = (corner, 'TAXI', corner), (corner, 'D', corner)
        terminal_edges.append(transition)

    # Also, I need to add some terminals in the 1D to allow exploration, these terminals happen
    # (taxi_loc, pass, dst) whenever taxi_loc = corner and taxi_loc != pass.

    for taxi in corners:
        for passenger in corners:
            for dst in corners:
                if taxi == passenger: continue
                if passenger == dst: continue
                transition = (taxi, passenger, dst), (taxi, 'Forbidden', None)
                terminal_edges.append(transition)


    graph = nx.DiGraph()
    graph.add_edges_from(transitions_1)
    graph.add_edges_from(transitions_2)
    graph.add_edges_from(terminal_edges)

    for node in graph.nodes():
        graph.add_edge(node, node)

    states = list(graph.nodes())

    initial_states = [s for s in states if s[1] not in ('D', 'TAXI', 'Forbidden')]

    goal_states = [s for s in states if s[1] == 'D']
    non_goal_states = [s for s in states if s[1] == 'Forbidden']

    goal_idxs = list(map(states.index, goal_states))

    q = np.full((len(states), 1), -1 / lambda_)

    q[goal_idxs] = 0

    G = np.diagflat(np.exp(q))

    A = nx.linalg.adjacency_matrix(graph).todense()
    P = A / A.sum(axis=1)

    return np.asarray(G), np.asarray(P), states, initial_states, non_goal_states, goal_states


class TaxiSubstasks:

    def __init__(self, r_dim, c, lambda_=1, goal_reward = 0, non_goal_reward = -1, t=4):
        states, P = _create_taxi_room(r_dim)
        self.r_dim = r_dim
        self.abstract_states = states
        self.P = P
        self.Z = np.ones((t, len(states)))
        self.goal_reward = goal_reward
        self.non_goal_reward = non_goal_reward
        self.c = c
        self.lambda_ = lambda_
        self.counter = 0

    def update(self, state, next_state, exit):
        # States in this setting are going to be of the shape of (taxi_XY, pass_LOC, dst_LOC)
        # Thus, I reach an exit state anytime I go from
        # (taxi_XY, pass_LOC, dst_LOC) -> (taxi_XY, pass_LOC', dst_LOC)
        taxi, passenger, dst = state
        next_taxi, next_passenger, next_dst = next_state

        if passenger != next_passenger:
            norm_state = (0, *taxi)
            norm_next_state = (1, *next_taxi)
        else:
            if exit:
                norm_state = (1, *taxi)
                norm_next_state = (1, *taxi)
            else:
                norm_state = (0, *taxi)
                norm_next_state = (0, *next_taxi)

        r = self._get_rewards(norm_state)

        s_idx, ns_idx = self.abstract_states.index(norm_state), self.abstract_states.index(norm_next_state)
        w_i_a = self.get_importance_weights(s_idx, ns_idx)

        old = self.Z[:, s_idx, None]
        next_ = self.Z[:, ns_idx, None]
        alpha = self.alpha

        self.Z[:, s_idx, None] = (1 - alpha) * old + w_i_a * alpha * np.exp(r / self.lambda_) * next_

    def _get_rewards(self, norm_state):

        corners = [(0, 0), (self.r_dim-1, 0), (0, self.r_dim-1), (self.r_dim-1, self.r_dim-1)]

        res = np.full((4, 1), self.non_goal_reward)

        if norm_state[0] and norm_state[1:] in corners:
            res[corners.index(norm_state[1:])] = self.goal_reward

        return res

    def get_importance_weights(self, state_idx, next_state_idx):

        values = self.P[state_idx, next_state_idx] * self.Z[:, next_state_idx]
        sums = np.einsum('j,ij->i', self.P[state_idx, :], self.Z)
        res = (values / sums).reshape(-1, 1)

        return self.P[state_idx, next_state_idx] / res

    def update_alpha(self):
        self.counter += 1
        self.alpha = self.c / (self.c + self.counter)

    def get_z_functions(self):
        return self.Z

def create_partitions(dim, states):

    partitions = {}
    corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]

    E_set = []

    for c_0 in corners + ['TAXI']:
        for c_1 in corners:
            if c_0 == c_1: continue
            partitions[c_0, c_1] = {}
            partitions[c_0, c_1]['states'] = [(taxi, pass_, dst) for (taxi, pass_, dst) in states if pass_ == c_0 and dst == c_1]

    for key in partitions:

        selection = np.zeros((4, 1))

        if key[0] == 'TAXI':
            exit_states = [(c, c, key[1]) for c in corners if c != key[1]] + [(key[1], 'D', key[1])]
            exit_states = sorted(exit_states, key=lambda x: (x[0][1], x[0][0]))
            selection = np.ones((4, 1))

            E_set.extend(exit_states)

        else:
            exit_states = [(key[0], 'TAXI', key[1])] + [(c, 'Forbidden', None) for c in corners if c!= key[0]]
            exit_states = sorted(exit_states, key=lambda x: (x[0][1], x[0][0]))

            selection[corners.index(key[0])] = 1

            E_set.extend(exit_states)

        partitions[key]['selection'] = selection
        partitions[key]['exit_states'] = exit_states

    return partitions, set(E_set)

class TaxiProblem:

    def __init__(self, c, dim, states, P, init_states, goal_states, non_goal_states, goal_reward = 0, non_goal_reward = -1):
        self.dim = dim
        self.states = states
        self.P = P
        self.init_states = init_states
        self.goal_states = goal_states
        self.non_goal_states = non_goal_states
        self.partitions, self.exit_set = create_partitions(dim, states)
        self.Z = np.ones((len(states), 1))
        self.c = c
        self.counter = 0
        self.goal_reward = goal_reward
        self.non_goal_reward = non_goal_reward
        self.alpha = 0

        exit_states_inside_partition = {h : [] for h in self.partitions}

        for h in self.partitions:
            for exit_state in self.partitions[h]['exit_states']:
                if exit_state not in self.goal_states + self.non_goal_states:
                    partition = exit_state[1], exit_state[2]
                    exit_states_inside_partition[partition].append(exit_state)

        self.exit_states_inside_partition = exit_states_inside_partition

    def reset(self):
        self.current_state = random.choice(self.init_states)

    def sample(self, Z):
        r = self.goal_reward if self.current_state in self.goal_states else self.non_goal_reward

        p = self._get_transition_prob(self.current_state, Z)

        next_state_idx = np.random.choice(range(len(self.states)), p=p)

        res = (self.current_state, r, self.states[next_state_idx])

        self.current_state = self.states[next_state_idx]

        return res

    def _get_transition_prob(self, state, Z):
        idx = self.states.index(state)
        if state in self.goal_states or Z is None:
            return self.P[idx, :]
        else:
            u = np.multiply(self.P[idx].reshape(-1, 1), Z)
            d = self.P[idx, :].reshape(-1, 1).T.dot(Z)
            return (u / d).flatten()

    def get_implicit_representation(self, h, subtasks):
        # Get an implicit representation of Z
        implicit_Z = self.Z.copy()

        exit_states = self.partitions[h]['exit_states']
        selection = np.array(self.partitions[h]['selection']).reshape(-1, 1)
        exit_idxs = list(map(self.states.index, exit_states))
        weights = implicit_Z[exit_idxs]

        interior_states = self.partitions[h]['states']
        auxiliary_ordering = sorted(interior_states, key=lambda x: (x[0][0], x[0][1]))
        partition_idxs = list(map(self.states.index, auxiliary_ordering))
        weights = weights * selection

        update_Z = weights.T.dot(subtasks)[:, :self.dim ** 2].reshape(-1, 1)
        implicit_Z[partition_idxs] = update_Z

        return implicit_Z

    def update_exit_state(self, state, subtasks, abstract_states):
        # I still need to mount the solution, for now assume the subtasks.

        taxi, passenger, dst = state
        h = state[1], state[2]
        exit_states, exit_selection = self.partitions[h]['exit_states'], self.partitions[h]['selection']
        # The state is an exit state for another partition, thus to update it we need to take (0, taxi)
        normalized_state = (0, *taxi)
        exit_idxs = list(map(self.states.index, exit_states))
        weights = self.Z[exit_idxs] * exit_selection
        state_idx = self.states.index(state)
        n_state_idx = abstract_states.index(normalized_state)
        new_value = weights.T.dot(subtasks).reshape(-1, 1)[n_state_idx]
        old_value = self.Z[state_idx].copy()

        self.Z[state_idx] = (1 - self.alpha) * old_value + self.alpha * new_value

    def update_exit_states_in_partition(self, h, subtasks, abstract_states):

        exit_states, selection = self.partitions[h]['exit_states'], self.partitions[h]['selection']

        exit_idxs = list(map(self.states.index, exit_states))
        weights = self.Z[exit_idxs] * selection

        new_values = weights.T.dot(subtasks).reshape(-1, 1)

        for s_ in self.exit_states_inside_partition[h]:
            normalized_s_ = s_[0]
            n_state_idx = abstract_states.index((0, *normalized_s_))
            s_index = self.states.index(s_)
            self.Z[s_index] = (1 - self.alpha) * self.Z[s_index] + self.alpha * new_values[n_state_idx]

    def get_explicit_Z(self, subtasks):

        Z = self.Z.copy()

        for h in self.partitions:
            interior_states = self.partitions[h]['states']
            auxiliary_ordering = sorted(interior_states, key=lambda x: (x[0][0], x[0][1]))
            partition_idxs = list(map(self.states.index, auxiliary_ordering))

            exit_states, selection = self.partitions[h]['exit_states'], self.partitions[h]['selection']
            exit_idxs = list(map(self.states.index, exit_states))
            weights = self.Z[exit_idxs] * selection
            inverse_selection = np.ones(selection.shape) - selection

            local = weights.T.dot(subtasks).reshape(-1, 1)

            Z[partition_idxs] = local[:len(partition_idxs)]
            Z[exit_idxs] = weights + local[len(partition_idxs):] * inverse_selection

        return Z

    def update_alpha(self):
        self.counter += 1
        self.alpha = self.c / (self.c + self.counter)

    def is_exit(self, state):
        return state in self.exit_set and state not in self.goal_states + self.non_goal_states

    def get_values_partition(self, partition, subtasks):
        states_ = self.partitions[partition]['states']
        idxs = list(map(self.states.index, states_))
        return states_, self.get_explicit_Z(subtasks)[idxs]
