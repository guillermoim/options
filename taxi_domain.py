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


def taxi_domain_MDP(dim=5, lambda_=1):
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

    A = nx.linalg.adjacency_matrix(graph).todense()
    P = A / A.sum(axis=1)

    return np.asarray(P), states, initial_states, non_goal_states, goal_states, corners


class TaxiDomain:

    def __init__(self, dims, goal_reward = 0, non_goal_reward = -1, penalty=-1e9):
        P, states, init_states, non_goal_states, goal_states, corners = taxi_domain_MDP(dims)
        self.states = states
        self.init_states = init_states
        self.terminal_states = non_goal_states + goal_states
        self.P = P
        self.goal_states = goal_states
        self.corners = corners
        self.Na = 6
        self.Ns = len(states)
        r = {}
        for s in states:
             if s not in self.terminal_states: r[s] = non_goal_reward
             elif s not in self.goal_states: r[s] = penalty
             elif s in self.goal_states: r[s] = goal_reward
        self.r = r

    def _compute_next_state(self, state, action):
        next_state = None
        if action == 0: # TOP
            next_state=((state[0][0]-1, state[0][1]) , state[1], state[2])
        elif action==1: # LEFT
            next_state=((state[0][0], state[0][1]-1) , state[1], state[2])
        elif action==2: # RIGHT
            next_state=((state[0][0], state[0][1]+1) , state[1], state[2])
        elif action==3: # BOTTOM
            next_state=((state[0][0]+1, state[0][1]), state[1], state[2])
        elif action==4: # PICKUP
            if state[0] == state[1]:
                next_state = state[0], 'TAXI', state[2]
            elif state[1] == 'TAXI' and state[0] == state[2]:
                next_state = state[0], 'D', state[2]
            elif state[1] == 'TAXI' and state[1] != state[2] and state[0] in self.corners:
                next_state = state[0], 'Forbidden', None
        elif action==5: # NoOP
            next_state = state
        
        return next_state


    def _is_legal_move(self, state, next_state):
        p1 = next_state in self.states
        p2 = self.P[self.states.index(state), self.states.index(next_state)] > 0 if p1 else False
        return p1 and p2

    def apply_action(self, action):
        os = self.current_state
        ns = self._compute_next_state(self.current_state, action)
        r = self.r[os]
        if os not in self.terminal_states and not self._is_legal_move(os, ns):
            ns = os
            r = self.penalty  # penalty
        self.current_state = ns

        return os, ns, r

    def applicable_actions(self, state):
        return [a for a in range(self.Na) if self._is_legal_move(state, self._compute_next_state(state, a))]

    def reset(self):
        idx = np.random.choice(range(len(self.init_states)))
        self.current_state = self.states[idx]