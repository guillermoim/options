import numpy as np
import networkx as nx
from itertools import product
from scipy import sparse

infty = 1e12

def room_domain_MDP(dims, room_size, goal_pos, goal_rooms):

    assert room_size % 2 > 0, "The room size should be an odd number"

    col_rooms, row_rooms = dims

    X = col_rooms * room_size
    Y = row_rooms * room_size

    graph = nx.grid_graph(dim=[X, Y])

    renaming = {n: (0, *n) for n in graph.nodes()}
    graph = nx.relabel_nodes(graph, renaming)

    # Change from one room to another happens at the middle position
    pass_p = room_size // 2
    cols = [x*(room_size)-1 for x in range(1, X)]
    rows = [y*(room_size)-1 for y in range(1, Y)]

    #Remove intra-connections
    for (s, u, v) in graph.nodes():
        if v in cols and (u % room_size != pass_p) and v != X-1:
            graph.remove_edge((s, u, v), (s, u, v + 1))
        if u in rows and (v % room_size != pass_p) and u != Y-1:
            graph.remove_edge((s, u, v), (s, u + 1, v))

    graph = nx.DiGraph(graph)

    # Place top and bottom terminal states
    for i in range(pass_p, col_rooms*room_size, room_size):
        graph.add_edge((0, 0, i), (0, -1, i))
        graph.add_edge((0, room_size*row_rooms-1, i), (0, room_size*row_rooms,  i))

    # Place left and right terminal states
    for j in range(pass_p, row_rooms * room_size, room_size):
        graph.add_edge((0, j, 0), (0, j, -1))
        graph.add_edge((0, j, room_size*col_rooms - 1), (0, j, room_size * col_rooms))

    # Add in-goal-positioned states
    for (i, j) in product(range(col_rooms), range(row_rooms)):
        goal_i, goal_j = (room_size*j)+goal_pos[0], (room_size*i)+goal_pos[1]
        graph.add_edge((0, goal_i, goal_j), (1, goal_i, goal_j))

    # self edges
    for node in graph.nodes():
        graph.add_edge(node, node)

    A = nx.linalg.graphmatrix.adjacency_matrix(graph)
    P = A.multiply(sparse.csr_matrix(1/A.sum(axis=1)))

    goal_states = [(1, room_size*j+goal_pos[0], room_size*i+goal_pos[1]) for (i,j) in goal_rooms]
    states = list(graph.nodes())

    terminal_states = [t for t in states if t[0] == 1 or P[states.index(t), states.index(t)] == 1]

    return states, terminal_states, goal_states, P.toarray()


class NRoomDomain():

    def __init__(self, dims, room_size, goal_pos, goal_rooms, goal_reward=0, non_goal_reward=-1, penalty=-1e12):
        states, terminal_states, goal_states, P = room_domain_MDP(dims, room_size, goal_pos, goal_rooms)
        self.states = states 
        self.terminal_states = [states[i] for i in range(len(states)) if P[i,i] == 1]
        self.interior_states = [s for s in states if s not in terminal_states]
        self.goal_states = goal_states
        r = {}
        for s in states:
            if s in terminal_states and s not in goal_states:
                r[s] = penalty
            elif s in goal_states:
                r[s] = goal_reward
            else:
                r[s] = non_goal_reward

        self.r = r
        self.P = P
        self.current_state = 0
        self.actions = [0,1,2,3,4,5] #T,L,R,B,G and NoOp
        self.Ns = len(self.states)
        self.Na = len(self.actions)
        self.penalty = penalty
    
    @staticmethod
    def _compute_next_state(state, action):
        next_state = None
        if action == 0: # TOP
            next_state=(state[0] , state[1]-1, state[2])
        elif action==1: # LEFT
            next_state=(state[0] , state[1], state[2]-1)
        elif action==2: # RIGHT
            next_state=(state[0] , state[1], state[2]+1)
        elif action==3: # BOTTOM
            next_state=(state[0], state[1]+1, state[2])
        elif action==4: # TO GOAL
            next_state=(1, state[1], state[2])
        elif action==5: # NoOP
            next_state = state
        
        return next_state
    
    def _is_legal_move(self, state, next_state):
        p1 = next_state in self.states
        p2 = self.P[self.states.index(state), self.states.index(next_state)] > 0 if p1 else False
        return p1 and p2

    def apply_action(self, action):
        os = self.current_state
        ns = NRoomDomain._compute_next_state(self.current_state, action)
        r = self.r[os]
        if not self._is_legal_move(os, ns):
            ns = os
            r = self.penalty  #penalty
        self.current_state = ns

        return os, ns, r

    def applicable_actions(self, state):
        return [a for a in range(self.Na) if self._is_legal_move(state, NRoomDomain._compute_next_state(state, a))]


    def reset(self):
        idx = np.random.choice(range(len(self.interior_states)))
        self.current_state = self.interior_states[idx]