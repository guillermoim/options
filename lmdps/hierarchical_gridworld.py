import numpy as np
import networkx as nx
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from .lmdps import eigenvector, power_method, solve_lmdp
from scipy import sparse


def create_transition_matrix_room(N_DIM, absorbing_states, self_move=True):

    graph = nx.grid_graph(dim=[N_DIM, N_DIM])

    if self_move:
        for i in graph.nodes():
            graph.add_edge(i, i)

    A = nx.linalg.graphmatrix.adjacency_matrix(graph, weight='weight').todense()
    P = A / A.sum(axis=1)

    for state in absorbing_states:
        P[state, :] = 0
        P[state, state] = 1

    return P


def create_gridworld_lmdp(N_DIM, goal_absorbing, non_goal_absorbing, goal=1, non_goal=0, self_move=True):
    '''
    :param N_DIM:
    :param t_states:
    :param interior_states:
    :param boundary_states:
    :return: P (transition prob. matrix), q (states cost rate), t_states (terminal states iterable)
    '''
    assert 0 < N_DIM < 20, "N_DIM must be between 1 and 19"
    assert isinstance(goal_absorbing, list) or isinstance(goal_absorbing, tuple), "t_states must be a list or tuple"
    assert all(isinstance(x, int) for x in goal_absorbing), "all states in t_states must be integers"

    # First of all, model mdp in as graph and add self-loops
    graph = nx.grid_graph(dim=[N_DIM,N_DIM])

    if self_move:
        for i in graph.nodes():
            graph.add_edge(i, i)

    A = nx.linalg.graphmatrix.adjacency_matrix(graph, weight='weight').todense()
    P = A /A.sum(axis=1)

    for t in goal_absorbing + non_goal_absorbing:
        P[t, :] = 0
        P[t, t] = 1

    q = np.full((N_DIM**2, 1), non_goal)
    for t in goal_absorbing:
        q[t] = goal

    return P, q, goal_absorbing, non_goal_absorbing



def create_flat_MDP(dims, room_size, goal_pos, goal_rooms, goal_r=0, non_goal_r=-1, t=1):

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

    # Place the top & bottom walls

    for i in range(pass_p, col_rooms*room_size, room_size):
        graph.add_edge((0, 0, i), (0, -1, i))
        graph.add_edge((0, room_size*row_rooms-1, i), (0, room_size*row_rooms,  i))

    # Place the left and right walls
    for j in range(pass_p, row_rooms * room_size, room_size):
        graph.add_edge((0, j, 0), (0, j, -1))
        graph.add_edge((0, j, room_size*col_rooms - 1), (0, j, room_size * col_rooms))

    for (i, j) in product(range(col_rooms), range(row_rooms)):

        goal_i, goal_j = (room_size*j)+goal_pos[0], (room_size*i)+goal_pos[1]
        graph.add_edge((0, goal_i, goal_j), (1, goal_i, goal_j))

    # self edges
    for node in graph.nodes():
        graph.add_edge(node, node)

    A = nx.linalg.graphmatrix.adjacency_matrix(graph)
    P = A.multiply(sparse.csr_matrix(1/A.sum(axis=1)))

    goals_indices = [list(graph.nodes).index((1, room_size*j+goal_pos[0], room_size*i+goal_pos[1])) for (i,j) in goal_rooms]

    q = np.full(P.shape[0], np.exp(non_goal_r/t))

    for i in goals_indices:
        q[i] = np.exp(goal_r/t)

    G = sparse.csr_matrix(np.diagflat(q))
    z = sparse.csr_matrix(np.ones((P.shape[0], 1)))
    P = sparse.csr_matrix(P)

    return graph, P.toarray(), G, z


def _create_room_hierarchical(N_DIM, terminal_neighbors):

    graph = nx.DiGraph(nx.grid_graph(dim=[N_DIM, N_DIM]))

    renaming = {n: (0, *n) for n in graph.nodes()}
    graph = nx.relabel_nodes(graph, renaming)

    for n in terminal_neighbors:
        t = terminal_neighbors[n]
        graph.add_edge(t, n)

    for i in graph.nodes():
        graph.add_edge(i, i)

    A = nx.linalg.graphmatrix.adjacency_matrix(graph, weight='weight').todense()

    P = A * 1 / A.sum(axis=1)

    return P, graph


def _create_TLRBG(N_DIM, goal_pos, t=1):

    assert N_DIM % 2 > 0, "N_DIM needs to be an odd number"

    # transition point
    pp = N_DIM // 2
    last_idx = N_DIM**2

    t_aux = {'TOP': (0, 0, pp), 'LEFT': (0, pp, 0), 'RIGHT': (0, pp, N_DIM-1), 'BOTTOM': (0, N_DIM-1, pp), 'GOAL': (0, *goal_pos)}
    P, graph = _create_room_hierarchical(N_DIM, t_aux)
    terminals = {'TOP': last_idx, 'LEFT': last_idx+1, 'RIGHT': last_idx+2, 'BOTTOM': last_idx+3, 'GOAL': last_idx+4}

    goals = {}
    states = list(graph.nodes())
    for ts in terminals:
        q = np.full((P.shape[0], 1), -1)
        q[terminals[ts], 0] = 0
        G = np.diagflat(np.exp(q / t))
        goals[ts] = power_method(P, G, 10000)

    G = goals['GOAL']
    T = goals['TOP']
    L = goals['LEFT']
    R = goals['RIGHT']
    B = goals['BOTTOM']

    scl = MinMaxScaler()

    scl.fit(G)
    G = scl.transform(G)
    scl.fit(T)
    T = scl.transform(T)
    scl.fit(L)
    L = scl.transform(L)
    scl.fit(R)
    R = scl.transform(R)
    scl.fit(B)
    B = scl.transform(B)

    return [T, L, R, B, G], P


def _compose_TLRBG(dims, room, goals):

    X, Y = dims
    x, y = room

    TLRBG = [0, 0, 0, 0, 0]

    if x != 0:  # TOP
        TLRBG[0] = 1
    if x != X - 1:  # BOTTOM
        TLRBG[3] = 1
    if y != 0:  # LEFT
        TLRBG[1] = 1
    if y != Y - 1:  # RIGHT
        TLRBG[2] = 1

    if room in goals:
        TLRBG[-1] = 1

    return np.array(TLRBG).reshape(-1, 1)


def hierarchical_solver(dims, R_DIM, goal_pos, goal_rooms, n_iter):
    X, Y = dims
    graph = nx.grid_graph(dim=[X, Y])
    total_terminals_per_room = np.zeros((Y, X))
    z_ = []
    N_t = 0

    # Store how many active terminal states there are per room.
    act_tstates_by_room = {}
    # Store the initial index in Z of each room.
    offsets_by_room = {}

    terminal_idxs = [R_DIM//2, R_DIM*(R_DIM//2),  R_DIM*(R_DIM//2)+(R_DIM-1), R_DIM*(R_DIM-1)+(R_DIM//2), R_DIM**2+5-1]

    T,L,R,B,G = _create_TLRBG(R_DIM, goal_pos)

    TLRBG = np.hstack([T,L,R,B,G]).T

    for room in graph:

        y, x = room
        # Here we get the active terminal states for each room in a binary fashion
        # Where each position represents TLRBG
        act_tstates = _compose_TLRBG(dims, room, goal_rooms)
        # Store the number of terminal states per room
        total_terminals_per_room[room] = act_tstates.sum()
        # Update the total number of terminal states
        N_t += act_tstates.sum()
        # Store the number of active states in this room
        act_tstates_by_room[room] = act_tstates
        # Compute the Z function for the room according to the active T states
        z_local = act_tstates.T.dot(TLRBG).reshape(-1, 1)

        # Get the indices in a 1D fashion of each terminal state
        terminals = np.array(terminal_idxs).reshape(-1, 1)
        # Indices of the active terminal states
        idxs = act_tstates * terminals
        offsets_by_room[room] = np.nonzero(idxs)[0].flatten().tolist() #np.nonzero(idxs)[0].tolist()

        # Take the specific values for each terminal states and store it
        values = z_local.flatten()[idxs[idxs > 0]].tolist()
        z_.extend(values)

    acc_indices = (total_terminals_per_room.cumsum().reshape(Y, X) - total_terminals_per_room.reshape(Y, X))

    Q = np.zeros((N_t, N_t))

    # Fill in Q by iterating over the neighbors
    room_locs = {}

    for room in graph.nodes():

        y, x = room

        pos_room = int(acc_indices[room])

        tstates = act_tstates_by_room[room]
        tstates = np.argwhere(np.any(tstates > 0, axis=1)).flatten().tolist()

        room_locs[room] = dict()

        for neighbor in graph.neighbors(room):
            y_, x_ = neighbor

            # i represents the iteration index, while c represents an "indicator" index for each of the
            # terminal states
            for i, c in enumerate(tstates):

                # recover the initial position of the neighbor in Z so that I can get the terminal states values
                # by adding the offsets
                pos_neighbor = int(acc_indices[neighbor])
                # translate from c index to a terminal index
                idx = terminal_idxs[c]

                if room in goal_rooms:
                    room_locs[room].update({'G': (room, offsets_by_room[room].index(4))})
                    Q[pos_room + i, pos_room + offsets_by_room[room].index(4)] = G[idx]

                    if c == 4:
                        # In the goal value I shall not modify anything, that's why it's only connected to itself
                        continue

                if y_ == y - 1:
                    room_locs[room].update({'T': (neighbor, offsets_by_room[neighbor].index(3))})
                    Q[pos_room + i, pos_neighbor + offsets_by_room[neighbor].index(3)] = T[idx]
                if y_ == y + 1:
                    room_locs[room].update({'B': (neighbor, offsets_by_room[neighbor].index(0))})
                    Q[pos_room + i, pos_neighbor + offsets_by_room[neighbor].index(0)] = B[idx]
                if x_ == x - 1:
                    room_locs[room].update({'L': (neighbor, offsets_by_room[neighbor].index(2))})
                    Q[pos_room + i, pos_neighbor + offsets_by_room[neighbor].index(2)] = L[idx]
                if x_ == x + 1:
                    room_locs[room].update({'R': (neighbor, offsets_by_room[neighbor].index(1))})
                    Q[pos_room + i, pos_neighbor + offsets_by_room[neighbor].index(1)] = R[idx]

    z_s = np.array(z_).reshape(-1, 1)

    for _ in range(n_iter):
        z_s = np.matmul(Q, z_s)

    Z = np.zeros((Y * R_DIM, X * R_DIM))

    # Recover the solution, according to the dictionaries constructed above
    functions = {}

    for room in room_locs:

        y, x = room

        functions[room] = []

        for loc in room_locs[room]:

            neighbor, offset = room_locs[room][loc]
            idx = int(acc_indices[neighbor] + offset)

            f = None

            if loc == 'T':
                f = z_s[idx] * T
            if loc == 'L':
                f = z_s[idx] * L
            if loc == 'R':
                f = z_s[idx] * R
            if loc == 'B':
                f = z_s[idx] * B
            if loc == 'G':
                f = G

            functions[room].append(f)
        functions[room] = np.array(functions[room]).sum(axis=0)

        auxiliary_z = functions[room][:R_DIM**2].reshape(R_DIM, R_DIM)

        Z[y * R_DIM:y * R_DIM + R_DIM, x * R_DIM:x * R_DIM + R_DIM] = auxiliary_z

    # TODO: to recover the full Z function I'd need to retrieve the walls and non-goals t states values

    return Q, Z, z_s