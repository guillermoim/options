import networkx.linalg
import numpy as np
import networkx as nx


def create_flat_MDP():
    dim = 5
    nav_locs = [(i,j) for i in range(dim) for j in range(dim)]

    corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]
    passenger_locs = corners + ['TAXI']
    destination_locs = corners

    states = [(xy, p, d) for xy in nav_locs for p in passenger_locs for d in destination_locs if p != d]

    transitions = []

    #create possible transitions
    loc_and_neighbors = {}

    for loc in nav_locs:
        neighbors = []
        #UP and LEFT
        if loc[0]-1 > -1: neighbors.append((loc[0]-1, loc[1]))
        if loc[1]-1 > -1: neighbors.append((loc[0], loc[1]-1))
        # DOWN and RIGHT
        if loc[0]+1 < dim: neighbors.append((loc[0]+1, loc[1]))
        if loc[1]+1 < dim: neighbors.append((loc[0], loc[1]+1))

        loc_and_neighbors[loc] = neighbors

    # I connect each
    # (taxi_loc, pass_location, dst_loc) as long as the pass_location and dst are different.
    for src in passenger_locs:
        for dst in destination_locs:
            if src == dst: continue
            for t in nav_locs:
                for neighbor in loc_and_neighbors[t]:
                    transition = (t, src, dst), (neighbor, src, dst)
                    transitions.append(transition)

    # I shall each 'exit' (pickup location) with the respective pass@taxi state
    for src in passenger_locs[:-1]:
        for dst in destination_locs:
            if src == dst: continue
            u = src, src, dst
            v = src, 'TAXI', dst
            transitions.append((u, v))

    # Four terminal states
    states.extend(['T1', 'T2', 'T3', 'T4'])

    t1 = corners[0], 'TAXI', corners[0]
    t2 = corners[1], 'TAXI', corners[1]
    t3 = corners[2], 'TAXI', corners[2]
    t4 = corners[3], 'TAXI', corners[3]

    transitions.append((t1, 'T1'))
    transitions.append((t2, 'T2'))
    transitions.append((t3, 'T3'))
    transitions.append((t4, 'T4'))

    graph = nx.DiGraph()
    graph.add_nodes_from(states)
    graph.add_edges_from(transitions)

    # Self edges
    for node in graph:
        graph.add_edge(node, node)

    for node in graph:
        if len(graph.edges(node)) > 2:
            print(node, len(graph.edges(node)), [e[1] for e in graph.edges(node)])


    return networkx.linalg.adj_matrix(graph)


if __name__ == '__main__':
    A = create_flat_MDP()
