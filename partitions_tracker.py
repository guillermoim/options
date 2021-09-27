import numpy as np

class HierarchyTracker:

    def __init__(self, dim, goal_pos, goal_rooms, r_dim=5):
        H = {}
        self.r_dim = r_dim

        for x in range(dim[1]):
            for y in range(dim[0]):
                room = (x, y)
                interior_states = _get_room_interior_states(room, r_dim)
                exit_states, exit_sel = _get_exit_states(room, goal_pos, dim, goal_rooms, r_dim)
                H[room] = {'exit_states': exit_states, 'exit_selection': exit_sel, 'interior_states': interior_states}

        self.H = H

    def is_exit(self, state):

        for h in self.H:
            if state in self.H[h]['exit_states']:
                return True

        return False

def _get_room_interior_states(room, r_dim=5):
    states = []
    Y, X = room[1] * r_dim, room[0] * r_dim
    for y in range(r_dim):
        for x in range(r_dim):
            states.append((0, Y + y, X + x))

    return states

def _normalize_cell(cell, room, r_dim=5):
    z, y, x = cell
    return z, y - room[1] * r_dim, x - room[0] * r_dim

def _de_normalize_cell(n_cell, room, r_dim=5):
    z, y, x = n_cell
    return z, y + room[1] * r_dim, x + room[0] * r_dim

def _get_exit_states(room, goal_pos, dim, goal_rooms, r_dim=5):

    exit_states = []

    X, Y = room
    mid_point = r_dim // 2

    y, x = Y * r_dim + mid_point, X * r_dim + mid_point

    y_goal, x_goal = goal_pos

    exit_states.append((0, y - (mid_point + 1), x))  # TOP
    exit_states.append((0, y, x - (mid_point + 1)))  # LEFT
    exit_states.append((0, y, x + (mid_point + 1)))  # RIGHT
    exit_states.append((0, y + (mid_point + 1), x))  # BOTTOM
    exit_states.append((1, (Y * r_dim) + y_goal, (X * r_dim) + x_goal))

    selection = [1, 1, 1, 1, 1]

    if Y == 0:
        selection[0] = 0
    if Y == dim[1] - 1:
        selection[3] = 0
    if X == 0:
        selection[1] = 0
    if X == dim[0] - 1:
        selection[2] = 0
    if room not in goal_rooms:
        selection[-1] = 0

    return exit_states, np.array(selection).reshape(-1,1)


def _id_room(cell, r_dim=5):
    _, y, x = cell

    room = (max(x // r_dim, 0), max(y // r_dim, 0))

    return room