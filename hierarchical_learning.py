import numpy as np

class HierarchicalLearning:

    def __init__(self, c, grid_size, states, terminal, goal_pos, goal_rooms, r_dim):
        H = {}
        self.r_dim = r_dim
        self.terminal = terminal

        exit_states_inside_room = {}

        for x in range(grid_size[1]):
            for y in range(grid_size[0]):
                room = (x, y)
                exit_states_inside_room[room] = []
                interior_states = _get_room_interior_states(room, r_dim)
                exit_states, exit_sel = _get_exit_states(room, goal_pos, grid_size, goal_rooms, r_dim)
                H[room] = {'exit_states': exit_states, 'exit_selection': exit_sel, 'interior_states': interior_states}

        for h in H:
            for exit_state in H[h]['exit_states']:
                if exit_state not in self.terminal:
                    room = _id_room(exit_state, r_dim)
                    exit_states_inside_room[room].append(exit_state)

        self.exit_states_inside_room = exit_states_inside_room

        self.H = H
        self.states = states
        self.Z = np.ones((len(states), 1))
        self.c = c
        self.counter = 0
        self.alpha = None

    def update_exit_state(self, state, subtasks, abstract_states):
        # I still need to mount the solution, for now assume the subtasks.
        room = _id_room(state, self.r_dim)

        exit_states, exit_selection = self.H[room]['exit_states'], self.H[room]['exit_selection']

        normalized_state = _normalize_cell(state, room, self.r_dim)

        exit_idxs = list(map(self.states.index, exit_states))
        weights = self.Z[exit_idxs] * exit_selection

        state_idx = self.states.index(state)
        n_state_idx = abstract_states.index(normalized_state)

        new_value = weights.T.dot(subtasks).reshape(-1, 1)[n_state_idx]
        old_value = self.Z[state_idx]

        self.Z[state_idx] = (1 - self.alpha) * old_value + self.alpha * new_value


    def update_exit_states_inside_room(self, room, subtasks, abstract_states):

        exit_states, exit_selection = self.H[room]['exit_states'], self.H[room]['exit_selection']

        exit_idxs = list(map(self.states.index, exit_states))
        weights = self.Z[exit_idxs] * exit_selection

        new_values = weights.T.dot(subtasks).reshape(-1, 1)

        for s_ in self.exit_states_inside_room[room]:
            normalized_s_ = _normalize_cell(s_, room, self.r_dim)
            n_state_idx = abstract_states.index(normalized_s_)

            s_index = self.states.index(s_)
            self.Z[s_index] = (1 - self.alpha) * self.Z[s_index] + self.alpha * new_values[n_state_idx]


    def get_implicit_Z(self, room, subtasks, full=True):

        implicit_Z = self.Z.copy()

        room_states = self.H[room]['interior_states']
        room_state_idxs = list(map(self.states.index, room_states))

        exit_states, selection = self.H[room]['exit_states'], self.H[room]['exit_selection']

        exit_idxs = list(map(self.states.index, exit_states))

        weights = self.Z[exit_idxs] * selection
        inverse_selection = np.ones(selection.shape) - selection

        local = weights.T.dot(subtasks).reshape(-1, 1)
        room_values = local[:len(room_state_idxs)]
        exit_values = local[len(room_state_idxs):]
        implicit_Z[room_state_idxs] = room_values
        implicit_Z[exit_idxs] = weights + exit_values * inverse_selection

        if full:
            return implicit_Z
        else:
            return room_values

    def get_explicit_Z(self, subtasks):

        Z = self.Z.copy()

        for h in self.H:
            states_in_h = self.H[h]['interior_states']
            state_idxs = list(map(self.states.index, states_in_h))
            #partial_in_h = self.get_implicit_Z(h, subtasks, False)

            exit_states, selection = self.H[h]['exit_states'], self.H[h]['exit_selection']
            exit_idxs = list(map(self.states.index, exit_states))
            weights = self.Z[exit_idxs] * selection
            inverse_selection = np.ones(selection.shape) - selection

            local = weights.T.dot(subtasks).reshape(-1, 1)

            Z[state_idxs] = local[:len(state_idxs)]
            Z[exit_idxs] = weights + local[len(state_idxs):] * inverse_selection

        return Z

    def update_alpha(self):
        self.counter += 1
        self.alpha = self.c / (self.c + self.counter)

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

def _get_exit_states(room, goal_pos, grid_size, goal_rooms, r_dim=5):

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
    if Y == grid_size[1] - 1:
        selection[3] = 0
    if X == 0:
        selection[1] = 0
    if X == grid_size[0] - 1:
        selection[2] = 0
    if room not in goal_rooms:
        selection[-1] = 0

    return exit_states, np.array(selection).reshape(-1,1)


def _id_room(cell, r_dim=5):
    _, y, x = cell

    room = (max(x // r_dim, 0), max(y // r_dim, 0))

    return room