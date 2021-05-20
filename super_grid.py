import numpy as np
import random
from lmdps import lmdps
from lmdps import hierarchical_gridworld as HG

def _prepare_grid(grid_size, r_dim, goal_pos, goal_rooms):
    '''

    Auxiliary private wrapper function to reuse code I had already.

    '''

    grid, P, _, _ = HG.create_flat_MDP(grid_size, r_dim, goal_pos, goal_rooms)
    goals = [(1, r_dim * room[1] + goal_pos[0], r_dim * room[0] + goal_pos[1]) for room in goal_rooms]
    #goal_idxs = [states.index((1, *goal)) for goal in goals_]

    return goals, list(grid.nodes()), P

class SuperGrid:

    '''
        This class simulates a SuperGrid environment.
    '''

    def __init__(self, grid_size, r_dim, goal_pos, goal_rooms, non_goal_reward, goal_reward):

        goals, states, P = _prepare_grid(grid_size, r_dim, goal_pos, goal_rooms)

        self.states = states
        self.non_terminal = states[:np.prod(grid_size) * r_dim**2]
        self.terminal = states[np.prod(grid_size) * r_dim**2:]
        self.goals = goals
        self.current_state = None
        self.P = P.copy()
        self.controls = P.copy()
        self.goal_reward = goal_reward
        self.non_goal_reward = non_goal_reward


    def reset(self):
        self.current_state = random.choice(self.non_terminal)
        return self.current_state

    def sample(self, Z=None):
        '''
            Returns a tuple of the shape (s_t, r_t, s_{t+1}) and updates the current state (next s_t)
        '''

        r = self.goal_reward if self.current_state in self.goals else self.non_goal_reward

        p = self._get_transition_prob(self.current_state, Z)

        next_state_idx = np.random.choice(range(len(self.states)), p=p)

        res = (self.current_state, r, self.states[next_state_idx])

        self.current_state = self.states[next_state_idx]

        return res

    def update_controls(self, Z):
        '''
            Gets the policy derived by a given Z function
        '''

        self.controls = lmdps.get_policy(self.P, Z)

    def _get_transition_prob(self, state, Z):

        idx = self.states.index(state)
        if state in self.terminal or Z is None:
            return self.P[idx, :]
        else:
            u = np.multiply(self.P[idx].reshape(-1, 1), Z)
            d = self.P[idx, :].reshape(-1, 1).T.dot(Z)
            return (u / d).flatten()

    def get_importance_weight(self, state, next_state):
        '''
            Gets the importance weight of a transition
        '''

        idx1, idx2 = self.states.index(state), self.states.index(next_state)
        return self.P[idx1, idx2] / self.controls[idx1, idx2]


class GridSubtasks():
    '''
        This class is an abstraction of the sub LMDPS plus the subtasks.
        It has a dictionary H (~partition) that have the rooms as keys and a tuple
        (state_indices, weight_indices, selection_indices) that are used to perform the compositionality.
    '''

    def __init__(self, r_dim, non_goal_reward, goal_reward, c, terminal_map):

        self.r_dim = r_dim
        self.non_goal_reward = non_goal_reward
        self.goal_reward = goal_reward
        self.terminal_map = terminal_map

        P_, sub_grid = HG._create_room_hierarchical(r_dim, terminal_map)

        self.abstract_states = list(sub_grid.nodes())
        self.P_ = np.asarray(P_)
        self.Z_i = _create_initial_Z(r_dim)
        self.c = c
        self.counter = 0
        self.alpha = 0
        self.lambda_ = 1

    def update(self, state, next_state, room):

        norm_state, norm_next_state = _normalize_cell(state, room), _normalize_cell(next_state, room)
        # Normalize the current state
        s_idx, ns_idx = self.abstract_states.index(norm_state), self.abstract_states.index(norm_next_state)
        # Get the rewards according to whether the transition is to an exit state or a regular one.
        r = _get_rewards_TLRBG(s_idx, self.r_dim, self.goal_reward, self.non_goal_reward)

        # Get the importance weight
        w_i_a = self.get_importance_weights(s_idx, ns_idx)

        old = self.Z_i[:, s_idx, None]
        next_ = self.Z_i[:, ns_idx, None]
        alpha = self.alpha

        self.Z_i[:, s_idx, None] = (1-alpha) * old + w_i_a * alpha * np.exp(r / self.lambda_) * next_

    def get_importance_weights(self, state_idx, next_state_idx):
            values = self.P_[state_idx, next_state_idx] * self.Z_i[:, next_state_idx]
            sums = np.einsum('j,ij->i', self.P_[state_idx, :], self.Z_i)
            res = (values / sums).reshape(-1, 1)

            try:
               return self.P_[state_idx, next_state_idx] / res
            except Exception as e:
                print(values[values > 0], sums, res)

    def update_alpha(self):
        self.counter += 1
        self.alpha = self.c / (self.c + self.counter)

    def get_z_functions(self):
        return self.Z_i


class PartitionsTracker:

    def __init__(self, grid_size, states, r_dim, goal_pos, goal_rooms):

        self.r_dim = r_dim

        H = {}
        for x in range(grid_size[1]):
            for y in range(grid_size[0]):
                room = (x, y)
                interior_states = _get_room_interior_states(room, self.r_dim)
                exit_states, exit_i = _get_exit_states(room, goal_pos, grid_size, goal_rooms)
                H[room] = {'interior_states': interior_states, 'exit_states': exit_states, 'exit_selection': exit_i}

        self.H = H           # Partitions dictionary
        self.r_dim = r_dim
        self.states = states

    def get_implicit_representation(self, current_Z, room, subtasks):
        # Get an implicit representation of Z
        implicit_Z = current_Z.copy()

        interior_states = self.H[room]['interior_states']
        exit_states = self.H[room]['exit_states']
        selection = np.array(self.H[room]['exit_selection']).reshape(-1, 1)
        exit_idxs = list(map(self.states.index, exit_states))

        weights = implicit_Z[exit_idxs]
        weights = weights * selection
        update_Z = weights.T.dot(subtasks).reshape(-1, 1)
        states_idxs = list(map(self.states.index, interior_states))
        implicit_Z[states_idxs] = update_Z[:self.r_dim**2]

        return implicit_Z

    def update_rooms_values(self, room, Z):
        pass

    def get_states_room(self, room):
        return self.H[room]['interior_states']

    def get_exit_states(self, room):
        return self.H[room]['exit_states'], np.array(self.H[room]['exit_selection']).reshape(-1, 1)


'''
 The following ones are private functions that are needed to certain tasks such as:
    - Identify which room the agent is
    - Normalize cells (i.e. identify the corresponding state in the sub LMDP)
    - Identify to which exit state I'm trasitioning {T, L, R, B, G}
    - Some other tasks.
'''


def _create_initial_Z(r_dim):
    Z = np.ones((5, r_dim**2 + 5))

    for i in range(Z.shape[0]):
        aux = np.full(5, 1)
        aux[i] = 1
        Z[i, -5:] = aux

    return Z


# This function identify the room a given cell is in
def _id_room(cell, r_dim=5):
    _, y, x = cell

    room = (max(x // r_dim, 0), max(y // r_dim, 0))

    return room


# This function normalizes a cell in a given room as if it was in (0,0)
def _normalize_cell(cell, room, r_dim=5):
    z, y, x = cell

    return z, y - room[1] * r_dim, x - room[0] * r_dim


# This function results in or None if it states in the same
def _id_transition(old_cell, new_cell):

    z, y, x = old_cell
    z_, y_, x_ = new_cell

    if abs(z - z_) > 0:
        return 'GOAL'

    elif y != y_:
        if y_ - y > 0:
            return 'BOTTOM'
        elif y_ - y < 0:
            return 'TOP'

    elif x != x_:
        if x_ - x > 0:
            return 'RIGHT'
        elif x_ - x < 0:
            return 'LEFT'

    else:
        return None


def _get_rewards_TLRBG(state_idx, r_dim=5, goal=0, non_goal=-1):

    res = np.full((5, 1), non_goal)

    if state_idx == 25:
        res[0] = goal
    elif state_idx == 26:
        res[1] = goal
    elif state_idx == 27:
        res[2] = goal
    elif state_idx == 28:
        res[3] = goal
    elif state_idx == 29:
        res[4] = goal

    return res


def _get_room_interior_states(room, r_dim=5):
    states = []
    Y, X = room[1] * r_dim, room[0] * r_dim
    for y in range(r_dim):
        for x in range(r_dim):
            states.append((0, Y + y, X + x))

    return states

def _get_exit_states(room, goal_pos, grid_size, goal_rooms, r_dim=5):
    exit_states = []

    X, Y = room
    mid_point = r_dim // 2

    y, x = Y * r_dim + mid_point, X * r_dim + mid_point

    y_goal, x_goal = goal_pos

    exit_states.append((0, y - (mid_point+1), x))  # TOP
    exit_states.append((0, y, x - (mid_point+1)))  # LEFT
    exit_states.append((0, y, x + (mid_point+1)))  # RIGHT
    exit_states.append((0, y + (mid_point+1), x))  # BOTTOM
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

    return exit_states, selection