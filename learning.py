import numpy as np

class Learning:

    '''
    Abstract class of which ZLearning and QLearning inherit
    '''

    def __init__(self, states, prob_matrix):
        self.states = states
        self.P = prob_matrix

class ZLearning(Learning):
    '''
        Modularization of ZLearning so it can be reused with more environments.
    '''

    def __init__(self, states, prob_matrix, c, tau=1):
        super().__init__(states, prob_matrix)
        self.tau = tau
        self.Z = np.ones(len(states)).reshape(-1, 1)
        self.runs = 0
        self.alpha = 0
        self.c = c

    def update(self, state, next_state, reward, weight=1):
        state_idx = self.states.index(state)
        next_state_idx = self.states.index(next_state)
        alpha = self.alpha
        self.Z[state_idx] = (1-alpha) * self.Z[state_idx] + alpha * weight * np.exp(reward/self.tau) * self.Z[next_state_idx]

    def update_alpha(self):
        self.runs+=1
        self.alpha = self.c / (self.c + self.runs +1)

    def update_states(self, state_idxs, values):
        '''
        Given a collection of state indices, update their Z value with the values passed in as parameters.
        This must hold [size(state_idxs) == size(values)]
        '''
        self.Z[state_idxs] = (1 - self.alpha) * self.Z[state_idxs] + self.alpha * values

    def get_alpha(self):
        return self.alpha

    def get_Z_function(self):
        return self.Z

    def get_value_at_states(self, state_idxs):
        return self.Z[state_idxs]





