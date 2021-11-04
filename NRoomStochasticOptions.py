from rooms_domain import room_domain_MDP
import numpy as np
from scipy.special import kl_div
import pickle

class NRoomStochasticOptions():

    def __init__(self, states_path, passive_dyn_path, optimal_policy_path, goal_reward=0, lambda_ = 1, non_goal_reward=-1, penalty=-1e3):
        
        states = pickle.load(open(states_path, 'rb'))
        terminal_states = states[-5:]
        self.states = states
        self.terminal_states = terminal_states
        
        self.P = np.load(open(passive_dyn_path, 'rb'))

        self.current_state = None
        self.actions = [0,1,2,3,4,5] #T,L,R,B,G and NoOp
        self.penalty = penalty

        r = np.zeros((len(states), 5))

        self.op = np.load(optimal_policy_path)

        self.P_a = np.full((5, len(states), len(self.actions), len(states)), np.NaN, dtype=np.float)

        for o in range(5):
            for idx, s in enumerate(states[:-5]):
                next_states = np.where(~np.isnan(self.op[o, idx, :]))[0]
                
                p = self.op[o, idx, next_states].copy()
                
                KL = kl_div(p, self.P[idx, next_states]).sum()

                r[idx, o] = non_goal_reward - lambda_ * KL
                
                for x, ns in enumerate(next_states):
                    a = NRoomStochasticOptions._identify_transition(s, states[ns])
                    self.P_a[o, idx, a, next_states] = np.roll(p, x)

        for i, t in enumerate(states[-5:]):
            idx = states.index(t)
            r_ = np.full(5, penalty)
            r_[i] = goal_reward
            r[idx] = r_

        self.r = r


    @staticmethod
    def _identify_transition(state, next_state):
        
        if next_state [1] == state[1]-1:
            return 0 
        elif next_state[2] == state[2]-1:
            return 1
        elif next_state[2] == state[2]+1:
            return 2
        elif next_state[1] == state[1]+1:
            return 3
        elif next_state[0] != state[0]:
            return 4
        elif next_state == state:
            return 5
    
    def apply_action(self, o, action):
        
        os = self.current_state
        os_idx = self.states.index(os)

        elements = np.where(~np.isnan(self.P_a[o, os_idx, action, :]))[0]
        
        ns_idx = np.random.choice(elements, p=self.P_a[o, os_idx, action, elements])
    
        r = self.r[os_idx, o]
        ns = self.states[ns_idx]

        return os, ns, r
    
    
    def applicable_actions(self, o, state):
        idx = self.states.index(state)
        A = self.P_a[o, idx, :, :]
        E1 = np.where(np.any(~np.isnan(A), axis=1))[0].tolist()
        E2 = np.where(np.any(A!=0, axis=1))[0].tolist()

        p_actions = np.intersect1d(E1, E2)

        return p_actions


    def reset(self):
        
        idx = np.random.choice(range(len(self.states[:-5])))
        self.current_state = self.states[idx]
        