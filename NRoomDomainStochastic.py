from rooms_domain import room_domain_MDP
import numpy as np
from scipy.special import kl_div

class NRoomDomainStochastic():

    def __init__(self, dims, room_size, goal_pos, goal_rooms, path=None, high_level=True, goal_reward=0, lambda_ = 1, non_goal_reward=-1, penalty=-1e12):
        states, terminal_states, goal_states, P = room_domain_MDP(dims, room_size, goal_pos, goal_rooms, high_level)
        self.states = states 
        self.terminal_states = [states[i] for i in range(len(states)) if P[i,i] == 1]
        self.interior_states = [s for s in states if s not in terminal_states]
        self.goal_states = goal_states
        
        self.P = P
        self.current_state = None
        self.actions = [0,1,2,3,4,5] #T,L,R,B,G and NoOp
        self.Ns = len(self.states)
        self.Na = len(self.actions)
        self.penalty = penalty
        self.P_a = np.full((self.Ns, self.Na, self.Ns), 0, dtype=np.float)

        r = {}

        if path != None:
            op = np.loadtxt(path)
            self.op = op
            for idx, s in enumerate(self.interior_states):
                next_states = np.where(~np.isnan(op[idx, :]))[0]
                p = op[idx, next_states].copy()

                KL = kl_div(p, self.P[idx, next_states]).sum()

                r[s] = non_goal_reward - lambda_ * KL
                for a in range(len(next_states)):
                    self.P_a[idx, a, next_states] = np.roll(p, a)

       
        for s in states:
            if s in goal_states:
                r[s] = goal_reward

        self.r = r

    def apply_action(self, action):
        
        os = self.current_state
        os_idx = self.states.index(os)
        p = self.P_a[self.states.index(os), action, :].nonzero()[0]
        
        ns_idx = np.random.choice(p, p=self.P_a[os_idx, action, p])
    
        r = self.r[os]
        ns = self.states[ns_idx]
        self.current_state = ns

        return os, ns, r
    
    
    def applicable_actions(self, state):
        idx = self.states.index(state)
        A = self.P_a[idx, :, :]
        p_actions = np.where(np.any(A!=0, axis=1))[0].tolist()
        return p_actions


    def reset(self):
        
        idx = np.random.choice(range(len(self.interior_states)))
        self.current_state = self.interior_states[idx]
        
