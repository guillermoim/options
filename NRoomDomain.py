from rooms_domain import room_domain_MDP
import numpy as np

class NRoomDomain():

    def __init__(self, dims, room_size, goal_pos, goal_rooms, high_level=True, goal_reward=0, non_goal_reward=-1, penalty=-1e12):
        states, terminal_states, goal_states, P = room_domain_MDP(dims, room_size, goal_pos, goal_rooms, high_level)
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
        #p3 = next_state not in self.terminal_states or next_state in self.goal_states

        return p1 and p2 #and p3

    def apply_action(self, action):
        os = self.current_state
        ns = NRoomDomain._compute_next_state(self.current_state, action)
        r = self.r[os]
        if os not in self.terminal_states and not self._is_legal_move(os, ns):
            ns = os
            r = self.penalty  #penalty
        self.current_state = ns

        return os, ns, r

    def applicable_actions(self, state):
        return [a for a in range(self.Na) if self._is_legal_move(state, NRoomDomain._compute_next_state(state, a))]


    def reset(self, option=1):
        if option == 1:
            idx = np.random.choice(range(len(self.interior_states)))
            self.current_state = self.interior_states[idx]
        else:
            ls = self.interior_states + [t for t in self.terminal_states if t not in self.goal_states]
            idx = np.random.choice(range(len(ls)))
            self.current_state = ls[idx]
