import numpy as np
from NRoomDomainStochastic import NRoomDomainStochastic
import pickle 

states = pickle.load(open('solutions_lmdps/states_3x3_goal@0-0_rooms5x5', 'rb'))


dims = (3,3)
room_size = 5
goal_pos = (2,3)
goal_rooms = [(0,0)]

env = NRoomDomainStochastic(dims, room_size, goal_pos, goal_rooms, path='solutions_lmdps/policy_3x3_goal@0-0_rooms5x5.txt')

states2 = env.states

Q = np.loadtxt('Q_stochastic_3x3.txt')
Z = np.loadtxt('solutions_lmdps/Z_3x3_goal@0-0_rooms5x5.txt')

for i in range(Z.shape[0]):
    print(i, states2[i], np.nanmax(Q[i, :]),states[i], np.log(Z[i]))