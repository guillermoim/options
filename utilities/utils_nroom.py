from itertools import product

def compute_next_state(state, action):
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

def _is_legal_move(state, next_state, F, P):
    p1 = next_state in F
    p2 = P[F.index(state), F.index(next_state)] > 0 if p1 else False
    #p3 = next_state not in self.terminal_states or next_state in self.goal_states

    return p1 and p2 #and p3

def applicable_actions(state, F, P):
    return [a for a in range(6) if _is_legal_move(state, compute_next_state(state, a), F, P)]

def option_is_applicable(room, room_size, option, F, goal_pos):
    X, Y = room[0]*room_size, room[1]*room_size
    goal_pos = X+goal_pos[0], Y+goal_pos[1]
    ts = [(0, X-1, Y+room_size//2), (0, X+room_size//2, Y-1), (0, X+room_size//2, Y+room_size), (0, X+room_size, Y+room_size//2), (1, *goal_pos)]
    return ts[option] in F
   
def applicable_options(room, room_size, states, goal_pos):
    return [o for o in range(5) if option_is_applicable(room, room_size, o, states, goal_pos)]

def get_exit_states(dims, room_size, states):
    
    exit_states = []
    rooms = set([room for room in product(range(dims[0]), range(dims[1]))])

    rooms = sorted(list(rooms), key=lambda x: x[0])
   
    for room in rooms:
        X, Y = room[0]*room_size, room[1]*room_size
        ts = [(0, X-1, Y+room_size//2), (0, X+room_size//2, Y-1), (0, X+room_size//2, Y+room_size), (0, X+room_size, Y+room_size//2)]
        local = []
        for t in ts:
            if t in states:
                exit_states.append(t)
                local.append(t)
    return exit_states


def project_state(cell, room, r_dim=5):
    z, x, y = cell
    return z, x - room[0] * r_dim, y - room[1] * r_dim

def id_room(cell, r_dim=5):
    _, x, y = cell

    room = (max(x // r_dim, 0), max(y // r_dim, 0))

    return room

def unproject_state(state, room, r_dim=5):
    z, x, y = state
    return z, x + room[0] * r_dim, y + room[1] * r_dim