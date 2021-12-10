def applicable_options(state, dim):
    
    corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]

    # compute applicable options at each state
    options = []

    # if passenger in at pickup location, then applicable option is to go to pickup location
    if state[1] != 'TAXI':
        options.append(corners.index(state[1]))
    # otherwise, the passenger is at the taxi, thus the taxi can go to any location
    elif state[1] == 'TAXI':
        #options.append(options_corners.index(state[2]))
        options.append(corners.index(state[2]))

    return options 

def compute_next_state(state, action):
        next_state = None
        if action == 0: # TOP
            next_state=((state[0][0]-1, state[0][1]) , state[1], state[2])
        elif action==1: # LEFT
            next_state=((state[0][0], state[0][1]-1) , state[1], state[2])
        elif action==2: # RIGHT
            next_state=((state[0][0], state[0][1]+1) , state[1], state[2])
        elif action==3: # BOTTOM
            next_state=((state[0][0]+1, state[0][1]), state[1], state[2])
        elif action==4: # PICKUP
            if state[0] == state[1]:
                next_state = state[0], 'TAXI', state[2]
            elif state[1] == 'TAXI' and state[0] == state[2]:
                next_state = state[0], 'D', state[2]
        elif action==5: # NoOP
            next_state = state
        
        return next_state

def compute_next_abs_state(state, action):
    next_state = None
    if action == 0: # TOP
        next_state=(state[0], state[1]-1, state[2])
    elif action==1: # LEFT
        next_state=(state[0], state[1], state[2]-1)
    elif action==2: # RIGHT
        next_state=(state[0], state[1], state[2]+1)
    elif action==3: # BOTTOM
        next_state=(state[0], state[1]+1, state[2])
    elif action==4: # PICKUP
        if state[0] == 0:
            next_state = (1, state[1], state[2])    
    elif action==5: # NoOP
        next_state = state
    
    return next_state

def _is_legal_move(state, next_state, states, P):
    p1 = next_state in states
    p2 = P[states.index(state), states.index(next_state)] > 0 if p1 else False
    return p1 and p2

def applicable_actions(state, states, P):
    return [a for a in range(6) if _is_legal_move(state, compute_next_state(state, a), states, P)]

def abs_applicable_actions(abs_state, abs_states, P):
    
    actions = []
    
    for a in range(6):
        abs_next_state = compute_next_abs_state(abs_state, a)
        p1 = abs_next_state in abs_states
        p2 = P[abs_states.index(abs_state), abs_states.index(abs_next_state)] > 0 if p1 else False
        if p1 and p2: actions.append(a)
    
    return actions

def get_exit_states(dim):
    corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]
    E_set = [(a, 'TAXI', b) for a in corners for b in corners if a!=b]
    #E_set = [(a, a, b) for a in corners for b in corners if a!=b]

    return E_set