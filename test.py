from taxi_domain import TaxiDomain

dim = 5

mdp = TaxiDomain(dim)
mdp.current_state = ((0,0), 'TAXI', (4,4))

options_corners = [(0,0), (dim-1,0), (0,dim-1), (dim-1,dim-1)]


for a in mdp.applicable_actions(mdp.current_state):
    state = mdp.current_state
    print(mdp._is_legal_move( ((0,0), (0,0), (0,4)), ((0,0), 'TAXI', (0,4))))
    print(mdp._is_legal_move(((0,0), 'TAXI', (0,4)), ((0,0), (0,0), (0,4))))