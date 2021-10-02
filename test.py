from taxi_domain import TaxiDomain

mdp = TaxiDomain(5)
mdp.reset()

print(mdp.current_state, mdp.applicable_actions(mdp.current_state))