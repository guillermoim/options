from taxi_domain import TaxiDomain, create_flat_taxi_MDP

L = create_flat_taxi_MDP(5)

print(len(L[2]))
print(L[1].shape)