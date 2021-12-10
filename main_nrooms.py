from nrooms_options import Q_options

errors = Q_options('configs/nrooms_3x3.pkl', 0.15, 0.3, 1000, 3000)
print(errors[-1])