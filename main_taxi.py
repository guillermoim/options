from taxi_options import Q_options

errors = Q_options('configs/taxi_5.pkl', 0.15, 0.3, 1000, 3000)
print(errors[-1])