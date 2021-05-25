# 5x5 grids, lambda=1
python run_taxi.py --variant 1 --config_name taxi_5_1.dict --seed 333 --save_path taxi_5_v1_333
python run_taxi.py --variant 2 --config_name taxi_5_1.dict --seed 333 --save_path taxi_5_v2_333
python run_taxi.py --variant 3 --config_name taxi_5_1.dict --seed 333 --save_path taxi_5_v3_333
python run_taxi.py --variant flat --config_name taxi_5_1.dict --seed 333 --save_path taxi_5_flat_333

# 5x5 grids, lambda=10
python run_taxi.py --variant 1 --config_name taxi_5_10.dict --seed 333 --save_path taxi_5_10_v1_333
python run_taxi.py --variant 2 --config_name taxi_5_10.dict --seed 333 --save_path taxi_5_10_v2_333
python run_taxi.py --variant 3 --config_name taxi_5_10.dict --seed 333 --save_path taxi_5_10_v3_333
python run_taxi.py --variant flat --config_name taxi_5_10.dict --seed 333 --save_path taxi_5_10_flat_333

# 10x10 grids, lambda=1
python run_taxi.py --variant 1 --config_name taxi_10_1.dict --seed 333 --save_path taxi_10_v1_333
python run_taxi.py --variant 2 --config_name taxi_10_1.dict --seed 333 --save_path taxi_10_v2_333
python run_taxi.py --variant 3 --config_name taxi_10_1.dict --seed 333 --save_path taxi_10_v3_333
python run_taxi.py --variant flat --config_name taxi_10_1.dict --seed 333 --save_path taxi_10_flat_333

# 10x10 grids, lambda=10
python run_taxi.py --variant 1 --config_name taxi_10_10.dict --seed 333 --save_path taxi_10_10_v1_333
python run_taxi.py --variant 2 --config_name taxi_10_10.dict --seed 333 --save_path taxi_10_10_v2_333
python run_taxi.py --variant 3 --config_name taxi_10_10.dict --seed 333 --save_path taxi_10_10_v3_333
python run_taxi.py --variant flat --config_name taxi_10_10.dict --seed 333 --save_path taxi_10_10_flat_333