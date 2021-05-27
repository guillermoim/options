# Configuration 4x4 with lambda=1
echo "4x4 rooms grid, lambda=1"
python run_gridworld.py --variant 1 --config_name 4_4_1.dict --seed 222 --save_path grid_4_4_1_v1_222
python run_gridworld.py --variant 2 --config_name 4_4_1.dict --seed 222 --save_path grid_4_4_1_v2_222
python run_gridworld.py --variant 3 --config_name 4_4_1.dict --seed 222 --save_path grid_4_4_1_v3_222
python run_gridworld.py --variant flat --config_name 4_4_1.dict --seed 222 --save_path grid_4_4_1_flat_222

# Configuration 8x8 with lambda=1
echo "8x8 rooms grid, lambda=1"
python run_gridworld.py --variant 1 --config_name 8_8_1.dict --seed 333 --save_path grid_8_8_1_v1_333
python run_gridworld.py --variant 2 --config_name 8_8_1.dict --seed 333 --save_path grid_8_8_1_v2_333
python run_gridworld.py --variant 3 --config_name 8_8_1.dict --seed 333 --save_path grid_8_8_1_v3_333
python run_gridworld.py --variant flat --config_name 8_8_1.dict --seed 333 --save_path grid_8_8_1_flat_333

# Configuration 4x4 with lambda=10
echo "4x4 rooms grid, lambda=10"
python run_gridworld.py --variant 1 --config_name 4_4_10.dict --seed 222 --save_path grid_4_4_10_v1_222
python run_gridworld.py --variant 2 --config_name 4_4_10.dict --seed 222 --save_path grid_4_4_10_v2_222
python run_gridworld.py --variant 3 --config_name 4_4_10.dict --seed 222 --save_path grid_4_4_10_v3_222
python run_gridworld.py --variant flat --config_name 4_4_10.dict --seed 222 --save_path grid_4_4_10_flat_222

# Configuration 8x8 with lambda=10
echo "8x8 rooms grid, lambda=10"
python run_gridworld.py --variant 1 --config_name 8_8_10.dict --seed 333 --save_path grid_8_8_10_v1_333
python run_gridworld.py --variant 2 --config_name 8_8_10.dict --seed 333 --save_path grid_8_8_10_v2_333
python run_gridworld.py --variant 3 --config_name 8_8_10.dict --seed 333 --save_path grid_8_8_10_v3_333
python run_gridworld.py --variant flat --config_name 8_8_10.dict --seed 333 --save_path grid_8_8_10_flat_333

# Configuration 4x4 with lambda=1, r_dim=9
echo "4x4 rooms grid, lambda=1, r_dim=9"
python run_gridworld.py --variant 1 --config_name 4_4_1_9.dict --seed 666 --save_path grid_4_4_1_9_v1_666
python run_gridworld.py --variant 2 --config_name 4_4_1_9.dict --seed 666 --save_path grid_4_4_1_9_v2_666
python run_gridworld.py --variant 3 --config_name 4_4_1_9.dict --seed 666 --save_path grid_4_4_1_9_666
python run_gridworld.py --variant flat --config_name 4_4_1_9.dict --seed 666 --save_path grid_4_4_1_9_flat_666
