from grid_domain import *
import argparse
import pickle
import numpy
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical Gridworld')
    parser.add_argument('--variant', type=str)
    parser.add_argument('--config_name', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    variant = args.variant
    config_path = args.config_name
    save_path = args.save_path
    seed = args.seed

    numpy.random.seed(seed)
    random.seed(seed)

    execution = pickle.load(open(f'configs_grid/{config_path}', 'rb'))

    X, Y = execution['grid_size']

    c0s = [5000, 10000]
    c1s = [1500]

    cs = [1000, 10000, 30000, 50000]

    if variant == '1':
        print('Executing variant 1')
        results = {}
        for c0 in c0s:
            for c1 in c1s:
                res1 = hlr_v1(c0, c1, **execution)
                results[c0, c1] = res1

        pickle.dump(results, open(f'results/{save_path}.pkl', 'wb'))

    if variant == '2':
        print('Executing variant 2')
        results = {}
        for c0 in c0s:
            for c1 in c1s:
                res2 = hlr_v2(c0, c1, **execution)
                results[c0, c1] = res2

        pickle.dump(results, open(f'results/{save_path}.pkl', 'wb'))

    if variant == '3':
        print('Executing variant 3')
        results = {}
        for c0 in c0s:
            for c1 in c1s:
                res3 = hlr_v3(c0, c1, **execution)
                results[c0, c1] = res3

        pickle.dump(results, open(f'results/{save_path}.pkl', 'wb'))

    if variant == 'flat':
        print('Executing variant flat')
        results = {}
        for c in cs:
            res = flat_Z_learning(c, **execution)
            results[c] = res

        pickle.dump(results, open(f'results/{save_path}.pkl', 'wb'))