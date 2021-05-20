import time
from lmdps import lmdps
import numpy as np
import os
import pandas as pd
from lmdps import hierarchical_gridworld as HGW
from lmdps.plotting import plot_as_matrix


def main():

    df = pd.DataFrame(columns=['rooms', 'time_H', 'time_sparse',
                               'diff_H_sparse', 'diff_VH_Vsparse'
                               ])

    experiments = [((2, 2), [(0, 1)], (2,3), (15,15)),
                   ((3, 3), [(0, 2)], (2,3), (15,15)),
                   ((4, 4), [(0, 3)], (2,3), (15,15)),
                   ((5, 5), [(0, 4)], (2,3), (17,17)),
                   ((6, 6), [(0, 5)], (2,3), (25,25)),
                   ((7, 7), [(0, 6)], (2,3), (27,27)),
                   ((8, 8), [(0, 7)], (2,3), (27,27)),
                   ((9, 9), [(0, 8)], (2,3), (29,29)),
                   ((10, 10), [(0, 9)], (2,3), (29,29)),
                   ]



    R_DIM = 5
    N_iter = 10000  # 10k

    times_H = {}
    times_dense = {}
    times_sparse = {}

    os.makedirs('results', exist_ok=True)

    for idx, (rooms, goals, goal_pos, figsize) in enumerate(experiments):

        print('Computing', rooms)

        x, y = rooms

        walls = 2 * 4 + 2*(x-2)+2*(y-2)

        # Hierarchical execution
        start_H = time.time()
        _, Z_H, _ = HGW.hierarchical_solver(rooms, R_DIM, goal_pos, goals, N_iter)
        end_H = time.time()
        time_H = end_H - start_H

        # Sparse and dense execution
        # Creation
        start_creation_time = time.time()
        _, P, G, z = HGW.create_flat_MDP(rooms, R_DIM, goal_pos, goals)
        end_creation_time = time.time()

        creation_time = end_creation_time - start_creation_time

        # Sparse
        start_sparse = time.time()
        z = lmdps.power_method(P, G, n_iter=N_iter)
        end_sparse = time.time()

        Z_sparse = z[:x*y*(R_DIM**2)].reshape((x*R_DIM, y*R_DIM))

        time_sparse = creation_time + (end_sparse - start_sparse)

        times_H[rooms] = time_H
        times_sparse[rooms] = time_sparse

        V_H = np.log(Z_H)
        V_sparse = np.log(Z_sparse.todense())

        diff_VH_Vsparse = np.abs(V_H - V_sparse).mean()

        diff_H_sparse = np.abs(Z_H - Z_sparse).mean()

        os.makedirs(f'results/{x}_{y}', exist_ok=True)

        np.save(f'results/{x}_{y}/z_H.npy', Z_H)
        np.save(f'results/{x}_{y}/z_S.npy', Z_sparse)

        df = df.append({'rooms': f'{x}_{y}', 'time_H': time_H, 'time_sparse': time_sparse,
                        'diff_H_sparse': diff_H_sparse, 'diff_VH_Vsparse': diff_VH_Vsparse
                   }, ignore_index=True)

        fig_dir = f'results/pictures/{x}_{y}'

        os.makedirs(fig_dir, exist_ok=True)

        plot_as_matrix(V_sparse, f'Hier. V function interior grid {rooms} rooms',
                       figsize=figsize,
                       save_path=f'{fig_dir}/V_sparse')

        plot_as_matrix(V_sparse, f'Sparse flat V function interior grid {rooms} rooms',
                       figsize=figsize,
                       save_path=f'{fig_dir}/V_hierarchical')

    df.to_csv('results/results.csv', index=False)


if __name__ == '__main__':
    main()
