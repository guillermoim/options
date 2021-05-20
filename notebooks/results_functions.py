import numpy as np
import matplotlib.pyplot as plt
from lmdps.plotting import plot_as_matrix as plotmat
import pickle

def get_MAE_HL_errors(path, title, xlim, ylim=1, figsize=(7,7), plot_only=[], key=None):
    f = open(path,'rb')
    d = pickle.load(f)
    plt.close('all')
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    
    
    auxiliary_d = {}
    for k in d:
        auxiliary_d[k] = min(d[k][2])
        
    min_combination_c0_c1 = min(auxiliary_d, key=auxiliary_d.get)

    ax.set_title(f'{title}. Min.error with {min_combination_c0_c1}')

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(d)))))
        
    plotted = []
    for k in d:
        
        if plot_only and k not in plot_only:
            continue 
        else:
            Z, S, errors, errors_i, _ = d[k]
            ax.plot(errors, linewidth=1, markevery=25)
            
        plotted.append(k)

    ax.legend([f'c={key} min = {min(d[key][3]):1.3E}' for key in plotted], fontsize=6)
    ax.set_xlim((0,xlim))
    ax.set_ylim((0,ylim))
    ax.set_ylabel('error')
    ax.set_xlabel('n_samples')
    plt.show()
    

    
    if key is None:
        return d[min_combination_c0_c1]
    else:
        return d[key]
    
def get_MAE_plain_Z_learning(path, title, xlim, ylim):
    
    aux = pickle.load(open(path, 'rb'))
    res = {k:aux[k][1] for k in aux}
    
    fig, ax = plt.subplots(1,1)

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(res)))))
    
    
    st = {k:min(res[k]) for k in res}
    min_c = min(st, key=st.get)
    
    ax.set_title(f'{title}. Min.error with {min_c}')

    
    for k in res:

        errors = res[k]
        ax.plot(errors, linewidth=1, markevery=25)

    ax.legend([f'c={key} min = {min(res[key]):1.3E}' for key in res], fontsize=8)
    ax.set_xlim((0,xlim))
    ax.set_ylim((0,ylim))
    ax.set_ylabel('error')
    ax.set_xlabel('n_samples')
    
    
    return res[min_c]


def get_MAE_LL_errors(path, title, xlim, ylim=1, figsize=(7,4)):
    f = open(path,'rb')
    d = pickle.load(f)
    plt.close('all')
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
       
    ax.set_title(f'{title}')

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(d)))))
    
    for k in d:
        
        Z, S, errors, errors_i, _ = d[k]
                
        ax.plot(errors_i, linewidth=1, markevery=25)

    ax.legend([f'c={key} min = {min(d[key][3]):1.3E}' for key in d], fontsize=6)
    ax.set_xlim((0,xlim))
    ax.set_ylim((0,ylim))
    ax.set_ylabel('error')
    ax.set_xlabel('n_samples')

    plt.show()