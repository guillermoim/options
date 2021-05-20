import numpy as np

def __init__():
    pass

def error_metric(Z_true, Z_hat, n_non_terminal):
    V_true = np.log(Z_true[:n_non_terminal])
    V_hat = np.log(Z_hat[:n_non_terminal])

    return (np.abs(V_true - V_hat)).mean()