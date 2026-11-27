import numpy as np

# LOSS FUNCTIONS #####################################################
def mse_loss(y, y_pred):
    return np.mean((y-y_pred)**2)

def df_mse_loss(y, y_pred):
    return y_pred - y