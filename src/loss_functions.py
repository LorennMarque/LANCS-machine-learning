import numpy as np

# LOSS FUNCTIONS #####################################################
def mse_loss(y, y_pred):
    return np.mean(y-y_pred)**2