import numpy as np

# What: 
# Different initalization methods

# So what:
# Used to prevent initial weigths from being too large or too small
# If not set correctly the values could get huge or shrink toward zero
    # Exploding or vanishing gradients

# Terms:
# n_inputs. Number of inputs to a layer
# n_outputs. Number of neurons in that layer

def glorot_init(n_inputs, n_outputs):
    print("Glorot Initalization")

    mean = 0
    s_deviation = (2 / (n_inputs + n_outputs))

    return np.random.normal(mean, s_deviation)


def he_init(n_inputs, n_outputs):
    # Designed for RELU related activations
    print("He Initalization")

    mean = 0
    s_deviation = (2 / (n_inputs))

    return np.random.normal(mean, s_deviation)



def lecun_init(n_inputs, n_outputs):
    print("LeCun Initalization")

    mean = 0
    s_deviation = (1 / (n_inputs))

    return np.random.normal(mean, s_deviation)
