import numpy as np

def relu(v):
    v = np.asarray(v)
    return np.where(v <= 0, 0, v)

def heaviside_step(v):
    v = np.asarray(v)
    return np.where(v < 0, 0, 1)

def sigmoid(v):
    v = np.asarray(v)
    return 1 / (1 + np.exp(-v))

def tanh(v):
    v = np.asarray(v)
    return (np.exp(v) - np.exp(-v)) / (np.exp(v) + np.exp(-v))

def silu(v):
    # Sigmoid linear unit
    v = np.asarray(v)
    return v * sigmoid(v)

# derivatives versions #####################################################################

def derivative_relu(v):
    v = np.asarray(v)
    # Incluso si no esta determinada en 0, asignamos v = 0 para permitir backpropagation
    return np.where(v > 0, 1, 0)

def derivative_heaviside(v):
    # No tiene pendiente, es constante en ambas partes
    # No esta definida en 0. No nos sirve para backpropagation.
    return 0

def derivative_sigmoid(v):
    s = sigmoid(v)
    return s * (1 - s)


def derivative_tanh(v):
    v = np.asarray(v)
    t = tanh(v)
    return 1 - t ** 2

def derivative_silu(v):
    v = np.asarray(v)
    s = sigmoid(v)
    return s + v * s * (1 - s)