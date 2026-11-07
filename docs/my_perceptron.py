import numpy as np

from activations import *

class MyPerceptron:
    def __init__(self, activation_function):
        self.weights = None
        self.bias = 0
        self.activation_function = activation_function

    def train(self, x, Y, learning_rate=0.5):

            # Convert inputs to NumPy arrays
            x = np.asarray(x, dtype=float)
            Y = np.asarray(Y).ravel()

            # Initialize weights (one per feature)
            self.weights = np.zeros(x.shape[1])

            for epoch in range(100):
                # Iterate over observations
                for row, Y_val in zip(x, Y):

                    # Forward propagation
                    v = row @ self.weights + self.bias

                    # Calculate prediction
                    estimated_y = self.activation_function(v)

                    # Calculate error
                    sigma = Y_val - estimated_y

                    # Update parameters
                    self.weights += learning_rate * sigma * row
                    self.bias += learning_rate * sigma
    
    def predict(self, x):
        x = np.asarray(x, dtype=float)
        predictions = []

        for row in x:
            v = row @ self.weights + self.bias
            prediction = self.activation_function(v)
            predictions.append(prediction)

        return np.array(predictions)


class MyMLP:
    def __init__(self, loss_function, hidden_layers):
        self.loss_function = loss_function
        self.hidden_layers = hidden_layers
