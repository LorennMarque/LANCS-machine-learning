import numpy as np
from src.initiations import *
from src.activations import *

# MLP class

class MLP:
    # For this MLP i followed the steps in Wk3-Lab instructions pdf
    def __init__(self, loss_function, hidden_layer_nodes):
        self.loss_function = loss_function
        self.hidden_layer_nodes = hidden_layer_nodes

    def train(self, x, Y, learning_rate=0.5, max_iter = 30, initialization = glorot_init, activation=relu):
        # NOTE: This only work for a single hidden layer ===============

        # Initialize the training by specifying:
        # a. The number of input features from training data
        n_input_nodes = x.shape[1] # Input nodes / features

        # b. The number of nodes for the hidden layer
        # self.hidden_layer_nodes
        # As a list. i.e: [2,5] would mean first hidden layer 2 nodes, second 5

        # c. The number of outputs (From Y values)
        output_values = Y.unique().tolist()
        print(f"Output values: {output_values}")


        # d. The learning rate (parameter with 0.5 by default)
        # e. Number of iterations (parameter with 30 by default)

        # f. The set of weigths going from input layer.
            # Matrix m x n
            # number of input nodes x number of hidden nodes (first layer)
        n_hidden = self.hidden_layer_nodes[0]

        weights_input2hidden = np.zeros((n_input_nodes, n_hidden))

        for row in range(n_input_nodes):
            for col in range(n_hidden):
                weights_input2hidden[row, col] = initialization(n_input_nodes, n_hidden)

        # g. The set of weights going from hidden to output layer
        weights_hidden2output = np.zeros((n_hidden, len(output_values)))

        for row in range(n_hidden):
            for col in range(len(output_values)):
                weights_hidden2output[row, col] = initialization(n_hidden, len(output_values))

        # LOOP START ==========================================================
        for epoch in range(1, max_iter):
            print(f"Epoch: {epoch}")

        # a. Foward propagation
            for row, y_act in zip(x, Y):
                print("===========================")
                print(f"Weights: {weights_input2hidden}")
                print(f"Row: {row}")
                print("===========================")
                z = row @ weights_input2hidden
                print(f"Multiplication w@row: {z}")

                print(f"=== ACTIVATION ===")
                print(f"Activación: {activation(z)}")
                v = activation(z)

                print("=== OUTPUT LAYER ===")
                z_2 = v @ weights_hidden2output

                y_pred = activation(z_2)
                print(f"Final result: {y_pred}")

            # b. Calculate the loss
                loss = self.loss_function(y_act, y_pred)
                print(f"Loss {loss}")

            # Backpropagate the error
            