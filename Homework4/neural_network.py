import numpy as np
import torch.nn as nn
import torch
import math
import torch.optim as optim

class NeuralNetwork():

    def __init__(self, sizes):

        self.sizes = sizes
        self.num_layers = len(sizes) - 1

        self.theta = self.init_theta()

        self.output_layers = []
        self.gradients = []

        # randomly initalize your weights and biases
        self.eta = .001

    def init_theta(self):

        weights = []

        for i in range(1, len(self.sizes)):
            rows = self.sizes[i - 1] + 1  # Add 1 for the bias term
            cols = self.sizes[i]

            layer = 2 * np.random.random(size=(rows, cols)) - 1
            weights.append(layer)

        return weights

    def backward(self, y):  # y is the target output

        # Get the output from the forward pass
        output_layer = self.output_layers[-1]
        hidden_layer = self.theta[-1]

        # Array to store gradients
        self.gradients = []

        # Gradient of the error w.r.t. the output - actual output minus expected
        de_do = np.subtract(output_layer, y)

        # Vectorize the sigmoid gradient function - output0 * (1 - output0)
        sg = np.vectorize(self.sigmoid_gradient)

        # Calculate sigmoid gradient using the output layer
        do_dn = sg(output_layer)

        # Bias term array to be appended
        one = np.array([[1]])

        # Last Output layer with bias term
        previous_out = np.concatenate((one, self.output_layers[-2]), axis=1)

        # Get the shape of the last output layer
        [row, col] = self.theta[-1].shape

        # Create matrices for the gradient and partial errors
        gradient = np.zeros(shape=(row, col))
        partial_errors = np.zeros(shape=(row, col))

        for c in range(0, col):
            for r in range(0, row):
                gradient[r][c] = previous_out[0][r] * de_do[0][c] * do_dn[0][c]
                partial_errors[r][c] = hidden_layer[r][c] * de_do[0][c] * do_dn[0][c]

        er, ec = previous_out.shape
        dE_dTheta = np.zeros(shape=(er, ec))
        for c in range(0, col):
            for r in range(0, row):
                dE_dTheta[0][r] += partial_errors[r][c]

        self.theta[-1] = self.theta[-1] - self.eta * gradient

        self.gradients.append(gradient)

        bias_error = dE_dTheta[0][0]

        # Cut out the bias error for back prop
        dE_dTheta = dE_dTheta[:, 1:]

        for i in range(len(self.output_layers) - 2, 0, -1):


            # Last Output layer with bias term
            previous_out = np.concatenate((one, self.output_layers[i - 1]), axis=1)

            # Hidden Layer
            hidden_layer = self.theta[i - 1]


            # new matrix for the partial errors
            pr, pc = hidden_layer.shape
            partial_errors = np.zeros(shape=(pr, pc + 1), dtype=float)

            # partial error sigmoid
            de_dout = sg(self.output_layers[i])

            # new matrix for the gradient
            gradient = np.ones(shape=hidden_layer.shape)

            [row, col] = hidden_layer.shape

            # Calculate the gradient matrix
            for c in range(0, col):
                for r in range(0, row):
                    gradient[r][c] = previous_out[0][r] * de_dout[0][c] * dE_dTheta[0][c]
                    partial_errors[r][c] = hidden_layer[r][c] * dE_dTheta[0][c] * de_dout[0][c]

            # Update layers using the gradient
            self.gradients.append(gradient)

            # Update the params
            self.theta[i - 1] = self.theta[i - 1] - self.eta * gradient

            dE_dTheta = np.zeros(previous_out.shape)
            for c in range(0, col):
                for r in range(0, row):
                    # print([r, c, partial_errors[r][c]])
                    dE_dTheta[0][c] += partial_errors[r][c]

            dE_dTheta = dE_dTheta[:, 1:]

    def calculate_mse(self, y):
        mse = np.sum(np.square(np.subtract(y, self.output_layers[-1])))
        return mse

    # NOTE: Update params shown but not used in back prop
    def updateParams(self, eta):
        # update weights based on your learning rate eta
        for i in range(0, len(self.theta)):
            self.theta[i] = self.theta[i] - eta * self.gradient[-i]

    def getLayer(self, index):
        # return requested weights
        return self.theta[index]

    def setLayer(self, index, layer):
        # print('--layer index: {} \n'.format(index))
        self.theta[index] = layer

    def sigmoid(self, x):
        if (x < 100 and x > -100):
            return 1 / (1 + math.exp(-x))
        elif (x > 100):
            return 1 / (1 + math.exp(-100))
        elif (x < -100):
            return 1 / (1 + math.exp(100))

    def sigmoid_gradient(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):

        # Reset the output layers
        self.output_layers = []

        # Add a bias term
        bias = np.array([[1]])

        x_bias = np.concatenate([bias, x.T])

        sigmoid_v = np.vectorize(self.sigmoid)

        out_layer = sigmoid_v(np.dot(x_bias.T, self.theta[0]))

        self.output_layers.append(x)
        # Append the first out layer
        self.output_layers.append(out_layer)

        # Perform additional passes
        for layer in self.theta[1:]:
            # Form a computed layer with the previous input and bias term of 1 (not bias weight)
            in_layer = np.concatenate((bias, out_layer.T), 0)

            # Perform dot product followed by sigmoid function to compute output of layer i
            out_layer = sigmoid_v(np.dot(in_layer.T, layer))
            # Keep track of computational layers
            self.output_layers.append(out_layer)

        # Used in HW3
        # row, col = out_layer.shape
        # for r in range(0, row):
        #     for c in range(0, col):
        #         if (out_layer[r][c] > 0.5):
        #             self.output_layers[-1][r][c] = 1
        #         else:
        #             self.output_layers[-1][r][c] = 0

        return out_layer







