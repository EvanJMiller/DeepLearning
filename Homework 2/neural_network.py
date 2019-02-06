import torch
import math
import numpy as np
import torch

class NeuralNetwork():

    def __init__(self, sizes):

        self.in_size = sizes[0] #The input layer is first
        self.sizes = sizes
        self.out_size = sizes[-1] #The output layer is last
        self.hidden_layer_sizes = sizes[1:len(sizes)-1] #hidden layer sizes
        self.hidden_layers = []

        for i in range(1, len(self.sizes)):

            m = sizes[i-1] + 1
            n = sizes[i]

            sigma = 1 / pow(m, 0.5)
            sample = np.random.normal(0, sigma, m*n)

            layer = torch.Tensor(sample.reshape(m, n))

            self.hidden_layers.append(layer)


    def getLayers(self):
        return self.hidden_layers

    def getLayer(self, index):
        # return requested weights
        return self.hidden_layers[index]

    def setLayer(self,index,  layer):
        self.hidden_layers[index] = layer

    def sigmoid(self,x):
        return torch.sigmoid(x)


    def forward(self, x):

        comp_layers = []

        bias = torch.Tensor([[1]])
        # Perform forward pass of input layer

        x_bias = torch.cat((bias, x), 0)

        comp_layer = torch.Tensor(np.dot(torch.t(x_bias), self.hidden_layers[0]))

        comp_layer = self.sigmoid(comp_layer)

        i = 2

        # Perform additional passes
        for layer in self.hidden_layers[1:]:

            comp_layer = torch.cat((bias, torch.t(comp_layer)), 0)

            comp_layer = self.sigmoid(torch.Tensor(np.dot(torch.t(comp_layer), layer)))

            comp_layers.append(comp_layer)

            i += 1

        return comp_layer





