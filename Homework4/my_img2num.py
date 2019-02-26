import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from neural_network import NeuralNetwork


#import any other packages you might need from pytorch

class MyImg2Num():

    def __init__(self):
        ### Define your model here as in HW3. We reccomend two hidden layers. You should do this homework only with linear layers as explained in class.
        ### Remember the output should be one hot encoding , this means the dimension of your output should be the same as the number of classes

        self.batch_size = 1
        self.test_batch_size = 1
        self.max_epochs = 1

        # Define the network for the model
        self.sizes = [784, 16, 16, 10]
        self.nn = NeuralNetwork(sizes=self.sizes)

        ### Since we have not explained the pytorch dataloader in class we are giving you an example here for the mnist dataset.
        ### Please read the pytorch documentation on dataloaders.

        self.train_loader = torch.utils.data.DataLoader(
           torchvision.datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
            batch_size=self.batch_size, shuffle=True)

        # Load test data
        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor()
                           ])),
        batch_size=self.test_batch_size, shuffle=False)



    def forward(self, x):
        ###Define your forward pass with your NeuralNetwork class here.
        self.nn.forward(x)

    def get_stopping_criteria(self):
        return True

    def to_one_hot(self, num):

        y = np.zeros(shape=(10, 1))
        y[num - 1][0] = 1
        return y

    def calculate_mse(self, layer, y):
        mse = np.sum(np.square(np.subtract(y, layer)))
        return mse

def train(net):


    epochs = 100
    counter = 1
    train_loader = net.train_loader
    correct = 0

    for epoch in range(0, epochs):
        # iterate over whole dataset
        for batch, (data, target) in enumerate(train_loader):

            #print(data.shape)

            # NeuralNetwork forward pass
            net.forward(data.view(-1, 784).numpy())

            # NeuralNetwork backward pass
            y = net.to_one_hot(target)
            net.nn.backward(y)

            # Get the expected output
            [m, loc] = torch.max(torch.tensor(net.nn.output_layers[-1]), 1)

            #Compare
            if(loc.item() == target):
                correct = correct + 1

            if(counter % 50 == 0):

                mse = net.calculate_mse(y, net.nn.output_layers[-1])
                [m, loc] = torch.max(torch.tensor(net.nn.output_layers[-1]), 1)
                print("epoch {}, iteration {}, mse: {}, accuracy: {}".format(epoch, counter, mse, correct/50))

                correct = 0

            counter += 1

    epochs = epochs + 1

def test(net):
    pass
