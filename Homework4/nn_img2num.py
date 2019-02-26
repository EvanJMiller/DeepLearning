import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import neural_network
import torch.nn.functional as F
import numpy as np

import torch.optim as optim

import numpy
#import any other packages you might need from pytorch

class NnImg2Num(nn.Module):

    def __init__(self):

        super(NnImg2Num, self).__init__() #initialize super class

        ### Define your model here in pytorch. We reccomend two hidden layers. You should do this homework only with linear layers as explained in class.
        ### Remember the output should be one hot encoding , this means the dimension of your output should be the same as the number of classes

        self.batch_size = 64
        self.test_batch_size = 1000
        self.log_interval = 10
        self.learning_rate = 0.1

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

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
        x = x.view(-1, 784)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x)

    def to_one_hot(self, arr):


        cols = len(arr)
        out = torch.zeros(cols, 10)

        #print(arr)

        for i in range(0, cols):
            y = np.zeros(shape=(10, 1))
            out[i][arr[i]] = 1
            #out[i] = (torch.tensor(y).float())

        return out.float()


def train(net, learning_rate, train_loader, epochs):

        epoch = 0
        log_interval = 10
        max_epochs = 100
        stopping_criteria = False

        training_rates = []
        losses = []

        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        net.train()

        for i in range(0, epochs):

            tr = test(net, net.test_loader)
            with open('training_rates.txt', 'a') as f:
                f.write("{}\n".format(tr))

            training_rates.append(tr)

            for batch_idx, (data, target) in enumerate(train_loader):

                # print(data.shape)
                #             # print(target)

                optimizer.zero_grad()  # zero the gradient buffers

                y = net.to_one_hot(target)

                output = net(data) #forward through the net
                loss = criterion(output, y) #calculate the loss

                loss.backward() # back prop
                losses.append(loss.item())
                optimizer.step()  # Does the update


                #if batch_idx % log_interval == 0:
                    #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #    i, batch_idx * len(data), len(train_loader.dataset),
                    #           100. * batch_idx / len(train_loader), loss.item()))
            with open('training_losses.txt', 'a') as f:
                avg = sum(losses) / len(losses)
                f.write("{}\n".format(avg))
                print("Avg loss: {}".format(avg))
                losses = []  # Clear losses list


def test(net, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    samples = 0
    with torch.no_grad():

        for data, target in test_loader:
            t = target.numpy()
            #for i in range(0, 1000):

            output = net(data)

            #test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            samples += len(data)

        per = 100 * float(correct)/samples
        print('Accuracy: {}/{} - {})'.format(correct, samples, per))
        return per