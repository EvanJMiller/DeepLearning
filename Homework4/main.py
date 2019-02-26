

import torch
import torchvision
from neural_network import NeuralNetwork
import torch.optim as optim
import numpy as np
from nn_img2num import NnImg2Num
import nn_img2num
from my_img2num import MyImg2Num
import my_img2num

import matplotlib.pyplot as plt

from my_img2num import MyImg2Num

import torchvision.transforms as transforms

def to_one_hot(num):
    y = np.zeros(shape=(10, 1))
    y[num - 1][0] = 1
    print([num,y])
    return y

if __name__ == "__main__":

    print("")
    #mynet = MyImg2Num()
    #my_img2num.train(mynet)

    #net = NnImg2Num()
    #nn_img2num.train(net, 0.01, net.train_loader, 200)
    #nn_img2num.test(net, net.test_loader)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    # plt.show()

    # with open('training_losses.txt', 'r') as f:
    #     lines = f.readlines()
    #
    # rates = []
    # epochs = 0
    # for rate in lines:
    #     epochs += 1
    #     rates.append(float(rate))
    #
    # plt.plot(range(0, epochs), rates)
    #
    # plt.title('nnImage2num')
    # plt.xlabel('epochs')
    # plt.ylabel('average losses')
    # plt.savefig('losses.png')
    # plt.show()






