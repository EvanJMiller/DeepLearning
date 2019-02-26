from neural_network import NeuralNetwork
import numpy as np
import copy
from pprint import pprint as pp
from random import shuffle



def calculate_mse(expected_output, y):
    mse = np.sum(np.square(np.subtract(y, expected_output)))
    return mse

if __name__ == "__main__":

    sizes = [2,2, 1]

    nn = NeuralNetwork(sizes)

    inputs = [np.float32([[0.0, 0.0]]), np.float32([[0.0, 1.0]]), np.float32([[1.0, 0.0]]), np.float32([[1.0, 1.0]])]
    labels = [np.float32([[0.0]]), np.float32([[1.0]]), np.float32([[1.0]]), np.float32([[1.0]])]

    training_samples = []
    training_labels = []

    for i in range(0, 1000):
        sample = np.random.randint(low=0, high=2, size=(1,2))

        if((sample[0][0] >= 1 and sample[0][1] <= 1) or (sample[0][0] <= 1 and sample[0][1] >= 1)):
        #if (sample[0][0] == 1 and sample[0][1] == 1):
            sample_label = 1
        else:
            sample_label = 0

        training_samples.append(sample)
        training_labels.append(sample_label)


    test_samples = []
    test_labels = []

    for i in range(0, 10000):
        test_sample = np.random.randint(low=0, high=2, size=(1, 2))
        if ((test_sample[0][0] >= 1 and test_sample[0][1] <= 1) or (test_sample[0][0] <= 1 and test_sample[0][1] >= 1)):
        #if (test_sample[0][0] == 1 and test_sample[0][1] == 1):
            test_label = float(1)
        else:
            test_label = 0

        test_samples.append(test_sample)
        test_labels.append(test_label)


    # Train
    for i in range(0, len(training_samples)):
        expected = nn.forward(training_samples[i])
        nn.backward(training_labels[i])


    # # Accuracy
    correct = 0
    incorrect = 0
    for i in range(0, len(test_samples)):

        output = nn.forward(test_samples[i])
        #pp(output)

        if(output > 0.5):
            expected = 1
        else:
            expected = 0

        if(expected == test_labels[i]):
            correct = correct + 1
        else:
            incorrect = incorrect + 1

        #print([test_samples[i], output, expected, test_labels[i]])

    print(correct / (correct + incorrect))












