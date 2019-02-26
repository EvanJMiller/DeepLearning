import numpy as np
from neural_network import NeuralNetwork

class AND():

    def __init__(self):

        self.sizes = [2, 1]
        self.nn = NeuralNetwork(self.sizes)

    def train(self):

        training_samples = []
        training_labels = []

        #Generate Training Data
        for i in range(0, 1000):
            sample = np.random.randint(low=0, high=2, size=(1, 2))

            if sample[0][0] == 1 and sample[0][1] == 1:

                sample_label = 1
            else:
                sample_label = 0

            training_samples.append(sample)
            training_labels.append(sample_label)

        for i in range(0, len(training_samples)):
                expected = self.nn.forward(training_samples[i])
                self.nn.backward(training_labels[i])
                # Updates to the parameters happens in the back prop call

    def infer(self, x1, x2):
        return self.nn.forward([x1, x2])

class OR():

    def __init__(self):

        self.sizes = [2, 1]
        self.nn = NeuralNetwork(self.sizes)

    def train(self):

        training_samples = []
        training_labels = []

        # Generate Training Data
        for i in range(0, 1000):
            sample = np.random.randint(low=0, high=2, size=(1, 2))

            if sample[0][0] == 1 or sample[0][1] == 1:

                sample_label = 1
            else:
                sample_label = 0

            training_samples.append(sample)
            training_labels.append(sample_label)

        for i in range(0, len(training_samples)):
            expected = self.nn.forward(training_samples[i])
            self.nn.backward(training_labels[i])
            # Updates to the parameters happens in the back prop call

    def infer(self, x1, x2):
        return self.nn.forward([x1, x2])

class NOT():

    def __init__(self):

        self.sizes = [1, 1]
        self.nn = NeuralNetwork(self.sizes)

    def train(self):

        training_samples = []
        training_labels = []

        # Generate Training Data
        for i in range(0, 1000):
            sample = np.random.randint(low=0, high=2, size=(1, 2))

            sample_label = not sample

            training_samples.append(sample)
            training_labels.append(sample_label)

        for i in range(0, len(training_samples)):
            expected = self.nn.forward(training_samples[i])
            self.nn.backward(training_labels[i])
            # Updates to the parameters happens in the back prop call

    def infer(self, x):
        return self.nn.forward(x)

class XOR():

    def __init__(self):
        self.sizes = [2, 2, 1]
        self.nn = NeuralNetwork(self.sizes)

    def train(self):

        training_samples = []
        training_labels = []

        for i in range(0, 1000):
            sample = np.random.randint(low=0, high=2, size=(1, 2))

            if ((sample[0][0] >= 1 and sample[0][1] <= 1) or (sample[0][0] <= 1 and sample[0][1] >= 1)):
                sample_label = 1
            else:
                sample_label = 0

            training_samples.append(sample)
            training_labels.append(sample_label)

        # Train
        for i in range(0, len(training_samples)/10):
            expected = self.nn.forward(training_samples[i])
            self.nn.backward(training_labels[i])

    def infer(self, x1, x2):
        return self.nn.forward([x1, x2])
