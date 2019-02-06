import torch
from neural_network import NeuralNetwork as NN

class AND():

    def __init__(self):	
        # appropriately initialize your neural network
        self.model = NN([2, 1])

    def __call__(self, x1, x2):
        # process boolean inputs and send them to your neural network for a forward pass
        x = torch.Tensor([[int(x1)], [int(x2)]])
        weights = torch.Tensor([[-25], [15], [15]])
        self.model.setLayer(0, weights)
        return self.model.forward(x)

class OR():

    def __init__(self):	
        pass
        # appropriately initialize your neural network
        self.model = NN([2, 1])

    def __call__(self, x1, x2):
        # process boolean inputs and send them to your neural network for a forward pass
        x = torch.Tensor([[int(x1)], [int(x2)]])
        weights = torch.Tensor([[-10], [15], [15]])
        self.model.setLayer(0, weights)
        return self.model.forward(x)


class NOT():

    def __init__(self):	
        # appropriately initialize your neural network
        self.model = NN([1, 1])
    def __call__(self, x1):
        # process boolean inputs and send them to your neural network for a forward pass
        x = torch.Tensor([[int(x1)]])
        weights = torch.Tensor([[10], [-20]])
        self.model.setLayer(0, weights)

        return self.model.forward(x)

class XOR():

    def __init__(self):	
        # appropriately initialize your neural network
        self.model = NN([2, 2, 1])

    def __call__(self, x1, x2):
        # process boolean inputs and send them to your neural network for a forward pass
        x = torch.Tensor([[int(x1)],  [int(x2)]])

        layer1 = torch.Tensor([[25,-10], [-15, 15], [-15, 15]])
        layer2 = torch.Tensor([[-25], [15], [15]])

        self.model.setLayer(0, layer1)
        self.model.setLayer(1, layer2)

        return self.model.forward(x)

