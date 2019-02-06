#from __future__ import print_function as print
from neural_network import NeuralNetwork as NN
import torch
from logic_gates import AND, OR, NOT, XOR

if __name__ == "__main__":

    nn = NN([2, 1])
    and_gate = AND()
    or_gate = OR()
    not_gate = NOT()
    xor_gate = XOR()


    print(and_gate(False, False))
    print(and_gate(False, True))
    print(and_gate(True, False))
    print(and_gate(True, True))
    print(" ")

    print(or_gate(False, False))
    print(or_gate(False, True))
    print(or_gate(True, False))
    print(or_gate(True, True))
    print("")

    print(not_gate(True))
    print(not_gate(False))
    print("")

    print(xor_gate(False, False))
    print(xor_gate(False, True))
    print(xor_gate(True, False))
    print(xor_gate(True, True))






