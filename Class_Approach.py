import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # print("Weights = ", self.weights)
        self.biases = np.zeros(n_neurons)
        # print("Biases = ", self.biases)


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # print("self.output = ", self.output)
        

input_data = [[1, 2, 3, 2.5],           # Sample Input Data
              [2.0, 5.0, 1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

'''
This current approach requires us to code the entire network in 2 phases.

Phase 1 consists of creating the layer structure where we will be creating each layer and also defining
the number of neurons for the respective layers.

Phase 2 involves the chaining of the layers one after another to perform the Forward Propagation of inputs 
in the neural network.
'''

layer1 = Layer_Dense(len(input_data[0]), len(input_data))       # Phase 1
layer2 = Layer_Dense(len(input_data), 10)
layer3 = Layer_Dense(10, 15)
layer4 = Layer_Dense(15, 1)

layer1.forward(input_data)                                      # Phase 2
layer2.forward(layer1.output)
layer3.forward(layer2.output)
layer4.forward(layer3.output)

print("Output = ", layer4.output)

'''
The Activation function is to be added to these layers. Also, Back propagation is also to be introduced into
these layers for better model creation.

To be added Layers
    - Flatten Layer
    - Dropout Layer

'''