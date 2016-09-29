import numpy as np
import scipy.special

# Neural Network class
class neuralNetwork:

    # initialise the network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set the number of nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # set the learning rate
        self.lr = learningrate

        # link weight matrix (set around a normal distribution)
        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # define the activation function (here a sigmoid function)
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the network
    def train(self, inputs_list, targets_list):
        # create a 2D array with inputs and targets
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate signals arriving into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the hidden output
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals arriving into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the final outputs
        final_outputs = self.activation_function(final_inputs)

        # Calculate output errors (target - actual)
        output_errors = targets - final_outputs

        # Calculate hidden errors
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weight between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))

        # Update the weight between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
        pass

    # query the network
    def query(self, inputs_list):
        # create a 2D array with inputs
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate signals arriving into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the hidden output
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals arriving into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the final outputs
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# test network
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

n.query([1.0, 0.5, -0.5])
