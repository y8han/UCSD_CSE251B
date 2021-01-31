################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2021
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
import random

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path + 'config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    Done: Normalize your inputs here to have 0 mean and unit variance.
    """
    inp = (inp.T - np.mean(inp, axis = 1)) / np.std(inp, axis = 1)
    inp = inp.T
    return inp


def one_hot_encoding(labels, num_classes=10):
    """
    Done: Encode labels using one hot encoding and return them.
    """
    new_labels = np.zeros([labels.size, labels.max() + 1])
    new_labels[np.arange(labels.size),labels] = 1
    return new_labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    piece of train data is separatd for validation
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


#Output layer (always softmax)
#Hiden layer -> try different activation functions defined in the class
def softmax(x):
    """
    Done: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    return (np.exp(x).T / np.sum(np.exp(x), axis = 1)).T
    raise NotImplementedError("Softmax not implemented")


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers (hidden layers).

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Done: Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-x))
        raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        Done: Implement tanh here.
        """
        return np.tanh(x)
        raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        Done: Implement ReLU here.
        """
        return x * (x > 0)
        raise NotImplementedError("ReLu not implemented")

    def leakyReLU(self, x):
        """
        Done: Implement leaky ReLU here.
        """
        return np.where(x > 0, x, x * 0.01)
        raise NotImplementedError("leakyReLu not implemented")

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        raise NotImplementedError("tanh gradient not implemented")

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        raise NotImplementedError("ReLU gradient not implemented")

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        raise NotImplementedError("leakyReLU gradient not implemented")


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = None    # Declare the Weight matrix
        self.b = None    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)
        #pass self.a to the activation class

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        raise NotImplementedError("Layer forward pass not implemented.")

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.  Layer objects
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.miniBatchSize = config['batch_size']
        self.learningRate = config['learning_rate']
        self.earlyStop = config['early_stop']
        self.epoches = config['epochs']
        self.earlyStopEpoch = config['early_stop_epoch']
        self.L2Penalty = config['L2_penalty']
        self.momentum = config['momentum']  #flag variable
        self.momentumGamma = config['momentum_gamma']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """

        return forward_results
        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        Done: compute the categorical cross-entropy loss and return it.
        This corss-entropy term is derived in PA1.
        '''
        loss = -np.multiply(targets, np.log(logits)) #multiply in an element-wise way
        loss = np.sum(loss) / (loss.shape[0] * loss.shape[1]) #average over the number of training examples and the number of categories
        return loss
        raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")




def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model. -> batch SGD
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    Forward + Backward
    """

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.  Only forward
    """




    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("../")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data and split the validation set
    x_train, y_train = load_data(path="../Data/", mode="train")

    #create a valibration split (10% from the total)
    idx = np.arange(0,x_train.shape[0])
    cut_valid = int(x_train.shape[0] / 10)
    random.shuffle(idx)
    #Obtain the validation set
    x_valid = x_train[0:cut_valid,:]
    y_valid = y_train[0:cut_valid,:]

    #Obtain the train set
    x_train = x_train[cut_valid:, :]
    y_train = y_train[cut_valid:,:]

    x_test,  y_test  = load_data(path="../Data/", mode="t10k")

    # TODO: train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)

    # TODO: Plots
    # plt.plot(...)
