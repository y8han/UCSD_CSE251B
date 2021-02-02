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
from utils import *

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

#Output layer gradient
def softmax_delta(x, Y, n):
    """
    Done: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    delta = (Y - x) / n
    return delta

    raise NotImplementedError("Softmax gradient not implemented")

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
        Done: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        # self.x record the value for a_j
        self.x = None
        self.z = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        Input: a_i
        Output: z_i
        """
        self.x = a #(without extra bias term)
        if self.activation_type == "sigmoid":
            self.z = np.insert(self.sigmoid(a), 0, 1, axis = 1)
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            self.z = np.insert(self.tanh(a), 0, 1, axis = 1)
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            self.z = np.insert(self.ReLU(a), 0, 1, axis = 1)
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            self.z = np.insert(self.leakyReLU(a), 0, 1, axis = 1)
            return self.leakyReLU(a)

    def backward(self, delta, n):
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

        return np.multiply(grad, delta) / n

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
        return np.where(x > 0, x, x * 0.1)
        raise NotImplementedError("leakyReLu not implemented")

    def grad_sigmoid(self):
        """
        Done: Compute the gradient for sigmoid here.
        """
        tmp = self.sigmoid(self.x)
        return np.multiply(tmp, 1 - tmp)
        raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self):
        """
        Done: Compute the gradient for tanh here.
        """
        tmp = 1 - np.tanh(self.x) ** 2
        return tmp
        raise NotImplementedError("tanh gradient not implemented")

    def grad_ReLU(self):
        """
        Done: Compute the gradient for ReLU here.
        """
        tmp = self.x
        tmp[np.where(tmp >= 0)] = 1
        tmp[np.where(tmp < 0)] = 0
        return tmp
        raise NotImplementedError("ReLU gradient not implemented")

    def grad_leakyReLU(self):
        """
        Done: Compute the gradient for leaky ReLU here.
        """
        tmp = self.x
        tmp[np.where(tmp > 0)] = 1
        tmp[np.where(tmp <= 0)] = 0.1
        return tmp
        raise NotImplementedError("leakyReLU gradient not implemented")


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = None    # Declare the Weight matrix
        self.b = None    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)
        #pass self.a to the activation class

        # self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        ## Initialize the weight matrix
        if activation == "ReLU" or activation == "leakyReLU":
            self.w = np.sqrt(2 / (in_units + out_units)) * np.random.randn(in_units, out_units)  #+1 for the bias
            self.b = np.sqrt(2 / (in_units + out_units)) * np.random.randn(1, out_units)  #+1 for the bias
        else:
            self.w = np.sqrt(2 / (in_units + out_units)) * np.random.randn(in_units, out_units)  #+1 for the bias
            self.b = np.sqrt(2 / (in_units + out_units)) * np.random.randn(1, out_units)  #+1 for the bias
        self.w = np.concatenate((self.b, self.w), axis = 0)

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Done: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        self.x = np.insert(x, 0, 1, axis = 1)  #for bias
        self.a = self.x @ self.w
        return self.a
        raise NotImplementedError("Layer forward pass not implemented.")

    def backward(self, delta, z, lammba):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        tmp = -z.T @ delta - lammba * 2 * self.w
        #add regularization term
        self.d_w = tmp[1:,:] #weights update
        self.d_b = tmp[0,:] #bias update
        return tmp
        raise NotImplementedError("Backprop for Layer not implemented.")

    def update(self, d_w, lr, gamma, d_w_previous, flag = False):  #Momentum update rule
        #delta -> regularization
        if d_w_previous is not None and flag:
            if d_w_previous != []:
                d_w = gamma * d_w_previous + (1 - gamma) * d_w
            else:
                d_w = (1 - gamma) * d_w
        self.w = self.w - lr * d_w

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
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1], config['activation']))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Done: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        layer + activation:
        layer1 + hiden units activation
        layer2 + hiden units activation
        ......
        last layer + softmax activation
        """
        self.x = x
        self.targets = targets
        tmp_x = x

        for i in range(len(self.layers)):
            tmp_x = self.layers[i].forward(tmp_x)

        forward_results= tmp_x
        self.y = softmax(forward_results)
        return self.y
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

    def backward(self, d_w_previous = None):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        forward_result = self.y
        X = self.x
        Y = self.targets
        delta = softmax_delta(forward_result, Y, self.miniBatchSize)
        X = np.insert(X, 0, 1, axis = 1)
        d_w_lists = []

        for i in range(len(self.layers)-1, -1, -1):
            if i % 2 == 0: #even layer -> layer object (update the weights)
                if i == 0:
                    d_w = self.layers[i].backward(delta, X, self.L2Penalty) #the first hidden layer
                    d_w_lists.append(d_w)
                else:
                    d_w = self.layers[i].backward(delta, self.layers[i - 1].z, self.L2Penalty)
                    d_w_lists.append(d_w)
            else: #odd layer -> activation object (compute delta)
                delta = self.layers[i].backward(delta @ (self.layers[i + 1].w[1:,:]).T, self.miniBatchSize)
        #update the weights
        update_index = 0
        for i in range(len(self.layers)-1, -1, -1):
            if i % 2 == 0:
                if d_w_previous is not None:
                    self.layers[i].update(d_w_lists[update_index], self.learningRate, self.momentumGamma, d_w_previous[update_index], self.momentum)
                else:
                    self.layers[i].update(d_w_lists[update_index], self.learningRate, self.momentumGamma, None, self.momentum)
                update_index += 1
        return d_w_lists
        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")


def AccuracyCompu(model, X_test, y_test):
    """
    Done: Calculate and return the accuracy on the test set.  Only forward
    """
    forward_result = model.forward(x = X_test, targets = None)
    loss = model.loss(logits = forward_result, targets = y_test)
    category = np.argmax(y_test, axis = 1)
    prediction = np.argmax(forward_result, axis = 1)
    correct = [1 if a == b else 0 for (a, b) in zip(category, prediction)] # compare prediction and actual y
    accuracy = sum(correct) / len(correct)
    return accuracy, loss

    raise NotImplementedError("Test method not implemented")


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Done: Train your model here.
    Implement batch SGD to train the model. -> batch SGD
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    Forward + Backward
    """
    W_lists = []
    Train_accuracy_lists = []
    Valid_accuracy_lists = []
    Train_loss_lists = []
    Valid_loss_lists = []
    train_accuracy, train_loss = AccuracyCompu(model, x_train, y_train)
    valid_accuracy, valid_loss = AccuracyCompu(model, x_valid, y_valid)
    Train_accuracy_lists.append(train_accuracy)
    Valid_accuracy_lists.append(valid_accuracy)
    Train_loss_lists.append(train_loss)
    Valid_loss_lists.append(valid_loss)
    d_w_lists = []
    w_ = []
    for i in range(int((len(model.layers) + 1) / 2)):
        d_w_lists.append([])
        w_.append(model.layers[2 * i].w)
    W_lists.append(w_)
    print("This is %d iteration"%(0))
    print("Accuracy of the training set: %f"%(train_accuracy))
    print("Accuracy of the validation set: %f"%(valid_accuracy))
    print("Loss of the training set: %f"%(train_loss))
    print("Loss of the validation set: %f"%(valid_loss))
    for i in range(model.epoches):
        idx = np.arange(0,x_train.shape[0])
        random.shuffle(idx)
        index = 0
        while index < x_train.shape[0]:
            if index + model.miniBatchSize <= x_train.shape[0]:
                x_train_batch = x_train[idx[index:index+model.miniBatchSize],:]
                y_train_batch = y_train[idx[index:index+model.miniBatchSize],:]
                index += model.miniBatchSize
            else:
                x_train_batch = x_train[idx[index:],:]
                y_train_batch = y_train[idx[index:],:]
                index = x_train.shape[0]
            forward_result = model.forward(x = x_train_batch, targets = y_train_batch)
            d_w_lists = model.backward(d_w_lists)  #backward -> update the weights
        # accuracy
        w_ = []
        for ii in range(int((len(model.layers) + 1) / 2)):
            w_.append(model.layers[2 * ii].w)
        W_lists.append(w_)
        train_accuracy, train_loss = AccuracyCompu(model, x_train, y_train)
        valid_accuracy, valid_loss = AccuracyCompu(model, x_valid, y_valid)
        Train_accuracy_lists.append(train_accuracy)
        Valid_accuracy_lists.append(valid_accuracy)
        Train_loss_lists.append(train_loss)
        Valid_loss_lists.append(valid_loss)
        print("This is %d iteration"%(i+1))
        print("Accuracy of the training set: %f"%(train_accuracy))
        print("Accuracy of the validation set: %f"%(valid_accuracy))
        print("Loss of the training set: %f"%(train_loss))
        print("Loss of the validation set: %f"%(valid_loss))
    #find the weights that give the best performance
    value, index = min((value, index) for (index, value) in enumerate(Valid_loss_lists))
    print(index)
    return Train_accuracy_lists, Valid_accuracy_lists, Train_loss_lists, Valid_loss_lists, W_lists[index]
    # raise NotImplementedError("Train method not implemented")

def ProblemB(x_train, y_train, config):
    #extract examples from train set (different categories)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    tmp = np.argmax(y_train, axis = 1)
    list = []
    epsilon = 0.01
    for i in range(10):
        list.append(np.where(tmp == i)[0][0])
    x_new = x_train[list]
    y_new = y_train[list]
    forward_result = model.forward(x = x_new, targets = y_new)
    d_w_lists = model.backward()

    #extract the weights update by backpropagation gradient
    Output_bias = d_w_lists[0][0][0] / model.miniBatchSize;#hidden -> Output
    Hidden_bias_1 = d_w_lists[1][0][0] / model.miniBatchSize#Hidden -> Hidden
    Hidden_bias_2 = d_w_lists[2][0][0] / model.miniBatchSize #Input -> Hidden
    Hidden_Output_Wight1 = d_w_lists[0][1][0] / model.miniBatchSize
    Hidden_Output_Wight2 = d_w_lists[0][2][0] / model.miniBatchSize
    Input_Hidden_weight1 = d_w_lists[2][1][0] / model.miniBatchSize
    Input_Hidden_weight2 = d_w_lists[2][2][0] / model.miniBatchSize

    #Numerical approximation
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[4].w[0][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[4].w[0][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Output_bias_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Output bias")
    print("BP: %f"%(Output_bias))
    print("Approxi: %f"%(Output_bias_numerical))
    print("error of output bias: %f"%(abs(Output_bias - Output_bias_numerical)))

    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[2].w[0][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[2].w[0][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Hidden_bias_1_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Hidden bias1")
    print("BP: %f"%(Hidden_bias_1))
    print("Approxi: %f"%(Hidden_bias_1_numerical))
    print("error of hidden bias1: %f"%(abs(Hidden_bias_1 - Hidden_bias_1_numerical)))

    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[0].w[0][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[0].w[0][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Hidden_bias_2_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Hidden bias2")
    print("BP: %f"%(Hidden_bias_2))
    print("Approxi: %f"%(Hidden_bias_2_numerical))
    print("error of hidden bias2: %f"%(abs(Hidden_bias_2 - Hidden_bias_2_numerical)))

    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[4].w[1][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[4].w[1][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Hidden_Output_Wight1_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Hidden Output Wight1")
    print("BP: %f"%(Hidden_Output_Wight1))
    print("Approxi: %f"%(Hidden_Output_Wight1_numerical))
    print("error of hidden to output weight1: %f"%(abs(Hidden_Output_Wight1 - Hidden_Output_Wight1_numerical)))

    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[4].w[2][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[4].w[2][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Hidden_Output_Wight2_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Hidden Output Wight2")
    print("BP: %f"%(Hidden_Output_Wight2))
    print("Approxi: %f"%(Hidden_Output_Wight2_numerical))
    print("error of hidden to output weight2: %f"%(abs(Hidden_Output_Wight2 - Hidden_Output_Wight2_numerical)))

    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[0].w[1][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[0].w[1][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Input_Hidden_weight1_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Input Hidden weight1")
    print("BP: %f"%(Input_Hidden_weight1))
    print("Approxi: %f"%(Input_Hidden_weight1_numerical))
    print("error of input to hidden weight1: %f"%(abs(Input_Hidden_weight1 - Input_Hidden_weight1_numerical)))

    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[0].w[2][0] += epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss1 = model.loss(forward_result, y_new)
    model  = Neuralnetwork(config)
    model.miniBatchSize = 10
    model.layers[0].w[2][0]-= epsilon
    forward_result = model.forward(x = x_new, targets = y_new)
    loss2 = model.loss(forward_result, y_new)
    Input_Hidden_weight2_numerical = (loss1 - loss2) / (2 * epsilon)
    print("Input Hidden weight2")
    print("BP: %f"%(Input_Hidden_weight2))
    print("Approxi: %f"%(Input_Hidden_weight2_numerical))
    print("error of input to hidden weight2: %f"%(abs(Input_Hidden_weight2 - Input_Hidden_weight2_numerical)))

if __name__ == "__main__":
    # Load the configuration.
    problemB = False #finish the problem b instead of running nn

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
    if problemB:
        ProblemB(x_train, y_train, config)
    # Done: train the model
    else:
        Train_accuracy_lists, Valid_accuracy_lists, Train_loss_lists, Valid_loss_lists, w_ = train(model, x_train, y_train, x_valid, y_valid, config)
        plotFunc(Train_accuracy_lists, Valid_accuracy_lists, config['activation'], "Accuracy", model.learningRate, model.momentumGamma, model.L2Penalty, True)
        plotFunc(Train_loss_lists, Valid_loss_lists, config['activation'], "Loss", model.learningRate, model.momentumGamma, model.L2Penalty, True)
        for i in range(int((len(model.layers) + 1) / 2)):
            model.layers[2 * i].w = w_[i]
        test_acc, test_loss = AccuracyCompu(model, x_test, y_test)
        print("Accuracy of the test set: %f"%(test_acc))
        print("Loss of the test set: %f"%(test_loss))
        plt.show()

    # TODO: Plots
    # plt.plot(...)
