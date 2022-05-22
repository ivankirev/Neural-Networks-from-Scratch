# Neural-Network-from-Scratch

This is an implementation of a Multi-Layer perceptron , i.e. a feed-forward neural network, without the use of TensorFlow. The architecture is as follows:

The input layer is followed by 5 hidden layers, each with 400 neurons, after which there is an output layer with 10 neurons (for each class). The activation function between all layers is LeakyRelu with slope 0.01, and for the output layer the softmax function is used as the activation function. The optimisation method is stochastic gradient descent and the loss function is categorical cross-entropy. 

