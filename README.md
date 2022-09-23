The project includes an implementation of a neural network (using only Numpy), Convolutional neural networks (CNN), and CNN with dropout. The aim is to explore the effects of the different techniques and understand how the models work in more detail. 

# Dataset

We explore how two different neural network architectures perform on a supervised classification task on the Fashion-MNIST dataset of images. This data set consists of two files: a training set with 60,000 images and a test set with 10,000 images belonging to 10 classes of items sold by a fashion online store, e.g., T-shirt/top, Trouser, Pullover, Dress, etc. The Fashion-MNIST dataset is well balanced, i.e., it has 6,000 images of each class in the training set and 1,000 images of each class in the test set.

# Neural Network from Scratch

This is an implementation of a Multi-Layer perceptron , i.e. a feed-forward neural network, without the use of TensorFlow. The architecture is as follows:

The input layer is followed by 5 hidden layers, each with 400 neurons, after which there is an output layer with 10 neurons (for each class). The activation function between all layers is LeakyRelu with slope 0.01, and for the output layer the softmax function is used as the activation function. The optimisation method is stochastic gradient descent and the loss function is categorical cross-entropy. 

# Convolutional Neural Network

Using TensorFlow, we implement a convolutional neural network according to the following architecture:

The input layer is followed by 5 hidden layers in total (of which the first 4 are convolutional layers and the last is a fully-connected layer), followed by the output layer with 10 neurons (one for each class). Regarding the hidden layers: all convolutional layers apply $3 \times 3$ filters, but the first two use 8 feature maps and the last two use 16 feature maps (also called ‘channels’). The last convolutional layer is followed by a $2 \times 2$ maximum pooling layer. The fully-connected layer has 64 neurons, and is followed by the output layer with 10 neurons. The activation function is the LeakyReLU(x) with a slope of 0.01 for x < 0 between all layers, and the softmax function is the activation function on the output layer. The optimisation method is stochastic gradient descent (SGD), and the loss function is (categorical) cross-entropy.

## Dropout in the fully connected layer

To reduce overfitting, we incorporate dropout in the fully connected layer. We now use only 80% of the training set for the actual training and leave the other 20% as a validation set. We scan over a suitable range of the dropout probability (range [0.1, 0.9] in steps of 0.1) to find an optimal value of this dropout probability, using accuracy on the validation set as the measure of performance for this search. 

Then we fix the optimal dropout, and retrain the model on the full training set. Evaluate the loss and accuracy over epochs for both the training and test sets, and compare them to the model without the dropout layer. The point is to observe the effect on overfitting of the incorporation of a dropout. 

# Dimensionality Reduction and K-means Clustering on the Fashion MNIST dataset

We use PCA dimensionality reduction on the Fashion MNIST dataset. Then we perform k-means clustering algorithm and choose an optimal k and interpret the results with relation to the dataset.