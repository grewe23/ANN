from ann import BasicNetwork

import activation_functions as af
import mnist_loader

# Load data (see data/README for instructions on downloading MNIST set)
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
    
# Create a basic three-layer network with [784, 30, 10] neurons
num_neurons = [784, 30, 10]

# By default, activation function is sigmoid
#net = BasicNetwork(num_neurons)

# Instead, we can use the tanh function
net = BasicNetwork(num_neurons, f=af.tanh_vec, fp=af.tanh_prime_vec)

# Have the basic network learn to classify the MNIST dataset, by 
# running the following
net.SGD(training_data,
        epochs=30,          # Number of full passes over training data
        mini_batch_size=10, # Number of samples for a single SGD step
        eta = 0.5,          # Learning rate
        test_data = test_data)