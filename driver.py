from ann import BasicNetwork

import activation_functions as af
import mnist_loader

import math

# Load data (see data/README for instructions on downloading MNIST set)
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
    

# Parameters of ANN
# - af_name: Activation function of neurons (see activation_functions.py)
# - epochs: Number of full passes over training data
# - mini_batch_size: Number of samples for a single SGD step
# - eta: Learning rate
params = {'layers': [784, 30, 10],
          'af_name': 'sigmoid',
          'epochs': 3,
          'mini_batch_size': 10,
          'eta': 0.5,
         }

# Instantiate the network
net = BasicNetwork(params['layers'],
                   f =af.functions[params['af_name']][0],
                   fp=af.functions[params['af_name']][1])

print "# Instantiated ANN:\n\
#   - Structure: {} neurons\n\
#   - Activation function: {}".format(params['layers'], params['af_name'])

print "# Running SGD:\n\
#   - Epochs: {}\n\
#   - Mini-batch size: {} per batch ({} batches)".format(
        params['epochs'], params['mini_batch_size'],
        len(training_data)/params['mini_batch_size'])
print "#------------------------------------------------------------"

# Have the basic network learn to classify the MNIST dataset, by
# running the following
test_results = net.SGD(training_data,
                       epochs = params['epochs'],
                       mini_batch_size = params['mini_batch_size'],
                       eta = params['eta'],
                       test_data = test_data)

print test_results
