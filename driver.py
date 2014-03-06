from ann import BasicNetwork
import mnist_loader

# Load data (see data/README for instructions on downloading MNIST set)
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
    
# Create a basic three-layer network with [784, 30, 10] neurons
net = BasicNetwork([784, 30, 10])

# Have the basic network learn to classify the MNIST dataset, by 
# running the following
net.SGD(training_data,
        epochs=10,          # Number of full passes over training data
        mini_batch_size=10, # Number of samples for a single SGD step
        eta = 1.0,          # Learning rate
        test_data = test_data)