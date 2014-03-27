from ann import OjaNetwork

import mnist_loader

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Pull all training examples for a given label
label = 3
training_data = mnist_loader.load_training_data_with_label(label)

# Optional loading of multiple classes
td2 = mnist_loader.load_training_data_with_label(7)
training_data.extend(td2)

# Run a single pass through data
net = OjaNetwork(784)
net.learn_oja(training_data, 0.02)

# Show the result
plt.imshow(np.reshape(net.weights, (28,28)), cmap=cm.Greys_r)
plt.show(True)