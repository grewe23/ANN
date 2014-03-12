from ann import BasicNetwork

import activation_functions as af
import mnist_loader

import boto
import numpy as np

# Some parameters needed for uploads to S3 bucket
AWS_ACCESS_KEY_ID = 'AKIAIYY272VA3C5R4DSQ'
AWS_SECRET_ACCESS_KEY = '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'
BUCKET_NAME = 'anndata'
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
bucket = conn.get_bucket(BUCKET_NAME)

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
          'eta': 1.0,
         }

N_iter = 20

for i in xrange(N_iter):

    # Filename for saving results (via shelve)
    savename = "ann_{}_n{}_ep{}_mb{}_eta{}_i{:04d}.npz".format(
                    params['af_name'],
                    "-".join([str(n) for n in params['layers']]),
                    params['epochs'],
                    params['mini_batch_size'],
                    "{:.2f}".format(params['eta']).replace('.','-'),
                    i,
                    )

    # Instantiate the network
    net = BasicNetwork(params['layers'],
                       f =af.functions[params['af_name']][0],
                       fp=af.functions[params['af_name']][1])

    print "#------------------------------------------------------------"
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

    # Save results to file (using numpy)
    np.savez(savename, params=params,
                       test_results=test_results,
                       biases=net.biases,
                       weights=net.weights)

    # Upload to S3
    print "# Uploading result to S3..."
    k = boto.s3.key.Key(bucket)
    k.key = savename
    k.set_contents_from_filename(savename)
    k.make_public()

