import numpy as np
import matplotlib.pyplot as plt

def display_weight_distr(data, n_bins=50):
    params  = data['params'].item()
    weights = data['weights']
     
    layers = params['layers']
    n_layers = len(layers)
    
    # There are n_layers - 1 set of weights
    for k in xrange(n_layers-1):
        plt.subplot(n_layers-1, 1, k+1)
        plt.hist(np.reshape(weights[k],(layers[k]*layers[k+1],1)),
                 bins=n_bins)
        plt.xlabel('Weights from layer {} to {}'.format(k,k+1))
        plt.ylabel('Frequency')
    plt.show()

# Example of running visualization code
source =  'ann_sigmoid_n784-30-10_ep10_mb10_eta5-00_i0000.npz'
data = np.load(source) # Fields: 'params', 'weights', 'biases', 'test_results'
display_weight_distr(data)
