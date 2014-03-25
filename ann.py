# Bare-bones network
from __future__ import division

import numpy as np
import random

import activation_functions as af

class OjaNetwork():
    '''
    Classic Oja network (single layer, single output, linear neuron)
    '''
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        
    def learn_oja(self, training_data, eta):
        n = len(training_data)
        for i in xrange(n):
            x = training_data[i]
            y = np.dot(self.weights, x)
            self.weights += eta/(i+1) * (x*y - y**2 * self.weights)

class BasicNetwork():
    '''
    Standard feedforward network with backpropagation
    '''
    def __init__(self, sizes, f=af.sigmoid_vec, fp=af.sigmoid_prime_vec):
        '''
        Initialize a basic feedforward ANN
        
        Inputs:
          - sizes: Number of neurons to use at each layer, e.g. [N1, N2, N3]
          - f:  Activation function (default: sigmoid). Should be vectorized
          - fp: Derivative of the activation function
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x,y in zip(sizes[:-1], sizes[1:])]
        self.f = f
        self.fp = fp
                        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.f(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        if test_data:
            n_test = len(test_data)
            
        n = len(training_data)
        
        test_results = [] # Keep track of test perf over epochs
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.backprop(mini_batch, eta)
            if test_data:
                n_correct = self.evaluate(test_data)
                print "Epoch {}: {} / {}".format(
                    j, n_correct, n_test)
                test_results.append(n_correct/n_test)
            else:
                print "Epoch {} complete".format(j)

        return test_results
        
    def backprop(self, training_data, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in training_data:
            # forward pass
            activation = x
            activations = [x]
            zs = []
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = self.f(z)
                activations.append(activation)
            
            # backward pass
            delta = self._cost_derivative(activations[-1], y) * \
                self.fp(zs[-1])
            nabla_b[-1] += delta
            nabla_w[-1] += np.dot(delta, activations[-2].transpose())
            
            # l indexes layers from the output layer, e.g. l=1 is the last
            # layer, l=2 is the second-last layer, and so on
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = self.fp(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, activations[-l-1].transpose())
                
        # Update after looking through all data
        n = len(training_data)
        self.weights = [w-eta*nw/n for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb/n for b, nb in zip(self.biases, nabla_b)]
            
    def _cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
        
