# Bare-bones network

import numpy as np
import random

class BasicNetwork():
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x,y in zip(sizes[:-1], sizes[1:])]
                        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        if test_data:
            n_test = len(test_data)
            
        n = len(training_data)
        
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.backprop(mini_batch, eta)
            if test_data:
                print "Epoch {}: {} / {}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {} complete".format(j)
        
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
                activation = sigmoid_vec(z)
                activations.append(activation)
            
            # backward pass
            delta = self._cost_derivative(activations[-1], y) * \
                sigmoid_prime_vec(zs[-1])
            nabla_b[-1] += delta
            nabla_w[-1] += np.dot(delta, activations[-2].transpose())
            
            # l indexes layers from the output layer, e.g. l=1 is the last
            # layer, l=2 is the second-last layer, and so on
            for l in xrange(2, self.num_layers):
                z = zs[-l]
                spv = sigmoid_prime_vec(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, activations[-l-1].transpose())
                
        # Update after looking through all data
        self.weights = [w-eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb for b, nb in zip(self.biases, nabla_b)]
            
    def _cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
        

# Numerical functions

def tanh(z):
    return np.tanh(z)
tanh_vec = np.vectorize(tanh)
def tanh_prime(z):
    return 1.0 - np.tanh(z)**2.0
tanh_vec = np.vectorize(tanh_prime)


def softplus(z):
    return np.log(1.0+np.exp(z))
softplus_vec = np.vectorize(softplus)
def softplus_prime(z):
    return  1.0/(1.0+np.exp(-z)) 
softplus_prime_vec = np.vectorize(softplus_prime)


def rectifier(z):
    return max(0.0,z)
rect_vec = np.vectorize(rectifier)
def rectifier_prime(z):
    return 1.0*(z>0)
rectifier_prime_vec = np.vectorize(rectifier_prime)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
sigmoid_vec = np.vectorize(sigmoid)
def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))
sigmoid_prime_vec = np.vectorize(sigmoid_prime)