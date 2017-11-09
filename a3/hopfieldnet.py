import random
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_mldata


class HopfieldNetwork:
    def __init__(self, input_shape=784):
        # Initialize variables
        self._seed = random.seed()
        self._shape = input_shape

        # Initialize weights
        self._init_w()

    # Initialize weights
    # Note: Number of "synapses" = neurons^2 - neurons
    #       Subtraction due to the fact that neurons cannot connect to themselves
    def _init_w(self):
        self._w = np.zeros((self._shape, self._shape), dtype=np.float64)
        return self._w

    def _train(self, X):
        for x in X:
            a = x.reshape((784, 1))
            b = a.T
            self._w += np.dot(a, b)

        self._w -= (np.identity(len(x)) * X.shape[0])
        return self._w

    # Calculate global energy
    def _global_energy(self):
        pass

    # Calculate quadratic energy function
    def _qef(self):
        pass

def main():
    # Load and prepare data
    mnist = fetch_mldata('MNIST original')
    data = mnist.data
    target = mnist.target

    # Append only 1's and 5's from MNIST to X and y lists
    X = []
    y = []
    [(X.append(data[i]), y.append(target[i])) for i in range(len(data)) if target[i] == 1 or target[i] == 5]

    # Convert X and y lists to numpy arrays
    X, y = (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.uint64))
    X = (X/128)-1
    X = X[0:100]
    y = y[0:100]


    # Initialize network
    net = HopfieldNetwork()
    print(net._train(X))


if __name__ == '__main__':
    main()