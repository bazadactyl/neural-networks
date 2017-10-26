import numpy as np
from load_mnist_data import load_mnist_data


def sigmoid(x, inverse=False):
    result = 1 / (1 + np.exp(-x))

    if not inverse:
        return result
    else:
        return result - (1 - result)


class FFNet:
    def __init__(self, arch=np.array([784, 40, 20, 10])):
        self._sample = np.random.normal(0, 1, size=(28, 28)).flatten().reshape(784, 1)
        self._arch = arch
        self._num_layers = len(self._arch)
        self._init_w()
        self._init_b()

    def _init_w(self, low=0, high=1):
        self._w = [np.random.normal(low, high, size=(j, i)) for i, j in zip(self._arch[:-1], self._arch[1:])]

    def _init_b(self, low=0, high=1):
        self._b = [np.random.normal(low, high, size=(i, 1)) for i in self._arch[1:]]

    def _activate(self, z):
        return sigmoid(z)

    def _calculate_z(self, x, n):
        return np.dot(self._w[n], x) + self._b[n]

    def _propagate(self, x):
        self._a = []
        self._o = []
        tmp = x

        for l in range(self._num_layers - 1):
            tmp = self._calculate_z(tmp, l)
            self._o.append(tmp)
            tmp = self._activate(tmp)
            self._a.append(tmp)

        return self._a, self._o

    def _backpropagate(self, x, y):
        # x = final activation
        # final = multiply((x-y),_activate(x, inverse=True))
        pass


def main():
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    net = FFNet()


if __name__ == '__main__':
    main()
