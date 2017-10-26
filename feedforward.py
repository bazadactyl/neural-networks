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

    def train_network(self, images, labels, learn_rate=2.0, epochs=50, batch_size=100):
        images = images.T
        labels = labels.T[0]
        num_images = images.shape[1]

        for epoch in range(epochs):
            # Permute the images for each epoch (yields better results)
            images = images.T
            np.random.shuffle(images)  # shuffle rows
            images = images.T

            # Split the images into batches (makes training faster)
            batches = []
            num_batches = int(num_images / batch_size)
            for i in range(num_batches):
                first = i * batch_size
                last = first + batch_size
                batch = np.array([row[first:last] for row in images])
                batches.append(batch)

            # Feed forward
            print('Hello')

            # Compute corrections
            print('Hello')

            # Backpropagate
            print('Hello')


def main():
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    net = FFNet()
    net.train_network(train_images, train_labels)


if __name__ == '__main__':
    main()
