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
        self.train_inputs = None
        self.train_targets = None
        self.training_set_size = 0
        self.input_batches = None
        self.target_batches = None

    def _init_w(self, low=0, high=1):
        self._w = [np.random.normal(low, high, size=(j, i)) for i, j in zip(self._arch[:-1], self._arch[1:])]

    def _init_b(self, low=0, high=1):
        self._b = [np.random.normal(low, high, size=(i, 1)) for i in self._arch[1:]]

    def _prepare_train_inputs(self, images):
        """Arrange the inputs such that each column respresents the pixels of one image."""
        self.train_inputs = images.T
        self.training_set_size = self.train_inputs.shape[1]

    def _prepare_train_targets(self, labels):
        """Arrange the targets such that each column represents the desired output for an image."""
        labels = labels.T[0]
        self.train_targets = np.zeros((10, self.training_set_size))
        for i, label in enumerate(labels):
            self.train_targets[label][i] = 1

    def _shuffle_training_set(self):
        """Permute the inputs. Yields better results during training."""
        permutation = np.random.permutation(self.training_set_size)
        self.train_inputs = self.train_inputs.T[permutation].T
        self.train_targets = self.train_targets.T[permutation].T

    def _batch_training_data(self, batch_size):
        num_images = self.train_inputs.shape[1]
        num_batches = int(num_images / batch_size)

        self.input_batches = []
        self.target_batches = []

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            input_batch = np.array([row[start:end] for row in self.train_inputs])
            self.input_batches.append(input_batch)
            target_batch = np.array([row[start:end] for row in self.train_targets])
            self.target_batches.append(target_batch)

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

    def train_network(self, train_images, train_labels, learn_rate=2.0, epochs=50, batch_size=100):
        self._prepare_train_inputs(train_images)
        self._prepare_train_targets(train_labels)

        for epoch in range(epochs):
            self._shuffle_training_set()
            self._batch_training_data(batch_size)

            for input_batch, target_batch in zip(self.input_batches, self.target_batches):
                self._feed_forward(input_batch)

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
