import random
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


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

    # Calculate global energy
    def _global_energy(self, active, threshold):
        num_neurons = len(active)
        energy = 0.0
        for neuron_i in range(num_neurons):
            energy += threshold[neuron_i] * active[neuron_i]
            for neuron_j in range(num_neurons):
                energy += -0.5 * self._w[neuron_i][neuron_j] * active[neuron_i] * active[neuron_j]
        return energy


    # Calculate quadratic energy function
    def _qef(self):
        pass

    def train(self, X):
        for x in X:
            a = x.reshape((784, 1))
            b = a.T
            self._w += np.dot(a, b)

        self._w -= (np.identity(x.size) * X.shape[0])
        return self._w

    def recover(self, x, threshold=0.0):
        num_neurons = x.size
        active = np.random.randint(2, size=num_neurons)
        threshold = [threshold] * num_neurons
        for i, _ in enumerate(active):
            active[i] = -1 if active[i] == 0 else 1
        history = []
        iteration = 0
        while not self.stable(history):
            neuron_i = random.choice(range(num_neurons))
            weight_sum = 0.0
            for neuron_j in range(num_neurons):
                weight_sum += self._w[neuron_i][neuron_j] * active[neuron_j]
            active[neuron_i] = 1 if weight_sum > threshold[neuron_i] else -1
            iteration += 1
            if iteration % 200 == 0:
                energy = self._global_energy(active, threshold)
                history.append(energy)
        recovered_image = active.reshape(28, 28)
        return recovered_image

    @staticmethod
    def degrade(x, noise):
        pixels_to_alter = round(x.size * noise)
        for _ in range(pixels_to_alter):
            pixel = random.choice(range(x.size))
            x[pixel] = 1 if x[pixel] == -1 else -1
        return x


    @staticmethod
    def stable(energy_history, check_last=5):
        if len(energy_history) < check_last:
            return False
        recent_states = energy_history[-check_last:]
        for i in range(check_last - 1):
            if recent_states[i] != recent_states[i + 1]:
                return False
        return True


def visualize(x):
    image = x.reshape(28, 28)

    fig, ax = plt.subplots()
    ax.imshow(x, interpolation='nearest')

    num_rows, num_cols = image.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < num_cols and 0 <= row < num_rows:
            z = x[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.show()


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

    # Regularize the training examples
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            X[row][col] = 1.0 if X[row][col] > 0.0 else -1.0

    # X = X[0:1]
    # y = y[0:1]

    X = X[-1:]
    y = y[-1:]

    # Initialize network
    net = HopfieldNetwork()
    print(net.train(X))

    # Test the network
    x = np.copy(X[0])
    x = net.degrade(x, 0.05)
    x = net.recover(x)
    visualize(x)


if __name__ == '__main__':
    main()
