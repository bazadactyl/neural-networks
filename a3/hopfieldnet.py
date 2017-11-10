import random
import numpy as np
import sys
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

    def restore(self, x, threshold=0.0):
        image = np.copy(x)
        num_neurons = image.size

        threshold = [threshold] * num_neurons
        active = np.random.randint(2, size=num_neurons)
        for i, _ in enumerate(active):
            active[i] = -1 if active[i] == 0 else 1

        history = []
        iteration = 0

        def activate(neuron_i):
            weight_sum = 0.0
            for neuron_j in range(num_neurons):
                weight_sum += self._w[neuron_i][neuron_j] * active[neuron_j]
            active[neuron_i] = 1 if weight_sum > threshold[neuron_i] else -1

        # Ensure each neuron was activated at least once, in random order
        remaining = list(range(num_neurons))
        while remaining:
            random_index = random.randint(0, len(remaining) - 1)
            random_neuron = remaining.pop(random_index)
            activate(random_neuron)

        # Recover random bits of the original image until the network is stable
        while not stable(history):
            random_neuron = random.choice(range(num_neurons))
            activate(random_neuron)
            iteration += 1
            if iteration % 100 == 0:
                energy = self._global_energy(active, threshold)
                history.append(energy)

        recovered_image = active
        return recovered_image


def load_mnist_data():
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

    return X, y


def degrade(x, noise):
    image = np.copy(x)
    pixels_to_alter = round(image.size * noise)
    for _ in range(pixels_to_alter):
        pixel = random.choice(range(x.size))
        image[pixel] = 1 if image[pixel] == -1 else -1
    return image


def stable(energy_history, check_last=5):
    if len(energy_history) < check_last:
        return False
    recent_states = energy_history[-check_last:]
    for i in range(check_last - 1):
        if recent_states[i] != recent_states[i + 1]:
            return False
    return True


def visualize_network(net):
    image = net._w
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')
    plt.suptitle('Hopfield Network Weights')
    num_rows, num_cols = image.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < num_cols and 0 <= row < num_rows:
            z = image[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.draw()


def visualize_neurons(net):
    weight_sums = [sum(neuron_weights) for neuron_weights in net._w]
    image = np.array(weight_sums).reshape(28, 28)

    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')
    plt.suptitle('Hopfield Network Neurons')
    num_rows, num_cols = image.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < num_cols and 0 <= row < num_rows:
            z = image[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.draw()


def visualize_before_after(original, degraded, restored):
    original = original.reshape(28, 28)
    degraded = degraded.reshape(28, 28)
    restored = restored.reshape(28, 28)
    num_rows, num_cols = original.shape

    l2_norm = np.linalg.norm(original - restored)

    fig, axis = plt.subplots(1, 3)
    left, center, right = axis[0], axis[1], axis[2]
    fig.tight_layout()
    plt.suptitle('L2 norm between original and restored: {:.2f}'.format(l2_norm))

    left.title.set_text('Original')
    left.imshow(original, interpolation='nearest')

    center.title.set_text('Degraded')
    center.imshow(degraded, interpolation='nearest')

    right.title.set_text('Restored')
    right.imshow(restored, interpolation='nearest')

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < num_cols and 0 <= row < num_rows:
            z = original[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    left.format_coord = format_coord
    center.format_coord = format_coord
    right.format_coord = format_coord

    plt.draw()


def main():
    try:
        num_samples = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage:\n\tpython3 hopfieldnet.py <num-training-samples>")
        return

    mnist, _ = load_mnist_data()
    images = [mnist[i] for i in np.random.choice(range(len(mnist)), num_samples, replace=False)]

    # Initialize network
    net = HopfieldNetwork()
    net.train(np.array(images))
    visualize_network(net)
    visualize_neurons(net)

    # Test the network
    for original in images:
        degraded = degrade(original, 0.05)
        recovered = net.restore(degraded)
        visualize_before_after(original, degraded, recovered)

    # Display the plots
    plt.show()


if __name__ == '__main__':
    main()
