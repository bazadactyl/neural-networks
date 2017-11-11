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

    def neuron_weights(self):
        return np.array([sum(neuron_weights) for neuron_weights in self._w])

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

        # def activate(neuron_i):
        #     weight_sum = 0.0
        #     for neuron_j in range(num_neurons):
        #         weight_sum += self._w[neuron_i][neuron_j] * active[neuron_j]
        #     active[neuron_i] = 1 if weight_sum > threshold[neuron_i] else -1

        # def activate(neuron_i):
        #     weight_sum = 0.0
        #     for neuron_j in range(num_neurons):
        #         weight_sum += self._w[neuron_i][neuron_j] * active[neuron_j]
        #     if weight_sum > threshold[neuron_i]:
        #         active[neuron_i] = 1
        #         for neuron_j in range(num_neurons):
        #             if self._w[neuron_i][neuron_j] > 0:
        #                 active[neuron_j] = 1
        #     else:
        #         active[neuron_i] = -1

        def activate(neuron_i, recursive=False):
            weight_sum = 0.0
            for neuron_j in range(num_neurons):
                weight_sum += self._w[neuron_i][neuron_j] * active[neuron_j]
            if weight_sum > threshold[neuron_i]:
                active[neuron_i] = 1
                if recursive:
                    return
                for neuron_j in range(num_neurons):
                    if self._w[neuron_i][neuron_j] > 0 and neuron_j in remaining:
                        activate(neuron_j, recursive=True)
                        remaining.remove(neuron_j)
            else:
                active[neuron_i] = -1

        # Ensure each neuron was activated at least once, in random order
        remaining = list(range(num_neurons))
        while remaining:
            random_index = random.randint(0, len(remaining) - 1)
            random_neuron = remaining.pop(random_index)
            activate(random_neuron)

        # Recover random bits of the original image until the network is stable
        # Comment out to sacrifice accuracy for speed
        while not stable(history):
            random_neuron = random.choice(range(num_neurons))
            activate(random_neuron)
            iteration += 1
            if iteration % 10 == 0:
                energy = self._global_energy(active, threshold)
                history.append(energy)
            if iteration == 1000:
                break

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
    """Flip random bits in the input image.
    :param x: the input image
    :param noise: percentage of bits to flip
    """
    image = np.copy(x)
    pixels_to_alter = round(image.size * noise)
    for _ in range(pixels_to_alter):
        pixel = random.choice(range(x.size))
        image[pixel] = 1 if image[pixel] == -1 else -1
    return image


def stable(energy_history, check_last=5):
    """Check if the Hopfield network has stabilized based upon recent energy states."""
    if len(energy_history) < check_last:
        return False
    recent_states = energy_history[-check_last:]
    for i in range(check_last - 1):
        if recent_states[i] != recent_states[i + 1]:
            return False
    return True


def flip(image):
    """Change the sign on each element of the image."""
    flipped = np.copy(image)
    flipped *= -1
    return flipped


def invert(image):
    """Replace all min. values with max. values in the 2D array and vice-versa."""
    inverted = np.copy(image)
    arr_min, arr_max = image.min(), image.max()
    for i in range(image.shape[0]):
        value = inverted[i]
        if value == arr_max:
            inverted[i] = arr_min
        elif value == arr_min:
            inverted[i] = arr_max
        else:
            continue
    return inverted


def image_norm(original, recovered):
    """Invert the recovered image if it yields a lower L2 norm with the original image."""
    inverted = invert(recovered)
    norm_recovered = np.linalg.norm(original - recovered)
    norm_inverted = np.linalg.norm(original - inverted)
    if norm_inverted < norm_recovered:
        return inverted, norm_inverted
    else:
        return recovered, norm_recovered


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


def visualize_neurons(network, title=None):
    weight_sums = [sum(neuron_weights) for neuron_weights in network._w]
    image = np.array(weight_sums).reshape(28, 28)
    image = flip(image)

    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')
    plt.suptitle(title or 'Hopfield Network State')
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

    if title:
        file_name = '{}.png'.format(title.replace(' ', '-'))
        fig.savefig(file_name)
    else:
        plt.draw()


def visualize_before_after(original, degraded, restored, network, title=None):
    original = original.reshape(28, 28)
    degraded = degraded.reshape(28, 28)
    restored = restored.reshape(28, 28)
    neurons = flip(network.neuron_weights().reshape(28, 28))
    num_rows, num_cols = original.shape

    fig, axis = plt.subplots(2, 2)
    top_left, top_right = axis[0, 0], axis[0, 1]
    bottom_left, bottom_right = axis[1, 0], axis[1, 1]

    plt.subplots_adjust(left=0.07, bottom=0.08, right=0.96, top=0.88, wspace=-0.30, hspace=0.28)
    plt.suptitle(title)

    top_left.title.set_text('Original')
    top_left.imshow(original, interpolation='nearest')

    top_right.title.set_text('Degraded')
    top_right.imshow(degraded, interpolation='nearest')

    bottom_left.title.set_text('Restored')
    bottom_left.imshow(restored, interpolation='nearest')

    bottom_right.title.set_text('Network State')
    bottom_right.imshow(neurons, interpolation='nearest')

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < num_cols and 0 <= row < num_rows:
            z = original[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    top_left.format_coord = format_coord
    top_right.format_coord = format_coord
    bottom_left.format_coord = format_coord

    if title:
        file_name = '{}.png'.format(title.replace(' ', '-'))
        fig.savefig(file_name)
    else:
        plt.draw()


def standard_run(num_samples):
    mnist, _ = load_mnist_data()
    images = [mnist[i] for i in np.random.choice(range(len(mnist)), num_samples, replace=False)]

    # Initialize network
    network = HopfieldNetwork()
    network.train(np.array(images))
    visualize_network(network)
    visualize_neurons(network)

    # Test the network
    for original in images:
        degraded = degrade(original, 0.10)
        recovered = network.restore(degraded, threshold=0)
        visualize_before_after(original, degraded, recovered, network)

    # Display the plots
    plt.show()


def experiment_run():
    mnist, _ = load_mnist_data()
    experiments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    repeat = 30
    noise = 0.20

    for num_samples in experiments:
        print("Networks of {} images:".format(num_samples))
        total_correct = 0
        for e in range(repeat):
            images = [mnist[i] for i in np.random.choice(range(len(mnist)), num_samples, replace=False)]
            network = HopfieldNetwork()
            network.train(np.array(images))
            visualize_neurons(network, title='Network storing {:02d} images (experiment #{:02d})'
                              .format(num_samples, e + 1))
            correct = 0
            threshold = 10

            for i, original in enumerate(images):
                degraded = degrade(original, noise)
                recovered = network.restore(degraded, threshold=0)
                recovered, l2_norm = image_norm(original, recovered)
                title = 'Network of {:02d} images, experiment {:02d}, image {:02d}, noise: {:.2f}, L2-norm: {:.2f}' \
                    .format(num_samples, e + 1, i + 1, noise, l2_norm)
                visualize_before_after(original, degraded, recovered, network, title=title)
                correct += (1 if l2_norm < threshold else 0)

            total_correct += correct
            accuracy = (correct / num_samples) * 100
            print("\tExperiment {:02d}: {} / {} correct - {:.2f}% accuracy"
                  .format(e + 1, correct, num_samples, accuracy))
            plt.close('all')

        total_accuracy = (total_correct / (repeat * num_samples)) * 100
        print("\tAccuracy for {:02d}-image networks: {:02d} / {:02d} correct - {:.2f}% accuracy\n"
              .format(num_samples, total_correct, repeat * num_samples, total_accuracy))


def main():
    try:
        num_samples = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage:\n\tpython3 hopfieldnet.py <num-training-samples>")
        return

    # standard_run(num_samples)
    experiment_run()


if __name__ == '__main__':
    main()
