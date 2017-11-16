import itertools
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata


class HopfieldNetwork:
    def __init__(self, shape):
        self.seed = random.seed()
        self.shape = shape
        self.state = None
        self.weights = None
        self.initialize_weights()
        self.thresholds = None
        self.initialize_thresholds()

    def initialize_weights(self):
        """Hopfield networks start with all weights set to zero."""
        self.weights = np.zeros((self.shape, self.shape), dtype=np.float64)

    def initialize_thresholds(self):
        self.thresholds = np.zeros(self.shape, dtype=np.float64)

    def neuron_weights(self):
        """Return a list containing each neuron's sum of weights.
        Used for visualizing the network."""
        # TODO Remove this function.
        return np.array([sum(neuron_weights) for neuron_weights in self.weights])

    def energy(self):
        """Compute the global energy of the input network state.
        Refer to https://en.wikipedia.org/wiki/Hopfield_network#Energy
        """
        w = self.weights
        s = self.state
        t = self.thresholds

        a = np.matmul(s.reshape(self.shape, 1), s.reshape(self.shape, 1).T)
        b = np.multiply(w, a)
        c = np.dot(t, s)
        energy = (-0.5 * np.sum(b)) - c

        return energy

    def energy_unoptimized(self):
        """Inefficient version the the energy function above.
        Uses individual multiplications instead of efficient matrix operations.
        """
        num_neurons = self.shape
        energy = 0.0
        for i in range(num_neurons):
            energy += self.thresholds[i] * self.state[i]
            for j in range(num_neurons):
                energy += -0.5 * self.weights[i][j] * self.state[i] * self.state[j]
        return energy

    def train_hebbian(self, images):
        """Train the Hopfield network using the Hebbian learning rule (1949).
        https://en.wikipedia.org/wiki/Hopfield_network#Hebbian_learning_rule_for_Hopfield_networks
        """
        for img in images:
            a = img.reshape((self.shape, 1))
            b = a.T
            self.weights += np.dot(a, b)
        self.weights -= (np.identity(images[0].size) * images.shape[0])
        return self.weights

    def train_hebbian_unoptimized(self, images):
        """Inefficient version of the train_hebbian function.
        Performs individual multiplications instead of efficient matrix operations."""
        n = self.shape
        for i, j in itertools.product(range(n), range(n)):
            self.weights[i][j] = sum([img[i] * img[j] for img in images]) / n
        return self.weights

    def train_storkey(self, images):
        """Train the Hopfield network using the Storkey learning rule (1997).
        https://en.wikipedia.org/wiki/Hopfield_network#The_Storkey_learning_rule
        """
        n = self.shape
        for img in images:
            for i, j in itertools.product(range(n), range(n)):
                wt = self.weights
                w = wt[i][j]
                x = img[i] * img[j]
                y = img[i] * (np.dot(wt[j], img) - wt[j][i] * img[i] - wt[j][j] * img[j])
                z = img[j] * (np.dot(wt[i], img) - wt[i][i] * img[i] - wt[i][j] * img[j])
                wt[i][j] = w + ((x - y - z) / n)

    def train_storkey_unoptimized(self, images):
        """Inefficient version of the train_storkey function.
        Performs individual multiplications instead of efficient matrix operations."""
        n = self.shape
        for img in images:
            for i, j in itertools.product(range(n), range(n)):
                w = self.weights[i][j]
                x = img[i] * img[j] / n
                y = img[i] * sum([self.weights[j][k] * img[k] for k in range(n) if k not in [i, j]]) / n
                z = img[j] * sum([self.weights[i][k] * img[k] for k in range(n) if k not in [i, j]]) / n
                self.weights[i][j] = w + x - y - z

    def activate(self, i):
        weight_sum = np.dot(self.weights[i], self.state)
        self.state[i] = 1 if weight_sum > self.thresholds[i] else -1

    def activate_unoptimized(self, i):
        num_neurons = self.shape
        weight_sum = 0.0
        for j in range(num_neurons):
            weight_sum += self.weights[i][j] * self.state[j]
        self.state[i] = 1 if weight_sum > self.thresholds[i] else -1

    def restore(self, degraded_image):
        """Recover the original pattern of the degraded input pattern."""
        self.state = np.copy(degraded_image)
        num_neurons = self.shape

        # During each iteration: ensure each neuron is activated at least once
        iterations = 0
        while iterations < 10:
            changed = False
            neurons = list(range(num_neurons))
            random.shuffle(neurons)
            while neurons:
                neuron = neurons.pop()
                old_state = self.state[neuron]
                self.activate(neuron)
                new_state = self.state[neuron]
                changed = True if old_state != new_state else changed
            iterations += 1
            if not changed:
                break

        recovered_image = np.copy(self.state)
        return recovered_image


def load_mnist_data():
    """Use scikit-learn to load the MNIST dataset.
    For our purposes we only need the 1's and 5's and require
    that the images contain 1 and -1 values only."""
    mnist = fetch_mldata('MNIST original')
    data = mnist.data
    target = mnist.target

    # Append only 1's and 5's from MNIST
    images = []
    labels = []
    for i in range(len(data)):
        if target[i] == 1 or target[i] == 5:
            images.append(data[i])
            labels.append(target[i])

    # Convert lists to numpy arrays
    images, labels = (np.asarray(images, dtype=np.float64), np.asarray(labels, dtype=np.uint64))

    # Regularize the training examples to 1-bit patterns
    for row, col in itertools.product(range(images.shape[0]), range(images.shape[1])):
        images[row][col] = 1.0 if images[row][col] > 0.0 else -1.0

    return images, labels


def degrade(image, noise):
    """Flip random bits in the input image.
    :param image: the input image
    :param noise: percentage of bits to flip
    """
    num_pixels = image.size
    degraded = np.copy(image)
    pixels_to_alter = round(num_pixels * noise)
    for _ in range(pixels_to_alter):
        pixel = random.choice(range(num_pixels))
        degraded[pixel] = 1 if degraded[pixel] == -1 else -1
    return degraded


def chop(image):
    """Clear the bottom half of the image's pixels.
    The affected pixels' values are set to the minimum pixel value of the image.
    """
    image = np.copy(image).reshape(28, 28)
    min_value = image.min()
    for i, j in itertools.product(range(14, 28), range(0, 28)):
        image[i][j] = min_value
    return image.reshape(784,)


def stable(energy_history, check_last=2):
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


def visualize_network(network, save=False):
    image = network.weights
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
    if save:
        file_name = 'network.png'
        fig.savefig(file_name)
    else:
        plt.draw()


def visualize_neurons(network, title=None, save=False):
    weight_sums = [sum(neuron_weights) for neuron_weights in network.weights]
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

    if title and save:
        file_name = '{}.png'.format(title.replace(' ', '-'))
        fig.savefig(file_name)
    else:
        plt.draw()


def visualize_before_after(original, degraded, restored, network, title=None, save=False):
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

    if title and save:
        file_name = '{}.png'.format(title.replace(' ', '-'))
        fig.savefig(file_name)
    else:
        plt.draw()


def standard_run(num_samples):
    mnist, _ = load_mnist_data()
    images = [mnist[i] for i in np.random.choice(range(len(mnist)), num_samples, replace=False)]

    # Initialize network
    network = HopfieldNetwork(shape=784)
    network.train_storkey(np.array(images))
    visualize_network(network)
    visualize_neurons(network)

    # Test the network
    for original in images:
        degraded = degrade(original, noise=0.10)
        # degraded = chop(degraded)
        recovered = network.restore(degraded)
        visualize_before_after(original, degraded, recovered, network)

    # Display the plots
    plt.show()


def experiment_run(save_figures=False):
    mnist, _ = load_mnist_data()
    experiments = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    repeat = 10
    noise = 0.20

    for num_samples in experiments:
        print("Networks of {} images:".format(num_samples))
        total_correct = 0
        for e in range(repeat):
            images = [mnist[i] for i in np.random.choice(range(len(mnist)), num_samples, replace=False)]
            network = HopfieldNetwork(shape=784)
            network.train_storkey(np.array(images))
            visualize_neurons(network, title='Network storing {:02d} images (experiment #{:02d})'
                              .format(num_samples, e + 1), save=save_figures)
            correct = 0
            threshold = 10

            for i, original in enumerate(images):
                degraded = degrade(original, noise)
                # degraded = chop(original)
                recovered = network.restore(degraded)
                l2_norm = np.linalg.norm(original - recovered)
                title = 'Network of {:02d} images, experiment {:02d}, image {:02d}, noise: {:.2f}, L2-norm: {:.2f}' \
                    .format(num_samples, e + 1, i + 1, noise, l2_norm)
                visualize_before_after(original, degraded, recovered, network, title=title, save=save_figures)
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
    experiment_run(save_figures=False)


if __name__ == '__main__':
    main()
