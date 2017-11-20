import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import decomposition
from sklearn.cluster import KMeans


class SOM:
    def __init__(self, wx, wy, wv, lr=0.5, sigma=5.0):
        """Steps:
            1.) Initialize wx*wy*784 random weights for each node n
            2.) Input training sample X
            3.) Find BMU by determining the node with a minimum L2 norm between x and n
            4.) Weights of all nodes n in a radius around the target node are adjusted, such that their weights
                shift closer to the weights of the target node by addition or subtraction of lr. Lr decreases
                as a function of distance from the target node.
            5.) Repeat for N training samples, diminishing the adjustment radius for each N samples
        """
        self._seed = random.seed()
        self._sigma = sigma
        self._lr = lr
        self._wx, self._wy, self._wv = wx, wy, wv
        self._init_w()
        self._normalize_w()
        self._init_m()
        self._init_nxny()

    def _init_w(self):
        """Initialize all weights to random values."""
        self._w = np.random.normal(loc=0.5, scale=1.0, size=(self._wx, self._wy, self._wv))
        return self._w

    def _normalize_w(self):
        """Normalize weights to range [0,1]."""
        self._w = self._w / self._w.max()
        return self._w

    def _init_m(self):
        """Initialize activation map."""
        self._m = np.zeros((self._wx, self._wy))
        return self._m

    def _init_nxny(self):
        """Initialize weight neighbourhood."""
        self._nx = np.arange(self._wx)
        self._ny = np.arange(self._wy)
        return self._nx, self._ny

    def _decay(self, x, t, max_iter):
        """Define decay function."""
        return x/(1+t/max_iter)

    def _activate(self, X):
        """Euclidean distance between each neuron and our input X."""
        for i in range(self._wx):
            for j in range(self._wy):
                self._m[i, j] = np.linalg.norm(self._w[i, j] - X)

        return self._m

    def _find_nn(self):
        """Find the smallest Euclidean distance of current activation (nearest neighbour)."""
        return np.unravel_index(self._m.argmin(), self._m.shape)

    def _gaussian(self, c, sigma):
        """# Convenient way of calculating Gaussian.
        The professor allowed referring to external libraries for SOM.
        From: https://github.com/JustGlowing/minisom
        """
        d = 2*np.pi*sigma**2
        ax = np.exp(-np.power(self._nx-c[0], 2)/d)
        ay = np.exp(-np.power(self._ny-c[1], 2)/d)
        return np.outer(ax,ay)

    def _update_w(self, t_X, t_w, t, iterations):
        """Weight update function.
        The professor allowed referring to external libraries for SOM.
        From: https://github.com/JustGlowing/minisom
        """
        eta = self._decay(self._lr, t, iterations/2)
        sig = self._decay(self._sigma, t, iterations/2)
        gaussian_map = self._gaussian(t_w, sigma=sig)*eta
        gaussian_map = np.repeat(gaussian_map[:, :, np.newaxis], 784, axis=2)

        # Calculate deltas, adjust weights, and normalize
        deltas = t_X - self._w
        self._w += gaussian_map*deltas
        self._normalize_w()

        return self._w

    def train_iter(self, X, iterations):
        """Train on X."""
        for i in range(iterations):
            j = np.random.randint(low=0, high=X.shape[0])

            self._activate(X[j])
            self._update_w(X[j], self._find_nn(), i, iterations)

        return self._w

    def approx_weights(self):
        """Approximate weights."""
        t = self._w.reshape(self._wx*self._wy,784)

        pca = decomposition.PCA(n_components=2).fit(t)
        pca_2d = pca.transform(t)

        return pca_2d.T

    def plot_neuron(self, neuron):
        reshaped = self._w.reshape(900, 784)
        image = reshaped[neuron].reshape(28, 28)

        fig, ax = plt.subplots()
        ax.imshow(image, interpolation='nearest')
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

    def plot_neurons(self, one_image, five_image, title=''):
        reshaped = self._w.reshape(900, 784)
        color = np.zeros(shape=900)

        for neuron in range(900):
            neuron_image_prototype = reshaped[neuron].reshape(28, 28)
            norm_with_one = np.linalg.norm(neuron_image_prototype - one_image)
            norm_with_five = np.linalg.norm(neuron_image_prototype - five_image)
            color[neuron] = 1 if norm_with_one < norm_with_five else 5

        color_grid = color.reshape(30, 30)

        fig, ax = plt.subplots()
        ax.imshow(color_grid, interpolation='nearest')
        num_rows, num_cols = color_grid.shape

        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if 0 <= col < num_cols and 0 <= row < num_rows:
                z = color_grid[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord
        plt.title("State of the Self-organizing Map")
        # plt.savefig('som-training-{}.png'.format(title))

    def plot_prototypes(self):
        for i in range(900):
            if i % 150 == 0:
                self.plot_neuron(i)

    def test_clustering_accuracy(self, images, labels, average_one, average_five):
        correct = 0
        num_images = images.shape[0]

        # For speed, only test part of the dataset
        num_rows = min(1000, num_images)
        random_rows = np.random.randint(num_images, size=num_rows)
        images_reduced = images[random_rows, :]
        labels_reduced = labels[random_rows]

        for image, label in zip(images_reduced, labels_reduced):
            neuron_norms = self._activate(image).reshape(900)
            winning_neuron = np.argmin(neuron_norms)
            weights = self._w.reshape(900, 784)
            prototype = weights[winning_neuron].reshape(28, 28)

            norm_with_one = np.linalg.norm(prototype - average_one)
            norm_with_five = np.linalg.norm(prototype - average_five)
            prototype_is_one = norm_with_one < norm_with_five
            prototype_is_five = norm_with_five <= norm_with_one

            if prototype_is_one and label == 1.0:
                correct += 1
            elif prototype_is_five and label == 5.0:
                correct += 1
            else:
                continue

        clustering_accuracy = (correct / num_rows) * 100
        print("Using SOM clustering, we got a clustering that is {:.2f}% accurate".format(clustering_accuracy))


def plot_clustered_2d_mnist_data(dim2_x, dim2_y, cluster, centroids, labels):
    one_x, one_y = [], []
    one_x_wrong, one_y_wrong = [], []
    five_x, five_y = [], []
    five_x_wrong, five_y_wrong = [], []

    for i in range(len(dim2_x)):
        x, y = dim2_x[i], dim2_y[i]
        if cluster[i] == 0 and labels[i] == 1:
            one_x.append(x)
            one_y.append(y)
        elif cluster[i] == 0 and labels[i] == 5:
            five_x_wrong.append(x)
            five_y_wrong.append(y)
        elif cluster[i] == 1 and labels[i] == 5:
            five_x.append(x)
            five_y.append(y)
        else:
            one_x_wrong.append(x)
            one_y_wrong.append(y)

    centroids_x = [row[0] for row in centroids]
    centroids_y = [row[1] for row in centroids]

    plt.subplots()
    plt.scatter(one_x_wrong, one_y_wrong, c='forestgreen', s=10)
    plt.scatter(five_x_wrong, five_y_wrong, c='indigo', s=10)
    plt.scatter(one_x, one_y, c='orangered', s=10)
    plt.scatter(five_x, five_y, c='dodgerblue', s=10)
    plt.scatter(centroids_x, centroids_y, c='black')

    plt.title("K-means Clustering of the MNIST Images in 2D")
    plt.legend([
        "'Ones' incorrectly clustered with 'Fives'",
        "'Fives' incorrectly clustered with 'Ones'",
        "Cluster of 'Ones'",
        "Cluster of 'Fives'",
        "Cluster centroids",
    ], loc='upper right')

    plt.draw()


def main():
    # Load and prepare data
    mnist = fetch_mldata('MNIST original')
    data = mnist.data
    target = mnist.target

    # Append only 1's and 5's from MNIST to X and y lists
    X = []
    X1 = []
    X5 = []
    y = []
    y1 = []
    y5 = []
    [(X.append(data[i]), y.append(target[i])) for i in range(len(data)) if target[i] == 1 or target[i] == 5]
    [(X1.append(data[i]), y1.append(target[i])) for i in range(len(data)) if target[i] == 1]
    [(X5.append(data[i]), y5.append(target[i])) for i in range(len(data)) if target[i] == 5]

    # Convert X and y lists to numpy arrays
    X, y = (np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.float64))
    X1, y1 = (np.asarray(X1, dtype=np.int64), np.asarray(y1, dtype=np.float64))
    X5, y5 = (np.asarray(X5, dtype=np.int64), np.asarray(y5, dtype=np.float64))

    # Regularize data to interval [0, 1]
    X = X / X.max()
    X1 = X1 / X1.max()
    X5 = X5 / X5.max()

    # Save the images representing the average 'One' and 'Five'
    average_one = X1.mean(axis=0).reshape(28, 28)
    average_five = X5.mean(axis=0).reshape(28, 28)

    # Visualize
    pca = decomposition.PCA(n_components=2).fit(X)
    dim2_data = pca.transform(X).T
    dim2_x, dim2_y = dim2_data
    # plt.scatter(dim2_x, dim2_y)
    # plt.draw()

    # Init SOM
    som = SOM(30, 30, 784)
    som.plot_neurons(average_one, average_five)

    # Train for 1000 iterations, randomly sampling from 1s and 5s
    for i in range(10):
        som.train_iter(X, iterations=20)
    som.plot_neurons(average_one, average_five)

    # Plot a few neuron prototypes
    # som.plot_prototypes()

    # Test SOM clustering accuracy
    som.test_clustering_accuracy(X, y, average_one, average_five)

    # Approximation of weights
    # weights = som.approx_weights()

    # Plot the PCA-reduced weights
    # plt.scatter(weights[0], weights[1])
    # plt.draw()

    # Perform K-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dim2_data.T)

    # Compute K-means clustering accuracy
    correct = 0
    total = y.size
    for i, label in enumerate(kmeans.labels_):
        label = 1.0 if label == 0 else 5.0
        if label == y[i]:
            correct += 1

    accuracy = (correct / total) * 100
    # since we hard-coded that cluster 0 is ONE and category 1 is FIVE
    clustering_accuracy = accuracy if accuracy > 50 else 100 - accuracy
    print("Using K-means clustering, we got a clustering that is {:.2f}% accurate".format(clustering_accuracy))

    # Visualize the K-means clustering
    plot_clustered_2d_mnist_data(dim2_x, dim2_y, kmeans.labels_, kmeans.cluster_centers_, y)

    # Display the plots we drew
    plt.show()


if __name__ == '__main__':
    main()
