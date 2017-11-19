import random
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, wx, wy, wv, lr=0.5, sigma=2.0):
        # Steps:
        # 1.) Initialize wx*wy*784 random weights for each node n
        # 2.) Input training sample X
        # 3.) Find BMU by determining the node with a minimum L2 norm between x and n
        # 4.) Weights of all nodes n in a radius around the target node are adjusted, such that their weights
        #     shift closer to the weights of the target node by addition or subtraction of lr. Lr decreases as a function
        #     of distance from the target node.
        # 5.) Repeat for N training samples, diminishing the adjustment radius for each N samples
        self._seed = random.seed()
        self._sigma = sigma
        self._lr = lr
        self._wx, self._wy, self._wv = wx, wy, wv
        self._init_w()
        self._normalize_w()
        self._init_m()
        self._init_nxny()

    # Initialize all weights
    def _init_w(self):
        self._w = np.random.normal(loc=0.5, scale=1.0, size=(self._wx, self._wy, self._wv))
        return self._w

    # Normalize weights to range [0,1]
    def _normalize_w(self):
        self._w = self._w / self._w.max()
        return self._w

    # Initialize activation map
    def _init_m(self):
        self._m = np.zeros((self._wx, self._wy))
        return self._m

    # Initialize weight neighbourhood
    def _init_nxny(self):
        self._nx = np.arange(self._wx)
        self._ny = np.arange(self._wy)
        return (self._nx, self._ny)

    # Define decay function
    def _decay(self, x, t, max_iter):
        return x/(1+t/max_iter)

    # Euclidean distance between each neuron and our input X
    # TODO: Make this more Pythonic and less like C
    def _activate(self, X):
        for i in range(self._wx):
            for j in range(self._wy):
                self._m[i,j] = np.linalg.norm(self._w[i,j] - X)

        return self._m

    # Find the smallest Euclidean distance of current activation (nearest neighbour)
    def _find_nn(self):
        return np.unravel_index(self._m.argmin(), self._m.shape)

    # Convenient way of calculating Gaussian from: 
    # https://github.com/JustGlowing/minisom
    def _gaussian(self, c, sigma):
        d = 2*np.pi*sigma**2
        ax = np.exp(-np.power(self._nx-c[0], 2)/d)
        ay = np.exp(-np.power(self._ny-c[1], 2)/d)

        return np.outer(ax,ay)

    # Weight update function
    # TODO: Implement sigma and LR decay per training iteration without copying code
    #       from https://github.com/JustGlowing/minisom
    def _update_w(self, t_X, t_w, t, iterations):
        eta = self._decay(self._lr, t, iterations/2)
        sig = self._decay(self._sigma, t, iterations/2)
        gaussian_map = self._gaussian(t_w, sigma=sig)*eta
        gaussian_map = np.repeat(gaussian_map[:,:,np.newaxis], 784, axis=2)

        # Calculate deltas, adjust weights, and normalize
        deltas = t_X - self._w
        self._w += gaussian_map*deltas
        self._normalize_w()

        return self._w

    # Train on X
    def train_iter(self, X, iterations):
        for i in range(iterations):
            j = np.random.randint(low=0, high=X.shape[0])

            self._activate(X[j])
            self._update_w(X[j], self._find_nn(), i, iterations)

        return self._w

    # Approx. weights
    def approx_weights(self):
        t = self._w.reshape(self._wx*self._wy,784)

        pca = decomposition.PCA(n_components=2).fit(t)
        pca_2d = pca.transform(t)

        return pca_2d.T


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
    X, y = (np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.float64))
    X = X/X.max()

    # Init SOM
    som = SOM(30,30,784)

    # Train for 1000 iterations, randomly sampling from 1s and 5s
    som.train_iter(X, iterations=1000)

    # Approximation of weights
    weights = som.approx_weights()


    # Plot the PCA-reduced weights
    plt.scatter(weights[0], weights[1])
    plt.show()


if __name__ == '__main__':
    main()