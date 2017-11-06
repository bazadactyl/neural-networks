import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_mldata


class HopfieldNetwork:
    def __init__(self):
        pass


def main():
    # Load and prepare data
    mnist = fetch_mldata('MNIST original')
    data = mnist.data
    target = mnist.target

    # Append only 1's and 5's from MNIST to X and y lists
    X = []
    y = []
    [(X.append(data[i]), y.append(target[i])) for i in range(len(data)) if target[i] == 0 or target[i] == 4]

    # Convert X and y lists to numpy arrays
    X, y = (np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.float64))

    # Initialize network
    net = HopfieldNetwork()


if __name__ == '__main__':
    main()