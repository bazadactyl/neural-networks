import gzip
import multiprocessing
import os
from struct import unpack
import numpy as np

CHUNK = 4  # bytes
NUMBER = 1  # bytes


def read_mnist_images(file_path):
    """This function was inspired by code we saw in https://github.com/hdmetor/NeuralNetwork

    :param file_path: gun-zipped file of MNIST images
    :return: array of image data
    """
    print('Loading MNIST images from {}...'.format(os.path.basename(file_path)))
    with gzip.open(file_path, 'rb') as f:
        f.read(CHUNK)  # first chunk not needed

        num_images = unpack('>I', f.read(CHUNK))[0]
        rows = unpack('>I', f.read(CHUNK))[0]
        cols = unpack('>I', f.read(CHUNK))[0]
        images = np.zeros((num_images, rows, cols), dtype=np.uint8)

        for i in range(num_images):
            for row in range(rows):
                for col in range(cols):
                    pixel_data = unpack('>B', f.read(NUMBER))[0]
                    images[i][row][col] = pixel_data

    print('Finished loading MNIST images from {}'.format(os.path.basename(file_path)))
    return images


def read_mnist_labels(file_path):
    """This function was inspired by code we saw in https://github.com/hdmetor/NeuralNetwork

    :param file_path: gun-zipped file of MNIST images
    :return: array of image data
    """
    print('Loading MNIST labels from {}...'.format(os.path.basename(file_path)))
    with gzip.open(file_path, 'rb') as f:
        f.read(CHUNK)  # first chunk not needed

        num_labels = unpack('>I', f.read(CHUNK))[0]
        labels = np.zeros((num_labels, 1), dtype=np.uint8)

        for i in range(num_labels):
            label_data = f.read(NUMBER)
            labels[i] = unpack('>B', label_data)[0]

    print('Finished loading MNIST labels from {}'.format(os.path.basename(file_path)))
    return labels


def load_mnist_from_numpy_files(data_dir):
    """Load the MNIST dataset from numpy files created in a previous run."""
    train_images = np.load(os.path.join(data_dir, 'train-images.npy'))
    train_labels = np.load(os.path.join(data_dir, 'train-labels.npy'))
    test_images = np.load(os.path.join(data_dir, 'test-images.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test-labels.npy'))

    return train_images, train_labels, test_images, test_labels


def load_mnist_from_mnist_files(data_dir):
    """Load the MNIST dataset from the original MNIST files."""
    train_images = read_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = read_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    test_images = read_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = read_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    np.save(os.path.join(data_dir, 'train-images.npy'), train_images)
    np.save(os.path.join(data_dir, 'train-labels.npy'), train_labels)
    np.save(os.path.join(data_dir, 'test-images.npy'), test_images)
    np.save(os.path.join(data_dir, 'test-labels.npy'), test_labels)

    return train_images, train_labels, test_images, test_labels


def load_mnist_data():
    """Load the MNIST dataset.

    :return: 4-tuple of train images and labels, and test images and labels
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    mnist_dir = os.path.join(curr_dir, 'mnist-data')

    try:
        print('Trying to load MNIST dataset from numpy files...')
        return load_mnist_from_numpy_files(mnist_dir)
    except IOError:
        print('Numpy files with MNIST dataset not found.')
        print('Trying to load MNIST dataset from raw MNIST files...')
        return load_mnist_from_mnist_files(mnist_dir)
