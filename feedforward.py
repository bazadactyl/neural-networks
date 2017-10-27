import numpy as np
import sklearn as sk
from load_mnist_data import load_mnist_data


def sigmoid(x, inverse=False):
    result = 1 / (1 + np.exp(-x))

    if not inverse:
        return result
    else:
        return result - (1 - result)


class FFNet:
    def __init__(self, arch=np.array([784, 40, 20, 10]), lr=0.1):
        self._arch = arch
        self._batch_size = 128
        self._lr = lr
        self._num_layers = len(self._arch)
        self._init_w()
        self._init_b()
        self.train_inputs = None
        self.train_targets = None
        self.training_set_size = 0
        self.input_batches = None
        self.target_batches = None


    def _init_w(self, low=0, high=1):
        ''' Initialize weights using gaussian distribution '''
        self._w = [np.random.normal(low, high, size=(j, i)) for i, j in zip(self._arch[:-1], self._arch[1:])]


    def _init_b(self, low=0, high=1):
        ''' Initialize biases using gaussian distribution '''
        self._b = [np.random.normal(low, high, size=(i, 1)) for i in self._arch[1:]]


    def _activate(self, z, inverse=False):
        ''' Sigmoid activation function '''
        return sigmoid(z, inverse)


    def _calculate_z(self, x, n):
        ''' Calculate raw output value z at layer n '''
        return np.dot(self._w[n], x) + self._b[n]


    def _propagate(self, x):
        ''' Calculate the activation and output values at each layer given
            input x '''
        self._a = []
        self._o = []
        self._a.append(x)
        self._o.append(x)
        tmp = x

        for l in range(self._num_layers-1):
            tmp = self._calculate_z(tmp, l)
            self._o.append(tmp)
            tmp = self._activate(tmp)
            self._a.append(tmp)

        return self._a, self._o


    def _backpropagate(self, y):
        ''' Propagate error signal backward from output layer '''
        self._d = []

        ''' Calculating the gradient for the output layer l = n '''
        out_d = np.multiply((self._a[-1] - y), 
                             self._activate(self._o[-1], inverse=True))
        self._d.append(out_d)

        ''' Calculating the gradient for all layers l = 0 ... n-1 '''
        for l in range(self._num_layers-2):
            d = np.multiply(np.dot((self._w[-(l+1)]).T, self._d[l]),
                            sigmoid(self._o[-(l+2)], inverse=True))
            self._d.append(d)

        ''' Reversal of the list of gradients '''
        self._d = self._d[::-1]
        return self._d


    def _adjust_weights(self):
        ''' Adjust weights according to gradients self._d '''
        self._w = [w - (self._lr / self._batch_size) * np.dot(d, a.T)
                   for w, d, a in zip(self._w, self._d, self._a)]
        return self._w


    def _adjust_biases(self):
        ''' Adjust biases according to gradients self._d '''
        self._b = [b - (self._lr / self._batch_size) * (np.sum(d, axis=1)).reshape(b.shape)
                   for b, d in zip(self._b, self._d)]
        return self._b


    def _prepare_train_inputs(self, images):
        ''' Arrange the inputs such that each column respresents the pixels of one image. '''
        self.train_inputs = images.T
        self.training_set_size = self.train_inputs.shape[1]


    def _prepare_train_targets(self, labels):
        ''' Arrange the targets such that each column represents the desired output for an image. '''
        labels = labels.T[0]
        self.train_targets = np.zeros((10, self.training_set_size))
        for i, label in enumerate(labels):
            self.train_targets[label][i] = 1


    def _shuffle_training_set(self):
        ''' Permute the inputs. Yields better results during training. '''
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


    def train_network(self, train_images, train_labels, epochs=50):
        self._prepare_train_inputs(train_images)
        self._prepare_train_targets(train_labels)

        for epoch in range(epochs):
            self._shuffle_training_set()
            self._batch_training_data(self._batch_size)

            for input_batch, target_batch in zip(self.input_batches, self.target_batches):
                self._propagate(input_batch[:,0])

            # Compute corrections
            print('Hello')

            # Backpropagate
            print('Hello')


def main():
    train_X, train_y, test_X, test_y = load_mnist_data()

    # Testing variables
    x = np.random.normal(0,1,size=(784,1))
    y = np.array((0,0,0,0,1,0,0,0,0,0)).reshape(10,1)
    
    # Test
    net = FFNet()
    a,o = net._propagate(x)
    d = net._backpropagate(y)
    net._adjust_weights()
    net._adjust_biases()

    print("Activation and Output shapes: ")
    for i in range(4):
        print(a[i].shape,o[i].shape)

    print("Delta shapes: ")
    for i in range(3):
        print(d[i].shape)

if __name__ == '__main__':
    main()

