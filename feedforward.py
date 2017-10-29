import matplotlib.pyplot as plt
import numpy as np
from load_mnist_data import load_mnist_data


def sigmoid(x, inverse=False):
    result = 1 / (1 + np.exp(-x))

    if not inverse:
        return result
    else:
        return result * (1 - result)


class FeedForwardNetwork:
    def __init__(self, arch=np.array([784, 50, 30, 10]), lr=1.0, batch_size=300):
        self._arch = arch
        self.batch_size = batch_size
        self._lr = lr
        self._num_layers = len(self._arch)
        self._init_w()
        self._init_b()
        self.train_inputs = None
        self.train_targets = None
        self.training_set_size = 0
        self.input_batches = None
        self.target_batches = None

    def _init_w(self, low=-1, high=1):
        ''' Initialize weights using gaussian distribution '''
        # self._w = [np.random.normal(low, high, size=(j, i)) for i, j in zip(self._arch[:-1], self._arch[1:])]
        self._w = [np.random.randn(j, i) for i, j in zip(self._arch[:-1], self._arch[1:])]

    def _init_b(self, low=-1, high=1):
        ''' Initialize biases using gaussian distribution '''
        # self._b = [np.random.normal(low, high, size=(i, 1)) for i in self._arch[1:]]
        self._b = [np.random.randn(i, 1) for i in self._arch[1:]]

    def _calculate_z(self, x, n):
        ''' Calculate raw output value z at layer n '''
        return np.dot(self._w[n], x) + self._b[n]

    def _propagate(self, x, return_label=False):
        """ Calculate the activation and output values at each layer given
            input x """
        self._a = []
        self._o = []
        self._a.append(x)
        self._o.append(x)
        tmp = x

        for l in range(self._num_layers-1):
            tmp = self._calculate_z(tmp, l)
            self._o.append(tmp)
            tmp = sigmoid(tmp)
            self._a.append(tmp)

        if return_label:
            return tmp
        else:
            return self._a, self._o

    def _backpropagate(self, y):
        """ Propagate error signal backward from output layer """
        self._d = []

        ''' Calculating the gradient for the output layer l = n '''
        out_d = np.multiply(
            self._a[-1] - y,
            sigmoid(self._o[-1], inverse=True)
        )
        self._d.append(out_d)

        ''' Calculating the gradient for all layers l = 0 ... n-1 '''
        for l in range(self._num_layers-2):
            weights_l = (self._w[-(l+1)]).T
            delta_l = self._d[l]
            d = np.multiply(np.dot(weights_l, delta_l), sigmoid(self._o[-(l+2)], inverse=True))
            self._d.append(d)

        ''' Reversal of the list of gradients '''
        self._d = self._d[::-1]
        return self._d

    def _adjust_weights(self):
        """ Adjust weights according to gradients self._d """
        learn_rate = self._lr / self.batch_size
        self._w = [w - learn_rate * np.dot(d, a.T)
                   for w, d, a in zip(self._w, self._d, self._a)]
        return self._w

    def _adjust_biases(self):
        """ Adjust biases according to gradients self._d """
        learn_rate = self._lr / self.batch_size
        self._b = [b - learn_rate * (np.sum(d, axis=1)).reshape(b.shape)
                   for b, d in zip(self._b, self._d)]
        return self._b

    def _prepare_train_inputs(self, images):
        """ Arrange the inputs such that each column represents the pixels of one image. """
        self.train_inputs = images.T
        self.training_set_size = self.train_inputs.shape[1]

    def _prepare_train_targets(self, labels):
        """ Arrange the targets such that each column represents the desired output for an image. """
        labels = labels.T[0]
        self.train_targets = np.zeros((10, self.training_set_size))
        for i, label in enumerate(labels):
            self.train_targets[label][i] = 1

    @staticmethod
    def _prepare_y(y):
        # Re-implementation of _prepare_train_targets to match our current data format
        a = np.zeros((y.shape[0], 10, 1))

        for i, label in enumerate(y):
            a[i][label] = 1
            a[i] = a[i].reshape(10,1)

        return a

    @staticmethod
    def shuffle_training_set(inputs, labels, labels_onehot):
        ''' Permute the inputs. Yields better results during training. '''
        num_examples = inputs.shape[1]
        permutation = np.random.permutation(num_examples)
        shuf_inputs = inputs.T[permutation].T
        shuf_labels = labels.T[permutation].T
        shuf_labels_onehot = labels_onehot.T[permutation].T
        return shuf_inputs, shuf_labels, shuf_labels_onehot

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

    def train_network(self, train_images, train_labels, epochs=500):
        self._prepare_train_inputs(train_images)
        self._prepare_train_targets(train_labels)

        for epoch in range(epochs):
            self.shuffle_training_set()
            self._batch_training_data(self.batch_size)

            for input_batch, target_batch in zip(self.input_batches, self.target_batches):
                self._propagate(input_batch[:,0])

            # Compute corrections
            print('Hello')

            # Backpropagate
            print('Hello')

    def test_network(self, test_input, test_labels):
        acc = []

        output_labels = self._propagate(test_input, return_label=True).T
        num_images = output_labels.shape[0]

        for i in range(num_images):
            output = np.argmax(output_labels[i])
            if output == test_labels[i]:
                acc.append(1)

        return float(len(acc)) / float(num_images)


def main():
    net = FeedForwardNetwork()

    train_x, train_y, test_x, test_y = load_mnist_data()

    train_x = train_x.reshape(60000, 784).T
    train_y = train_y.T[0]

    test_x = test_x.reshape(10000, 784).T
    test_y = test_y.T[0]

    train_y_onehot = np.zeros((10, 60000))
    for i, label in enumerate(train_y):
        train_y_onehot[label][i] = 1

    num_training_examples = train_x.shape[1]

    # iterations = (num_training_examples - (num_training_examples % net.batch_size)) / net.batch_size
    # epochs = 100

    pre_accuracy = net.test_network(test_x, test_y) * 100
    print("Testing accuracy pre-training on the 10,000 element test set: {:.2f}".format(pre_accuracy))

    num_folds = int(num_training_examples / net.batch_size)
    print("Performing {}-fold cross validation while training on the 60,000 element train set".format(num_folds))

    train_error = []
    test_error = []
    cv_error = []

    for k in range(num_folds):
        # Deep copy the training set because we want to manipulate it
        x = np.copy(train_x)
        y = np.copy(train_y).reshape(1, 60000)
        y_oh = np.copy(train_y_onehot)

        # Indices of the cross-validation (test) examples
        cv_start = net.batch_size * k
        cv_end = net.batch_size * (k + 1)

        # Get the cross-validation (test) examples
        cv_columns = [x for x in range(cv_start, cv_end)]
        cv_x = x[:, cv_columns]
        cv_y = y[:, cv_columns]
        cv_y_oh = y_oh[:, cv_columns]

        # Remove cross-validation (test) examples from the training set
        x = np.delete(x, cv_columns, 1)
        y = np.delete(y, cv_columns, 1)
        y_oh = np.delete(y_oh, cv_columns, 1)

        # Shuffle the training examples for better results
        x, y, y_oh = net.shuffle_training_set(x, y, y_oh)

        training_batches = filter(lambda fold: fold != k, range(num_folds))
        for i in training_batches:
            batch_start = net.batch_size * i
            batch_end = net.batch_size * (i + 1)

            x_batch = np.array([row[batch_start:batch_end] for row in x])
            y_batch = np.array([row[batch_start:batch_end] for row in y_oh])

            a, o = net._propagate(x_batch)
            d = net._backpropagate(y_batch)
            net._adjust_weights()
            net._adjust_biases()

        cv_y = cv_y.reshape(net.batch_size,)
        cv_acc = net.test_network(cv_x, cv_y) * 100

        train_acc = net.test_network(train_x, train_y) * 100
        test_acc = net.test_network(test_x, test_y) * 100
        print("Fold #{}".format(k+1))
        print("\tCross-validation Accuracy: {:.2f}%".format(cv_acc))
        print("\t    Training Set Accuracy: {:.2f}%".format(train_acc))
        print("\t     Testing Set Accuracy: {:.2f}%".format(test_acc))

        train_error.append(100.0 - train_acc)
        test_error.append(100.0 - test_acc)
        cv_error.append(100.0 - cv_acc)

    print("\nFinished training with {}-fold cross-validation!".format(num_folds))
    print("\tLearning rate: {:.4f}".format(net._lr / net.batch_size))
    print("\tExamples per fold: {}".format(net.batch_size))
    print("Average cross-validation error: {:.2f}%".format(sum(cv_error) / num_folds))

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 0.9, 3)])
    plt.plot(train_error)
    plt.plot(test_error)
    plt.plot(cv_error)
    plt.xlabel("Fold Number")
    plt.ylabel("Error Rate (%)")
    plt.legend([
        'Training Set (all 60000 examples)',
        'Test Set (all 10000 examples)',
        'Cross-validation Set (300 per fold)',
    ], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()

