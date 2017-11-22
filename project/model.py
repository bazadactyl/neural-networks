import tensorflow as tf
import numpy as np

# Define tensor constants
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 20
PREDICTIONS = 7 * 7 * 30
BATCH_SIZE = 1

# Layer definition convenience functions
class YOLONet:
    def __init__(self, X):
        model = self.build(X)

    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

        return var

    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        dtype = tf.float32

        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

    ''' Convenience function to generate a convolutional layer
            Args: input, filter, stride, name, layer
            Returns: output layer
    '''
    def convolution(self, input_, filter_, stride, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay('weights',
                                                      shape=filter_,
                                                      stddev=5e-2,
                                                      wd=0.0)
            conv = tf.nn.conv2d(input_, kernel, stride, padding)
            biases = self._variable_on_cpu('biases', [filter_[-1]],
                                      tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            output = tf.nn.relu(pre_activation, name=scope.name)

        return output

    ''' Convenience function to generate a fully connected layer
            Args: input, filter, stride, name, layer
            Returns: output layer
    '''
    def fully_connected(self, input_):
        with tf.variable_scope('fc1') as scope:
            n = 384

            reshape = tf.reshape(input_, [BATCH_SIZE, -1])
            dim = reshape.get_shape()[1].value

            weights = self._variable_with_weight_decay('weights', shape = [dim,n],
                                                                  stddev = 0.04,
                                                                  wd = 0.004)
            biases = self._variable_on_cpu('biases', [n], tf.constant_initializer(0.1))

            fc = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)

            return fc

    # Construct our model
    def build(self, images):

        # First set
        conv1 = self.convolution(images,
                                 filter_=[7, 7, 3, 64],
                                 stride=[1, 2, 2, 1],
                                 name="conv1")

        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool1")

        # Second set
        conv2 = self.convolution(pool1,
                                 filter_=[3, 3, 64, 192],
                                 stride=[1, 1, 1, 1],
                                 name="conv2")

        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool2")

        # Third set
        conv3 = self.convolution(pool2,
                                 filter_=[1, 1, 192, 128],
                                 stride=[1, 1, 1, 1],
                                 name="conv3")

        conv4 = self.convolution(conv3,
                                 filter_=[3, 3, 128, 256],
                                 stride=[1, 1, 1, 1],
                                 name="conv4")

        conv5 = self.convolution(conv4,
                                 filter_=[1, 1, 256, 256],
                                 stride=[1, 1, 1, 1],
                                 name="conv5")

        conv6 = self.convolution(conv5,
                                 filter_=[3, 3, 256, 512],
                                 stride=[1, 1, 1, 1],
                                 name="conv6")

        pool3 = tf.nn.max_pool(conv6,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool3")

        # Fourth set
        conv7 = self.convolution(pool3,
                                 filter_=[1, 1, 512, 256],
                                 stride=[1, 1, 1, 1],
                                 name="conv7")

        conv8 = self.convolution(conv7,
                                 filter_=[3, 3, 256, 512],
                                 stride=[1, 1, 1, 1],
                                 name="conv8")

        conv9 = self.convolution(conv8,
                                 filter_=[1, 1, 512, 256],
                                 stride=[1, 1, 1, 1],
                                 name="conv9")

        conv10 = self.convolution(conv9,
                                  filter_=[3, 3, 256, 512],
                                  stride=[1, 1, 1, 1],
                                  name="conv10")

        conv11 = self.convolution(conv10,
                                  filter_=[1, 1, 512, 256],
                                  stride=[1, 1, 1, 1],
                                  name="conv11")

        conv12 = self.convolution(conv11,
                                  filter_=[3, 3, 256, 512],
                                  stride=[1, 1, 1, 1],
                                  name="conv12")

        conv13 = self.convolution(conv12,
                                  filter_=[1, 1, 512, 256],
                                  stride=[1, 1, 1, 1],
                                  name="conv13")

        conv14 = self.convolution(conv13,
                                  filter_=[3, 3, 256, 512],
                                  stride=[1, 1, 1, 1],
                                  name="conv14")

        conv15 = self.convolution(conv14,
                                  filter_=[1, 1, 512, 512],
                                  stride=[1, 1, 1, 1],
                                  name="conv15")

        conv16 = self.convolution(conv15,
                                  filter_=[3, 3, 512, 1024],
                                  stride=[1, 1, 1, 1],
                                  name="conv16")

        pool4 = tf.nn.max_pool(conv16,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="SAME",
                               name="pool4")

        # Fifth set
        conv17 = self.convolution(pool4,
                                  filter_=[1, 1, 1024, 512],
                                  stride=[1, 1, 1, 1],
                                  name="conv17")

        conv18 = self.convolution(conv17,
                                  filter_=[3, 3, 512, 1024],
                                  stride=[1, 1, 1, 1],
                                  name="conv18")

        conv19 = self.convolution(conv18,
                                  filter_=[1, 1, 1024, 512],
                                  stride=[1, 1, 1, 1],
                                  name="conv19")

        conv20 = self.convolution(conv19,
                                  filter_=[3, 3, 512, 1024],
                                  stride=[1, 1, 1, 1],
                                  name="conv20")

        conv21 = self.convolution(conv20,
                                  filter_=[3, 3, 1024, 1024],
                                  stride=[1, 1, 1, 1],
                                  name="conv21")

        conv22 = self.convolution(conv21,
                                  filter_=[3, 3, 1024, 1024],
                                  stride=[1, 2, 2, 1],
                                  name="conv22")

        # Sixth set
        conv23 = self.convolution(conv22,
                                  filter_=[3, 3, 1024, 1024],
                                  stride=[1, 1, 1, 1],
                                  name="conv23")

        conv24 = self.convolution(conv23,
                                  filter_=[3, 3, 1024, 1024],
                                  stride=[1, 1, 1, 1],
                                  name="conv24")

        # Fully-connected layers
        with tf.variable_scope('fc') as scope:
            n = 4096

            reshape = tf.reshape(conv24, [BATCH_SIZE, -1])
            dim = reshape.get_shape()[1].value

            weights = self._variable_with_weight_decay('weights',
                                                       shape=[dim, 4096],
                                                       stddev=0.04,
                                                       wd=0.004)

            biases = self._variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))

            fc = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # Softmax layer
        with tf.variable_scope('softmax') as scope:
            weights = self._variable_with_weight_decay('weights',
                                                       shape=[4096, 7 * 7 * 30],
                                                       stddev=1 / 4096,
                                                       wd=0.0)

            biases = self._variable_on_cpu('biases', 7 * 7 * 30, tf.constant_initializer(0.0))

            softmax = tf.add(tf.matmul(fc, weights), biases, name=scope.name)

        return softmax

def main():
    x = tf.random_normal([2000, 224, 224, 3], mean=-1, stddev=4, dtype=tf.float32)
    YOLONet(x)

if __name__ == '__main__':
    main()