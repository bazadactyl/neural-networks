import tensorflow as tf
import numpy as np

# Define tensor constants
IMAGE_SIZE = (224,224)
NUM_CLASSES = 20
PREDICTIONS = 7*7*30

# Define training constants

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

		var = _variable_on_cpu(
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
	def convolution(self, input, filter, stride, name, padding='SAME'):
		with tf.variable_scope(name) as scope:
			kernel = _variable_with_weight_decay('weights',
												 shape=shape,
												 stddev=5e-2,
												 wd=0.0)
			conv = tf.nn.conv2d(input, kernel, stride, padding)
			biases = _variable_on_cpu('biases', 
									  [shape[-1]],
									  tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			output = tf.nn.relu(pre_activation, name=scope.name)

		return output

	''' Convenience function to generate a fully connected layer
			Args: input, filter, stride, name, layer
			Returns: output layer
	'''
	def fully_connected(self, input):
		with tf.variable_scope('fc1') as scope:
			reshape = tf.reshape(input, [FLAGS.batch_size, -1])
			dim = reshape.get_shape()[1].value

			weights = _variable_with_weight_decay('weights',
												  shape=[dim,?],
												  stddev=0.04,
												  wd=0.004)
			biases = _variable_on_cpu('biases', [?], 
												tf.constant_initializer(0.1))
			fc = tf.nn.relu(tf.matmul(reshape, weights) + biases,
							name=scope.name)

			return fc

	# Construct our model
	def build(self, images):

		# First set
		conv1 = self.convolution(images, filter=[7,7,3,64], 
										 stride=[1,2,2,1],
										 name="conv1")

		pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],
									  stride=[1,2,2,1],
									  name="pool1")

		# Second set
		# TODO: FIX INCORRECT INPUT SHAPE TO ADJUST FOR CONV1 AND POOL1
		conv2 = self.convolution(pool1, filter=[3,3,64,192],
										stride=[1,1,1,1],
										name="conv2")

		pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1],
									  stride=[1,2,2,1],
									  name="pool2")

		# Third set
		# TODO: FIX INCORRECT INPUT SHAPE TO ADJUST FOR POOL2
		conv3 = self.convolution(pool2, filter=[1,1,192,128],
										stride=[1,1,1,1],
										name="conv3")

		conv4 = self.convolution(conv3, filter=[3,3,128,256],
										stride=[1,1,1,1],
										name="conv4")

		conv5 = self.convolution(conv4, filter=[1,1,256,256],
										stride=[1,1,1,1],
										name="conv5")

		conv6 = self.convolution(conv5, filter=[3,3,256,512],
										stride=[1,1,1,1],
										name="conv6")

		pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1],
									  stride=[1,2,2,1],
									  name="pool3")

		# Fourth set
		# TODO: FIX INCORRECT INPUT SHAPE TO ADJUST FOR POOL3
		conv7 = self.convolution(pool3, filter=[1,1,512,256],
										stride=[1,1,1,1],
										name="conv7")

		conv8 = self.convolution(conv7, filter=[3,3,256,512],
										stride=[1,1,1,1],
										name="conv8")

		conv9 = self.convolution(conv8, filter=[1,1,512,256],
										stride=[1,1,1,1],
										name="conv9")

		conv10 = self.convolution(conv9, filter=[3,3,256,512],
										 stride=[1,1,1,1],
										 name="conv10")

		conv11 = self.convolution(conv10, filter=[1,1,512,256],
										  stride=[1,1,1,1],
										  name="conv11")

		conv12 = self.convolution(conv11, filter=[3,3,256,512],
										  stride=[1,1,1,1],
										  name="conv12")

		conv13 = self.convolution(conv12, filter=[1,1,512,256],
										  stride=[1,1,1,1],
										  name="conv13")

		conv14 = self.convolution(conv13, filter=[3,3,256,512],
										  stride=[1,1,1,1],
										  name="conv14")

		conv15 = self.convolution(conv14, filter=[1,1,512,512],
										  stride=[1,1,1,1],
										  name="conv15")

		conv16 = self.convolution(conv15, filter=[3,3,512,1024],
										  stride=[1,1,1,1],
										  name="conv16")

		pool4 = tf.nn.max_pool(conv16, ksize=[1,2,2,1],
									   stride=[1,2,2,1],
									   name="pool4")

		# Fifth set
		# TODO: FIX INCORRECT INPUT SHAPE TO ADJUST FOR POOL4
		conv17 = self.convolution(pool4, filter=[1,1,1024,512],
										 stride=[1,1,1,1],
										 name="conv17")

		conv18 = self.convolution(conv17, filter=[3,3,512,1024],
										  stride=[1,1,1,1],
										  name="conv18")

		conv19 = self.convolution(conv18, filter=[1,1,1024,512],
										 stride=[1,1,1,1],
										 name="conv19")

		conv20 = self.convolution(conv19, filter=[3,3,512,1024],
										  stride=[1,1,1,1],
										  name="conv20")

		conv21 = self.convolution(conv20, filter=[3,3,1024,1024],
										  stride=[1,1,1,1],
										  name="conv21")

		conv22 = self.convolution(conv21, filter=[3,3,1024,1024],
										  stride=[1,2,2,1],
										  name="conv22")

		# Sixth set
		# TODO: FIX INCORRECT INPUT SHAPE TO ADJUST FOR CONV22
		conv23 = self.convolution(conv22, filter=[3,3,1024,1024],
										  stride=[1,1,1,1],
										  name="conv23")

		conv24 = self.convolution(conv23, filter=[3,3,1024,1024],
										  stride=[1,1,1,1],
										  name="conv24")

		# Fully-connected layers
		# TODO: Determine size of the nth dimension here
		#		Add another FC layer
		with tf.variable_scope('fc') as scope:
			n = ?

			reshape = tf.reshape(conv24, [FLAGS.batch_size, -1])
			dim = reshape.get_shape()[1].value

			weights = _variable_with_weight_decay('weights',
												  shape=[dim,n],
												  stddev=0.04,
												  wd=0.004)
			biases = _variable_on_cpu('biases', [n], 
												tf.constant_initializer(0.1))
			
			fc = tf.nn.relu(tf.matmul(reshape, weights) + biases,
							name=scope.name)

		# Softmax layer
		# TODO: Determine size of the nth dimension here
		with tf.variable_scope('softmax') as scope:
			n = ?

			weights = _variable_with_weight_decay('weights',
												  shape=[n,7*7*30],
												  stddev=1/n,
												  wd=0.0)

			biases = _variable_on_cpu('biases', 7*7*30, 
									  tf.constant_initializer(0.0))

			softmax = tf.add(tf.matmul(fc, weights), 
							 biases, name=scope.name)

		return softmax