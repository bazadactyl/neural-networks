import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import cifar10

batch_size = 256
test_size = 256
epochs = 200
logs_path = '/tmp/tensorboard_a4/'


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def _conv2d(X, weights, name, strides=[1,1,1,1], padding='SAME'):
    with tf.name_scope(name) as scope:
        return tf.nn.conv2d(X,
                            weights,
                            strides=strides,
                            padding=padding,
                            name=name)

def _relu(X, name):
    with tf.name_scope(name) as scope:
        return tf.nn.relu(X)

def _max_pool(X, name, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME'):
    with tf.name_scope(name) as scope:
        return tf.nn.max_pool(X,
                              ksize=ksize,
                              strides=strides,
                              padding=padding,
                              name=name)

def _dropout(X, sigma, name):
    with tf.name_scope(name) as scope:
        return tf.nn.dropout(X, sigma)

def _matmul(X, y, name):
    with tf.name_scope(name) as scope:
        return tf.matmul(X, y)

def model(X, w_1, w_2, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = _conv2d(X, w_1, name='conv1')
    l1 = _relu(l1a, name='relu1')
    l1 = _max_pool(l1, name='pool1')
    l1 = _dropout(l1, sigma=p_keep_conv, name='dropout1')

    l2 = _conv2d(l1, w_2, name='conv2')
    l2 = _relu(l2, name='relu2')
    l2 = _max_pool(l2, name='pool2')
    l2 = _dropout(l2, sigma=p_keep_conv, name='dropout2')

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])
    l3 = _dropout(l3, sigma=p_keep_conv, name='dropout3')

    l4 = _matmul(l3, w_fc, name='matmul1')
    l4 = _relu(l4, name='relu3')
    l4 = _dropout(l4, sigma=p_keep_hidden, name='dropout4')

    pyx = _matmul(l4, w_o, name='pyx')
    return pyx


encoder = OneHotEncoder()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = encoder.fit(y_train).transform(y_train).toarray()
y_test = encoder.fit(y_test).transform(y_test).toarray()

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w_1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
w_2 = init_weights([3, 3, 32, 64])
w_fc = init_weights([64 * 8 * 8, 625])  # FC 32 * 14 * 14 inputs, 625 outputs
w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_1, w_2, w_fc, w_o, p_keep_conv, p_keep_hidden)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

with tf.name_scope("train_op") as scope:
    train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)

with tf.name_scope("predict_op") as scope:
    predict_op = tf.argmax(py_x, 1)
    tf.summary.scalar("predict_op", predict_op)

with tf.Session() as sess:
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, sess.graph)
    tf.global_variables_initializer().run()

    for i in range(epochs):
        batch_starts = range(0, len(X_train), batch_size)
        batch_ends = range(batch_size, len(X_train) + 1, batch_size)
        training_batches = zip(batch_starts, batch_ends)

        for start, end in training_batches:
            train_graph = sess.run(train_op, feed_dict={ 
                X: X_train[start:end],
                Y: y_train[start:end],
                p_keep_conv: 0.6,
                p_keep_hidden: 0.9, 
            })

        test_indices = np.arange(len(X_test))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        labels = np.argmax(y_test[test_indices], axis=1)
        predictions = sess.run(predict_op, feed_dict={
            X: X_test[test_indices],
            p_keep_conv: 1.0,
            p_keep_hidden: 1.0
        })

        accuracy = np.mean(labels == predictions)
        print("Epoch: {}  Accuracy: {}".format(i + 1, accuracy))
