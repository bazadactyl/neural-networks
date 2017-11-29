import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import cifar10

batch_size = 128
test_size = 256


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_1, w_2, w_fc, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X,
                                  w_1,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name="conv1"))

    l1 = tf.nn.max_pool(l1a,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name="pool1")

    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2 = tf.nn.relu(tf.nn.conv2d(l1,
                                 w_2,
                                 strides=[1, 1, 1, 1],
                                 padding="SAME",
                                 name="conv2"))

    l2 = tf.nn.max_pool(l2,
                        ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1],
                        padding="SAME",
                        name="pool2")

    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
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

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.0001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(X_train), batch_size),
                             range(batch_size, len(X_train) + 1, batch_size))

        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: X_train[start:end], Y: y_train[start:end],
                                          p_keep_conv: 0.6, p_keep_hidden: 0.9})

        test_indices = np.arange(len(X_test))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(("Epoch: %d" % i), np.mean(np.argmax(y_test[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: X_test[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
