import math
import sys
import time
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


def init_weights(shape):
    sigma = math.sqrt(2) * math.sqrt(2 / (1850 + 200 + 100))
    return tf.Variable(tf.random_normal(shape, stddev=sigma))


def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.leaky_relu(tf.matmul(X, w_h1), alpha=0.05)
    h2 = tf.nn.leaky_relu(tf.matmul(h1, w_h2), alpha=0.05)
    return tf.matmul(h2, w_o)

# Import the cached LFW people data
data = np.load('lfw/lfw_people_data.npy')
images = np.load('lfw/lfw_people_images.npy')
target = np.load('lfw/lfw_people_target.npy')
target_names = np.load('lfw/lfw_people_target_names.npy')

# Information about the dataset
num_samples = data.shape[0]
num_features = data.shape[1]
num_labels = target.max() + 1

# Perform PCA on the data on request
try:
    num_components = int(sys.argv[1])
    print("Extracting the top {} eigenfaces from {} faces".format(num_components, num_samples))
    pca = PCA(n_components=num_components, svd_solver='randomized', whiten=True).fit(data)
    eigenfaces = pca.components_.reshape((num_components, 50, 37))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    data = pca.transform(data)
    num_features = data.shape[1]
except (IndexError, ValueError):
    print("NOT PERFORMING PCA THIS RUN")
    print("To perform PCA:\n\tpython3 eigenfaces.py <num-pca-components>\n")
    time.sleep(2)
    data = data / data.max()  # regularize to interval [0, 1]
    data = (data * 2) - 1  # regularize to interval [-1, 1]

# Convert the target data to one-hot format
labels = np.zeros(shape=(num_samples, num_labels))
for i, x in enumerate(target):
    labels[i][x] = 1

# Set hidden layer sizes
size_h1 = tf.constant(160, dtype=tf.int32)
size_h2 = tf.constant(60, dtype=tf.int32)

# Set input and output layer sizes
X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, num_labels])

# Initialize weights
w_h1 = init_weights([num_features, size_h1])
w_h2 = init_weights([size_h1, size_h2])
w_o = init_weights([size_h2, num_labels])

# Training parameters
num_folds = 10
kf = KFold(n_splits=num_folds)
folds = list(kf.split(data))
batch_size = 64
epochs = 100
learn_rate = 0.020
fold_accuracy = []

# Define Tensorflow operations
py_x = model(X, w_h1, w_h2, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  # compute costs
train_op = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost)  # construct an optimizer
predict_op = tf.argmax(py_x, 1)

for fold in range(num_folds):
    print('Using fold {:02d} / {:02d} as the training fold:'.format(fold + 1, num_folds))

    train_indices, test_indices = folds[fold]
    trX, teX = data[train_indices], data[test_indices]
    trY, teY = labels[train_indices], labels[test_indices]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(1, epochs + 1):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

            if epoch % 20 == 0:
                epoch_accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))
                print('\t{} --- {:.2f}%'.format(epoch, epoch_accuracy * 100))

    fold_accuracy.append(epoch_accuracy)
    print('Accuracy with fold #{} as training: {:.2f}%\n'.format(fold + 1, fold_accuracy[-1] * 100))

accuracy = sum(fold_accuracy) / len(fold_accuracy)
print('Average {}-fold cross validation accuracy: {:.2f}%'.format(num_folds, accuracy * 100))
