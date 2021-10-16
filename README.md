# Neural Networks

We studied neural networks and applied them to text recognition and face recognition as part of the COMP4107 course at Carleton University.

## Text Recognition

### Using feed-forward neural networks

We used neural networks to recognize numbers from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

Check out the [write-up](a2/write-up.pdf).

Our best model achieved an accuracy of 96%:

![Chart of error rate for our best performing model](a2/plots/ff-bestest-performance-small.png)

### Experimenting with Hopfield networks

We experimented with using Hopfield networks to remember images using Storkey's and Hebb's learning rules for later retrieval.

See part 1 of [this write-up](a3/write-up/write-up.pdf).

#### Accuracy attained with Hebb's vs. Storkey's rules

![Accuracy attained with Hebb's vs. Storkey's rules](a3/figures/q1/accuracy.png)

#### Visualization of a Hopfield ntwork

![Visualization of a Hopfield Network](a3/figures/q1/hebb-network-weights.png)

#### Image restoration from a Hopfield network

![Image restoration from a Hopfield network](a3/figures/q1/hebb-recovery-01.png)

### Experimenting with self-organizing maps (SOM)

Continuing with the MNIST dataset, we experimented with using SOMs to recognize the hand-written numbers.

See part 2 of [this write-up](a3/write-up/write-up.pdf).

#### Visualizing the state of a SOM

![Visualizing the state of a SOM](a3/figures/q2/som-network-state-after-training.png)

#### K-means clustering of the MNIST data reduced to 2D

![K-means clustering of the MNIST data reduced to 2D](a3/figures/q2/kmeans-clustering-small-dots.png)

## Face Recognition

We used neural networks to recognize faces from the [LFW dataset](http://vis-www.cs.umass.edu/lfw/).

See part 3 of [this write-up](a3/write-up/write-up.pdf).

We were able to achieve an accuracy of 83% in recognizing faces using a regular feed-forward network.

We then used [principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) to reduce the size of the network and see how much accuracy it retains:

![Accuracy chart for face recognition](a3/figures/q3/pca-accuracy.png)

## General Image Recognition

Later in the course, we tackled the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which contains images of various animals and objects.

We used [convolutional neural networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) for this task.

We achieved an accuracy of 75%:

![CNN accuracy on the CIFAR-10 dataset](a4/figures/accuracy.png)

Check out [our write-up here](a4/write-up.ipynb).