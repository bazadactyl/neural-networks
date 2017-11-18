import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

non_pca_accuracy = 83.39

num_eigenfaces = [
    100, 60, 40, 20,
     14, 12, 10,  9,
      8,  7,  6,  5,
      4,  3,  2,  1,
]
pca_accuracy = [
    82.37, 80.36, 76.40, 70.34,
    62.50, 53.19, 51.32, 48.99,
    48.99, 48.06, 47.13, 46.74,
    44.10, 41.15, 41.15, 41.15,
]

fig = plt.figure()
ax = fig.gca()

plt.axhline(y=non_pca_accuracy, color='red')
plt.plot(num_eigenfaces, pca_accuracy)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.axis([100, 0, 30, 100])
plt.xlabel("Number of PCA Components")
plt.ylabel("Facial Recognition Accuracy (%)")
plt.legend(['Data without PCA (1850 features)', 'Data with X-many PCA components'], loc='upper right')

plt.show()
