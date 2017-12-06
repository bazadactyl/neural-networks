import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

accuracy = [45.70, 46.09, 51.95, 53.12, 62.89, 66.40, 58.98, 66.40, 60.15, 67.96, 69.14, 69.92, 70.31, 72.26, 75.39]
epoch = list(range(1, len(accuracy) + 1))

fig = plt.figure()
ax = fig.gca()

plt.plot(epoch, accuracy)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Classification Accuracy (%)")
plt.title("CNN Classifier for the CIFAR-10 Dataset")

plt.show()
