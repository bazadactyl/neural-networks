import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

hebb_accuracy = [
    100.00, 100.00, 16.67, 42.50, 11.00,
    16.67, 7.14, 11.88, 7.78, 18.00,
    10.45, 7.50, 5.77, 6.07, 7.00,
    6.25, 8.24, 9.44, 3.95, 6.75
]
storkey_accuracy = [
    100.00, 92.50, 68.33, 51.25, 44.00,
    37.50, 34.29, 32.50, 27.78, 25.00,
    24.55, 25.00, 23.08, 20.71, 21.00,
    21.56, 17.65, 14.44, 16.58, 15.75
]

fig = plt.figure()
ax = fig.gca()

plt.plot(hebb_accuracy)
plt.plot(storkey_accuracy)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Number of patterns stored in the network")
plt.ylabel("Accuracy of Degraded Pattern Recovery (%)")
plt.legend(['Hebb Learning Rule', 'Storkey Learning Rule'], loc='upper right')

plt.show()
