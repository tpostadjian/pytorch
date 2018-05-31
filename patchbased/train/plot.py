import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', nargs='+', help='file list (read by numpy.loadtxt())')
parser.add_argument('-l', nargs='+', help='label list')
parser.add_argument('-a', nargs='+', help='axe names')
parser.add_argument('-c', nargs='+', help='color for each plot')

args = parser.parse_args()

files = args.f
labels = args.l
axes_label = args.a
colors = args.c

y = []
for f in files:
    y.append(np.loadtxt(f))

epochs = y[0].shape[0]
x = np.linspace(1, epochs, epochs)

fig = plt.figure()
plt.xlabel(axes_label[0])
plt.ylabel(axes_label[1])
for y_arr, label, color in zip(y, labels, colors):
    plt.plot(x, y_arr, '-', color=color, label=label)

plt.legend()
plt.show()
