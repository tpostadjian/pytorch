import numpy as np
import matplotlib.pyplot as plt


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


f = open("kappa_girondeFT.txt", "r")
rows = 23
cols = 22
dilat = 1
img = np.zeros((rows * dilat, cols * dilat), dtype=np.float)
list = []
while True:
    line = f.readline().rstrip('\n')
    if line == '':
        break
    for i in range(0, dilat * dilat):
        list.append(line)

for i in range(0, rows * dilat):
    for j in range(0, cols * dilat):
        if is_number(list[i + j * rows * dilat]):
            img[i, j] = float(list[i + j * rows * dilat])

v = np.linspace(0., 1., 10, endpoint=True)
plt.axis("off")
plt.imshow(img, interpolation='none', cmap='hot')
plt.colorbar()
plt.savefig('kappa_girondeFT.png')
